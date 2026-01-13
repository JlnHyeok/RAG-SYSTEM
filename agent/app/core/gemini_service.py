import logging
from typing import Optional
import google.generativeai as genai
import asyncio
from functools import lru_cache

from app.core.config import settings

logger = logging.getLogger(__name__)


class GeminiService:
    """Google Gemini LLM 서비스 클래스"""
    
    def __init__(self):
        self.model = None
        self._initialized = False
        
    async def initialize(self, test_connection: bool = False):
        """Gemini API 초기화"""
        try:
            if not settings.GEMINI_API_KEY:
                logger.warning("GEMINI_API_KEY가 설정되지 않음. Gemini 서비스 비활성화")
                self._initialized = False
                return
            
            # Gemini API 설정
            genai.configure(api_key=settings.GEMINI_API_KEY)
            
            # 모델 초기화 (환경변수에서 모델명 가져오기, 없으면 기본값 사용)
            model_name = settings.GEMINI_MODEL or "gemini-2.0-flash-exp"
            self.model = genai.GenerativeModel(model_name)
            
            # 연결 테스트 (선택적)
            if test_connection:
                try:
                    await self._test_connection()
                except Exception as e:
                    if "quota" in str(e).lower() or "429" in str(e):
                        logger.warning(f"Gemini API 할당량 초과, 연결 테스트 건너뛰기: {e}")
                    else:
                        logger.error(f"Gemini API 연결 테스트 실패: {e}")
                        raise
            
            self._initialized = True
            logger.info("Gemini API 초기화 완료")
            
        except Exception as e:
            if "quota" in str(e).lower() or "429" in str(e):
                logger.warning(f"Gemini API 할당량 초과, 기본 설정으로 초기화: {e}")
                # 기본 설정으로라도 초기화
                genai.configure(api_key=settings.GEMINI_API_KEY)
                model_name = settings.GEMINI_MODEL or "gemini-2.0-flash-exp"
                self.model = genai.GenerativeModel(model_name)
                self._initialized = True
                logger.info("Gemini API 기본 초기화 완료 (연결 테스트 미실행)")
            else:
                logger.error(f"Gemini API 초기화 실패: {e}")
                raise
    
    async def _test_connection(self):
        """Gemini API 연결 테스트"""
        try:
            test_prompt = "안녕하세요. 연결 테스트입니다."
            response = await asyncio.to_thread(
                self.model.generate_content,
                test_prompt
            )
            
            if not response.text:
                raise Exception("Gemini API 응답이 비어있습니다")
                
            logger.info("Gemini API 연결 테스트 성공")
            
        except Exception as e:
            logger.error(f"Gemini API 연결 테스트 실패: {e}")
            raise
    
    def _is_quota_exceeded(self, error: Exception) -> bool:
        """할당량 초과 에러 여부 확인"""
        error_str = str(error).lower()
        return "quota" in error_str or "429" in error_str
    
    def _create_generation_config(self, max_tokens: int, temperature: float) -> genai.types.GenerationConfig:
        """일관된 GenerationConfig 생성 - 더 완전한 답변을 위한 설정"""
        return genai.types.GenerationConfig(
            max_output_tokens=max_tokens,
            temperature=temperature,
            top_p=0.95,  # 더 다양한 응답을 위해
            top_k=40,   # 토큰 선택 범위 확장
            candidate_count=1,  # 하나의 완전한 답변 생성
            stop_sequences=[]   # 중단 시퀀스 없음으로 완전한 답변 보장
        )
    
    @lru_cache(maxsize=100)
    def _cached_generate(self, prompt_hash: str, prompt: str) -> str:
        """자주 사용되는 프롬프트 캐싱"""
        if not self._initialized:
            raise RuntimeError("Gemini 서비스가 초기화되지 않았습니다")
            
        response = self.model.generate_content(prompt)
        return response.text
    
    async def generate_answer(
        self,
        question: str,
        context: str,
        max_tokens: int = 2000,  # 기본값을 1000에서 2000으로 증가
        temperature: float = 0.1
    ) -> str:
        """컨텍스트 기반 답변 생성"""
        try:
            if not self._initialized:
                await self.initialize()
            
            # 프롬프트 구성
            prompt = self._build_rag_prompt(question, context)
            
            # 짧은 프롬프트는 캐시 사용
            if len(prompt) < 500:
                prompt_hash = str(hash(prompt))
                try:
                    return self._cached_generate(prompt_hash, prompt)
                except Exception as e:
                    if self._is_quota_exceeded(e):
                        return self._get_quota_exceeded_response(question, context)
                    raise
            
            # 긴 프롬프트는 비동기 처리
            def generate():
                response = self.model.generate_content(
                    prompt,
                    generation_config=self._create_generation_config(max_tokens, temperature)
                )
                return response.text
            
            return await asyncio.to_thread(generate)
            
        except Exception as e:
            logger.error(f"Gemini 답변 생성 실패: {e}")
            if self._is_quota_exceeded(e):
                return self._get_quota_exceeded_response(question, context)
            return self._get_fallback_response(e)
    
    async def generate_with_system_prompt(
        self,
        system_prompt: str,
        user_message: str,
        max_tokens: int = 500,
        temperature: float = 0.7
    ) -> str:
        """시스템 프롬프트와 함께 답변 생성 (일반 대화용)"""
        try:
            if not self._initialized:
                await self.initialize()
            
            # 시스템 프롬프트 + 사용자 메시지 조합
            full_prompt = f"""
{system_prompt}

사용자 질문: {user_message}

답변:"""
            
            def generate():
                response = self.model.generate_content(
                    full_prompt,
                    generation_config=self._create_generation_config(max_tokens, temperature)
                )
                return response.text
            
            return await asyncio.to_thread(generate)
            
        except Exception as e:
            logger.error(f"Gemini 일반 대화 생성 실패: {e}")
            
            # 할당량 초과 시 기본 응답
            if self._is_quota_exceeded(e):
                return self._get_intelligent_fallback_response(user_message, quota_exceeded=True)
            
            # 기타 에러 시 기본 응답 반환 (할당량 초과가 아닌 경우)
            return self._get_intelligent_fallback_response(user_message, quota_exceeded=False)
    
    def _build_rag_prompt(self, question: str, context: str) -> str:
        """RAG용 프롬프트 템플릿 구성"""
        return f"""당신은 도움이 되는 AI 어시스턴트입니다. 주어진 컨텍스트를 바탕으로 사용자의 질문에 정확하고 완전한 답변을 제공해주세요.

중요한 규칙:
1. 컨텍스트에 있는 정보만을 사용해서 답변하세요
2. 컨텍스트에 없는 정보는 추측하지 말고, 모르겠다고 답변하세요
3. 답변은 한국어로 작성하세요
4. 가능하면 구체적인 근거를 제시하세요
5. 출처 정보가 있다면 언급해주세요
6. 답변을 완전히 끝까지 작성하세요 - 중간에 끊지 마세요
7. 표나 목록이 있다면 모든 항목을 포함하세요
8. 상세하고 완성된 답변을 제공하세요

컨텍스트:
{context}

질문: {question}

답변 (완전하고 상세하게 작성):"""
    
    async def generate_summary(self, text: str, max_length: int = 200) -> str:
        """텍스트 요약 생성"""
        try:
            if not self._initialized:
                await self.initialize()
            
            prompt = f"""다음 텍스트를 {max_length}자 이내로 요약해주세요. 핵심 내용을 놓치지 말고 간결하게 정리하세요.

텍스트:
{text}

요약:"""
            
            def generate():
                response = self.model.generate_content(
                    prompt,
                    generation_config=self._create_generation_config(max_length * 2, 0.3)  # 한국어 특성상 여유분, 낮은 temperature
                )
                return response.text
                
            return await asyncio.to_thread(generate)
            
        except Exception as e:
            logger.error(f"요약 생성 실패: {e}")
            return text[:max_length] + "..."
    
    async def generate_keywords(self, text: str, max_keywords: int = 5) -> list:
        """텍스트에서 키워드 추출"""
        try:
            if not self._initialized:
                await self.initialize()
            
            prompt = f"""다음 텍스트에서 가장 중요한 키워드 {max_keywords}개를 추출해주세요. 각 키워드는 쉼표로 구분하여 나열하세요.

텍스트:
{text}

키워드:"""
            
            def generate():
                response = self.model.generate_content(prompt)
                keywords_text = response.text.strip()
                return [kw.strip() for kw in keywords_text.split(',')]
                
            return await asyncio.to_thread(generate)
            
        except Exception as e:
            logger.error(f"키워드 추출 실패: {e}")
            return []
    
    def _get_fallback_response(self, error: Exception) -> str:
        """오류 발생 시 기본 응답"""
        error_str = str(error).lower()
        if "quota" in error_str or "429" in error_str:
            return "현재 Gemini API 할당량을 초과했습니다. 잠시 후 다시 시도해주세요."
        elif "rate limit" in error_str:
            return "현재 서비스 사용량이 많아 잠시 후 다시 시도해주세요."
        elif "api key" in error_str:
            return "API 인증 문제가 발생했습니다. 관리자에게 문의하세요."
        else:
            return "답변 생성 중 오류가 발생했습니다. 다시 시도해주세요."
    
    def _get_quota_exceeded_response(self, question: str, context: str) -> str:
        """할당량 초과 시 컨텍스트 포함 답변"""
        if not context or len(context.strip()) == 0:
            return "현재 Gemini API 할당량을 초과했습니다. 잠시 후 다시 시도해주세요."
        
        # 컨텍스트를 구조화하여 표시
        context_lines = context.strip().split('\n')
        formatted_parts = []
        
        current_doc = ""
        current_content = []
        
        for line in context_lines:
            if line.startswith('[문서'):
                # 이전 문서 저장
                if current_doc and current_content:
                    content_text = '\n'.join(current_content).strip()
                    if content_text:
                        formatted_parts.append(f"**{current_doc}**\n{content_text}")
                
                # 새 문서 시작
                current_doc = line.strip('[]')
                current_content = []
            elif line.strip():
                current_content.append(line)
        
        # 마지막 문서 저장
        if current_doc and current_content:
            content_text = '\n'.join(current_content).strip()
            if content_text:
                formatted_parts.append(f"**{current_doc}**\n{content_text}")
        
        result = "현재 Gemini API 할당량을 초과했습니다.\n\n"
        result += f"'{question}' 질문과 관련하여 다음 문서 내용을 찾았습니다:\n\n"
        
        if formatted_parts:
            result += "\n\n".join(formatted_parts)
        else:
            result += context[:800] + ("..." if len(context) > 800 else "")
        
        return result
    
    def _get_intelligent_fallback_response(self, user_message: str, quota_exceeded: bool = False) -> str:
        """지능적인 fallback 응답 생성"""
        import re
        
        message_lower = user_message.lower()
        quota_msg = " 현재 API 할당량 초과로 제한적이지만," if quota_exceeded else ""
        
        # 인사 관련 패턴
        greeting_patterns = ['안녕', '하이', '헬로', '반가', '처음', '시작']
        if any(pattern in message_lower for pattern in greeting_patterns):
            return f"안녕하세요! 저는 RAG 기반 AI 어시스턴트입니다.{quota_msg} 문서를 업로드하시면 관련 질문에 답변해드릴 수 있어요."
        
        # 정체성/소개 관련 패턴  
        identity_patterns = ['누구', '뭐야', '뭐하는', '어떤', '소개', '자기소개', '정체', '이름']
        if any(pattern in message_lower for pattern in identity_patterns):
            return f"저는 RAG(Retrieval Augmented Generation) 기반 AI 어시스턴트입니다.{quota_msg} 문서를 분석하고 질문에 답변하는 것이 주 기능이에요!"
        
        # 기능/능력 관련 패턴 (더 포괄적으로)
        function_patterns = ['기능', '할 수 있', '능력', '무엇을', '어떻게', '방법', '도움', '지원', '서비스', 
                           '할수있', '가능한', '제공', '특징', '장점', '용도', '역할', '일', '업무']
        if any(pattern in message_lower for pattern in function_patterns):
            quota_note = "\n\n현재 API 할당량 초과이지만 문서 업로드 후 질문해보세요!" if quota_exceeded else "\n\n문서를 업로드하고 관련 질문을 해보세요!"
            return f"""저의 주요 기능은 다음과 같습니다:

1. 📄 문서 업로드 및 분석 (PDF, Word, 텍스트 등)
2. 🔍 업로드된 문서에서 정보 검색 및 질의응답
3. 🌏 다국어 문서 처리 지원 (한국어, 영어 등)
4. 👁️ OCR을 통한 이미지 내 텍스트 추출
5. 🎯 벡터 검색 기반 유사도 매칭
6. ⚡ 실시간 스트리밍 응답{quota_note}"""
        
        # 사용법/방법 관련 패턴
        usage_patterns = ['사용', '이용', '활용', '시작', '설정', '설치', '실행', '작동', '운영']
        if any(pattern in message_lower for pattern in usage_patterns):
            return f"""사용 방법은 간단합니다:

1. 📤 문서 업로드: PDF, Word, 텍스트 파일을 시스템에 업로드
2. ❓ 질문하기: 업로드한 문서에 관련된 질문 입력
3. 💬 답변 받기: AI가 문서를 분석하여 정확한 답변 제공
4. 🔄 실시간 대화: 추가 질문으로 더 깊이 있는 정보 탐색{quota_msg}"""
        
        # 파일/문서 관련 패턴
        file_patterns = ['파일', '문서', '업로드', '올리', '지원', '포맷', '형식', '종류']
        if any(pattern in message_lower for pattern in file_patterns):
            return f"""지원하는 파일 형식:

📄 문서: PDF, Word (.docx), 텍스트 (.txt)
🖼️ 이미지: JPG, PNG (OCR로 텍스트 추출)
📊 기타: 마크다운, CSV 등

최대 50MB까지 업로드 가능합니다.{quota_msg}"""
        
        # 기본 응답
        if quota_exceeded:
            return "현재 Gemini API 할당량을 초과했습니다. 문서 기반 질문은 여전히 가능하니, 문서를 업로드하고 관련 질문을 해보시겠어요?"
        else:
            return "도움이 필요하시면 언제든지 말씀해주세요! 문서를 업로드하신 후 관련 질문을 해보시거나, 저의 기능에 대해 궁금한 점이 있으시면 물어보세요."
    
    def get_service_info(self) -> dict:
        """서비스 정보 반환"""
        return {
            "service": "Google Gemini",
            "model": settings.GEMINI_MODEL or "gemini-2.0-flash-exp (default)",
            "initialized": self._initialized,
            "cache_info": self._cached_generate.cache_info()._asdict() if hasattr(self._cached_generate, 'cache_info') else {}
        }
    
    async def cleanup(self):
        """리소스 정리"""
        if hasattr(self._cached_generate, 'cache_clear'):
            self._cached_generate.cache_clear()
        
        self._initialized = False
        self.model = None
        
        logger.info("Gemini 서비스 리소스 정리 완료")


# 전역 Gemini 서비스 인스턴스
gemini_service = GeminiService()