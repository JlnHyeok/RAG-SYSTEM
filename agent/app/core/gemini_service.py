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
            model_name = settings.GEMINI_MODEL or "gemini-2.0-flash-lite"
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
                model_name = settings.GEMINI_MODEL or "gemini-2.0-flash-lite"
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
                return "현재 Gemini API 할당량을 초과했습니다. 문서 기반 질문은 여전히 가능하니, 문서를 업로드하고 관련 질문을 해보시겠어요?"
            
            # 기타 에러 시 기본 응답 반환 (재귀 호출 방지)
            return self._get_basic_fallback_response(user_message)
    
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
    
    async def _get_intelligent_fallback_response(self, user_message: str, quota_exceeded: bool = False) -> str:
        """지능적인 fallback 응답 생성 - LLM을 활용한 자연어 이해"""
        
        # API 할당량 초과 시에는 간단한 기본 응답
        if quota_exceeded:
            return "현재 Gemini API 할당량을 초과했습니다. 문서 기반 질문은 여전히 가능하니, 문서를 업로드하고 관련 질문을 해보시겠어요?"
        
        # LLM이 사용 가능한 경우, 자연어로 의도 파악 후 적절한 응답 생성
        try:
            if self._initialized:
                system_prompt = """당신은 RAG 기반 AI 어시스턴트입니다. 사용자의 메시지 의도를 파악하고 적절한 응답을 생성하세요.

당신의 정보:
- 이름: RAG 기반 AI 어시스턴트
- 주요 기능: 문서 업로드 및 분석, 질의응답, 다국어 지원, OCR, 벡터 검색
- 지원 파일: PDF, Word, 텍스트, 이미지 (최대 50MB)
- 특징: 실시간 스트리밍, 정확한 정보 검색

응답 가이드라인:
1. 인사/첫 만남: 간단한 소개와 문서 업로드 안내
2. 정체성/소개 질문: RAG 시스템과 주요 기능 설명
3. 기능/사용법 질문: 구체적인 사용 방법과 기능 목록 제공
4. 파일/문서 관련: 지원 형식과 업로드 방법 안내
5. 기타: 도움이 되는 일반적인 안내

한국어로 친근하고 도움이 되는 톤으로 응답하세요."""

                response = await self.generate_with_system_prompt(
                    system_prompt=system_prompt,
                    user_message=user_message,
                    max_tokens=300,
                    temperature=0.7
                )
                return response
                
        except Exception as e:
            logger.warning(f"LLM 기반 fallback 응답 생성 실패: {e}")
        
        # LLM 실패 시 기본 응답
        return self._get_basic_fallback_response(user_message)
    
    def _get_basic_fallback_response(self, user_message: str) -> str:
        """기본 fallback 응답 (LLM 실패 시)"""
        return """안녕하세요! 저는 RAG 기반 AI 어시스턴트입니다.

📄 **주요 기능:**
• 문서 업로드 및 분석 (PDF, Word, 텍스트 등)
• 업로드된 문서 기반 질의응답
• 다국어 문서 처리 지원
• OCR을 통한 이미지 텍스트 추출
• 실시간 스트리밍 응답

💡 **사용법:**
1. 문서를 업로드해주세요
2. 문서 내용에 대해 자유롭게 질문하세요
3. 정확한 답변을 받아보세요!

도움이 필요하시면 언제든지 말씀해주세요! 🚀"""
    
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