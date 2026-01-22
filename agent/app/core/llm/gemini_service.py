import logging
import hashlib
from typing import Optional, Dict, Any, List, Tuple
import google.generativeai as genai
import asyncio
from functools import lru_cache
from datetime import datetime, timedelta

from app.core.config import settings
from app.models.exceptions import (
    GeminiAPIError,
    QuotaExceededError,
    ModelNotInitializedError
)

logger = logging.getLogger(__name__)


class GeminiService:
    """
    Google Gemini LLM 서비스 클래스
    
    Gemini API와의 모든 상호작용을 처리하는 서비스 클래스.
    컨텍스트 기반 답변 생성, 프롬프트 캐싱, 요약 및 키워드 추출 기능 제공.
    
    Attributes:
        model: Gemini GenerativeModel 인스턴스
        model_name: 사용 중인 모델 이름
        _initialized: 초기화 완료 여부
        _prompt_cache: 프롬프트 결과 캐시
    """
    
    def __init__(self) -> None:
        self.model: Optional[genai.GenerativeModel] = None
        self.model_name: str = ""
        self._initialized: bool = False
        self._prompt_cache: Dict[str, Tuple[str, datetime]] = {}  # hash -> (result, timestamp)
        self._cache_ttl_minutes: int = 30
        
    async def initialize(self, test_connection: bool = False) -> None:
        """
        Gemini API 초기화
        
        Args:
            test_connection: 연결 테스트 수행 여부
            
        Raises:
            GeminiAPIError: API 초기화 실패 시
        """
        try:
            if not settings.GEMINI_API_KEY:
                logger.warning("GEMINI_API_KEY가 설정되지 않음. Gemini 서비스 비활성화")
                self._initialized = False
                return
            
            # Gemini API 설정
            genai.configure(api_key=settings.GEMINI_API_KEY)
            
            # 모델 초기화
            self.model_name = settings.GEMINI_MODEL or "gemini-2.0-flash-lite"
            self.model = genai.GenerativeModel(self.model_name)
            
            # 연결 테스트 (선택적)
            if test_connection:
                try:
                    await self._test_connection()
                except QuotaExceededError:
                    logger.warning("Gemini API 할당량 초과, 연결 테스트 건너뛰기")
                except GeminiAPIError as e:
                    logger.error(f"Gemini API 연결 테스트 실패: {e}")
                    raise
            
            self._initialized = True
            logger.info(f"Gemini API 초기화 완료 (Model: {self.model_name})")
            
        except Exception as e:
            if self._is_quota_exceeded(e):
                logger.warning(f"Gemini API 할당량 초과, 기본 설정으로 초기화: {e}")
                genai.configure(api_key=settings.GEMINI_API_KEY)
                self.model_name = settings.GEMINI_MODEL or "gemini-2.0-flash-lite"
                self.model = genai.GenerativeModel(self.model_name)
                self._initialized = True
                logger.info("Gemini API 기본 초기화 완료 (연결 테스트 미실행)")
            else:
                logger.error(f"Gemini API 초기화 실패: {e}")
                raise GeminiAPIError(f"Gemini API 초기화 실패: {str(e)}")
    
    async def _test_connection(self) -> None:
        """
        Gemini API 연결 테스트
        
        Raises:
            QuotaExceededError: API 할당량 초과 시
            GeminiAPIError: 연결 테스트 실패 시
        """
        try:
            test_prompt = "안녕하세요. 연결 테스트입니다."
            response = await asyncio.to_thread(
                self.model.generate_content,
                test_prompt
            )
            
            if not response.text:
                raise GeminiAPIError("Gemini API 응답이 비어있습니다")
                
            logger.info("Gemini API 연결 테스트 성공")
            
        except Exception as e:
            if self._is_quota_exceeded(e):
                raise QuotaExceededError()
            logger.error(f"Gemini API 연결 테스트 실패: {e}")
            raise GeminiAPIError(f"Gemini API 연결 테스트 실패: {str(e)}")
    
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
    
    @lru_cache(maxsize=200)
    def _cached_generate_lru(self, prompt_hash: str, prompt: str) -> str:
        """
        LRU 캐시 기반 프롬프트 생성 (짧은 프롬프트용)
        
        Args:
            prompt_hash: 프롬프트 해시값
            prompt: 실제 프롬프트 텍스트
            
        Returns:
            생성된 응답 텍스트
            
        Raises:
            ModelNotInitializedError: 모델 미초기화 시
        """
        if not self._initialized:
            raise ModelNotInitializedError()
            
        response = self.model.generate_content(prompt)
        
        # 안전한 텍스트 추출
        if not response.candidates:
            logger.warning("Gemini 응답에 candidates가 없습니다. (Safety Block 가능성 - LRU Cache)")
            return "죄송합니다. 답변을 생성할 수 없습니다."
            
        try:
            return response.text
        except Exception:
            if response.candidates and response.candidates[0].content.parts:
                return response.candidates[0].content.parts[0].text
            return ""
    
    def _get_cached_result(self, prompt: str) -> Optional[str]:
        """
        TTL 기반 캐시에서 결과 조회
        
        Args:
            prompt: 프롬프트 텍스트
            
        Returns:
            캐시된 결과 또는 None
        """
        prompt_hash = hashlib.md5(prompt.encode()).hexdigest()
        
        if prompt_hash in self._prompt_cache:
            result, timestamp = self._prompt_cache[prompt_hash]
            # TTL 확인
            if datetime.now() - timestamp < timedelta(minutes=self._cache_ttl_minutes):
                logger.debug(f"프롬프트 캐시 히트: {prompt_hash[:8]}...")
                return result
            else:
                # 만료된 캐시 삭제
                del self._prompt_cache[prompt_hash]
        
        return None
    
    def _set_cached_result(self, prompt: str, result: str) -> None:
        """
        TTL 기반 캐시에 결과 저장
        
        Args:
            prompt: 프롬프트 텍스트
            result: 생성된 결과
        """
        prompt_hash = hashlib.md5(prompt.encode()).hexdigest()
        self._prompt_cache[prompt_hash] = (result, datetime.now())
        
        # 캐시 크기 제한 (500개 초과 시 가장 오래된 항목 제거)
        if len(self._prompt_cache) > 500:
            oldest_key = min(self._prompt_cache.keys(), 
                           key=lambda k: self._prompt_cache[k][1])
            del self._prompt_cache[oldest_key]
    
    async def generate_answer(
        self,
        question: str,
        context: str,
        max_tokens: int = 2000,
        temperature: float = 0.1
    ) -> str:
        """
        컨텍스트 기반 답변 생성
        
        Args:
            question: 사용자 질문
            context: 검색된 문서 컨텍스트
            max_tokens: 최대 응답 토큰 수
            temperature: 생성 다양성 (0.0~1.0)
            
        Returns:
            생성된 답변 텍스트
            
        Raises:
            GeminiAPIError: API 호출 실패 시
        """
        try:
            if not self._initialized:
                await self.initialize()
            
            # 프롬프트 구성
            prompt = self._build_rag_prompt(question, context)
            
            # TTL 캐시 확인 (중간 길이 프롬프트)
            if len(prompt) < 2000:
                cached_result = self._get_cached_result(prompt)
                if cached_result:
                    return cached_result
            
            # 짧은 프롬프트는 LRU 캐시도 사용
            if len(prompt) < 500:
                prompt_hash = hashlib.md5(prompt.encode()).hexdigest()
                try:
                    result = self._cached_generate_lru(prompt_hash, prompt)
                    self._set_cached_result(prompt, result)  # TTL 캐시에도 저장
                    return result
                except QuotaExceededError:
                    return self._get_quota_exceeded_response(question, context)
                except Exception as e:
                    if self._is_quota_exceeded(e):
                        return self._get_quota_exceeded_response(question, context)
                    raise
            
            # 긴 프롬프트는 비동기 처리
            def generate() -> str:
                response = self.model.generate_content(
                    prompt,
                    generation_config=self._create_generation_config(max_tokens, temperature)
                )
                
                # 안전한 텍스트 추출
                if not response.candidates:
                    logger.warning("Gemini 응답에 candidates가 없습니다. (Safety Block 가능성)")
                    if response.prompt_feedback:
                        logger.warning(f"Prompt Feedback: {response.prompt_feedback}")
                    return "죄송합니다. 답변을 생성할 수 없습니다 (안전 정책 또는 오류)."
                
                try:
                    return response.text
                except Exception as e:
                    logger.error(f"response.text 추출 실패: {e}")
                    # 대체 접근 시도
                    if response.candidates and response.candidates[0].content.parts:
                        return response.candidates[0].content.parts[0].text
                    return ""
            
            result = await asyncio.to_thread(generate)
            
            # 결과를 TTL 캐시에 저장
            self._set_cached_result(prompt, result)
            return result
            
        except QuotaExceededError:
            return self._get_quota_exceeded_response(question, context)
        except Exception as e:
            logger.error(f"Gemini 답변 생성 실패: {e}")
            if self._is_quota_exceeded(e):
                return self._get_quota_exceeded_response(question, context)
            return self._get_fallback_response(e)
    
    async def generate_with_system_prompt(
        self,
        system_prompt: str,
        user_message: str,
        max_tokens: int = 4096,
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