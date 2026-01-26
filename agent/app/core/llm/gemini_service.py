import logging
import hashlib
import asyncio
from typing import Optional, Dict, Any, List, Tuple
from functools import lru_cache
from datetime import datetime, timedelta

from google import genai
from google.genai import types

from app.core.config import settings
from app.models.exceptions import (
    GeminiAPIError,
    QuotaExceededError,
    ModelNotInitializedError
)

logger = logging.getLogger(__name__)


class GeminiService:
    """
    Google Gemini LLM 서비스 클래스 (Updated for google-genai SDK 1.0+)
    
    Gemini API와의 모든 상호작용을 처리하는 서비스 클래스.
    컨텍스트 기반 답변 생성, 프롬프트 캐싱, 요약 및 키워드 추출 기능 제공.
    
    Attributes:
        client: google.genai.Client 인스턴스
        model_name: 사용 중인 모델 이름
        _initialized: 초기화 완료 여부
        _prompt_cache: 프롬프트 결과 캐시
    """
    
    def __init__(self) -> None:
        self.client: Optional[genai.Client] = None
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
            
            # Gemini Client 초기화
            self.client = genai.Client(api_key=settings.GEMINI_API_KEY)
            self.model_name = settings.GEMINI_MODEL or "gemini-2.0-flash-lite"
            
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
                self.client = genai.Client(api_key=settings.GEMINI_API_KEY)
                self.model_name = settings.GEMINI_MODEL or "gemini-2.0-flash-lite"
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
            # SDK 1.0+: client.models.generate_content
            response = await asyncio.to_thread(
                self.client.models.generate_content,
                model=self.model_name,
                contents=test_prompt
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
    
    def _create_generation_config(self, max_tokens: int, temperature: float) -> types.GenerateContentConfig:
        """
        일관된 GenerationConfig 생성
        SDK 1.0+에서는 types.GenerateContentConfig 사용
        """
        return types.GenerateContentConfig(
            max_output_tokens=max_tokens,
            temperature=temperature,
            top_p=0.95,
            top_k=40,
            candidate_count=1,
            stop_sequences=[]
        )
    
    def _extract_text_from_response(self, response) -> str:
        """
        Gemini 응답 객체에서 텍스트를 안전하게 추출
        SDK updated: response.text property is preferred
        """
        try:
            if not response:
                return ""
            
            # 1. response.text (Standard access)
            if hasattr(response, 'text') and response.text:
                return response.text
                
            # 텍스트가 없는 경우 상세 분석
            if hasattr(response, 'candidates') and response.candidates:
                candidate = response.candidates[0]
                if hasattr(candidate, 'content') and candidate.content:
                    if hasattr(candidate.content, 'parts') and candidate.content.parts:
                        for part in candidate.content.parts:
                            if hasattr(part, 'function_call') and part.function_call:
                                logger.warning(f"모델이 텍스트 대신 함수 호출을 반환했습니다: {part.function_call.name}")
                            if hasattr(part, 'executable_code') and part.executable_code:
                                logger.warning("모델이 실행 가능한 코드를 반환했습니다.")
                    else:
                        logger.warning(f"Candidate content parts가 비어있습니다. Content: {candidate.content}")
                else:
                    logger.warning("Candidate content가 없습니다.")
            
            logger.warning("Gemini 응답 텍스트 없음 (구조 확인 필요)")
            return ""
            
        except Exception as e:
            logger.error(f"Gemini 텍스트 추출 실패: {e}")
            return ""
    
    @lru_cache(maxsize=200)
    def _cached_generate_lru(self, prompt_hash: str, prompt: str) -> str:
        """LRU 캐시 기반 프롬프트 생성"""
        if not self._initialized:
            raise ModelNotInitializedError()
            
        response = self.client.models.generate_content(
            model=self.model_name,
            contents=prompt
        )
        
        return self._extract_text_from_response(response)
    
    def _get_cached_result(self, prompt: str) -> Optional[str]:
        """TTL 기반 캐시에서 결과 조회"""
        prompt_hash = hashlib.md5(prompt.encode()).hexdigest()
        
        if prompt_hash in self._prompt_cache:
            result, timestamp = self._prompt_cache[prompt_hash]
            if datetime.now() - timestamp < timedelta(minutes=self._cache_ttl_minutes):
                logger.debug(f"프롬프트 캐시 히트: {prompt_hash[:8]}...")
                return result
            else:
                del self._prompt_cache[prompt_hash]
        
        return None
    
    def _set_cached_result(self, prompt: str, result: str) -> None:
        """TTL 기반 캐시에 결과 저장"""
        prompt_hash = hashlib.md5(prompt.encode()).hexdigest()
        self._prompt_cache[prompt_hash] = (result, datetime.now())
        
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
        """컨텍스트 기반 답변 생성"""
        try:
            if not self._initialized:
                await self.initialize()
            
            prompt = self._build_rag_prompt(question, context)
            
            # 캐시 로직
            if len(prompt) < 2000:
                cached_result = self._get_cached_result(prompt)
                if cached_result:
                    return cached_result
            
            if len(prompt) < 500:
                prompt_hash = hashlib.md5(prompt.encode()).hexdigest()
                try:
                    result = self._cached_generate_lru(prompt_hash, prompt)
                    self._set_cached_result(prompt, result)
                    return result
                except Exception as e:
                    if not self._is_quota_exceeded(e):
                        logger.warning(f"LRU 생성 실패, 일반 생성 시도: {e}")
            
            # 비동기 생성 (Sync wrapper in Thread)
            def generate() -> str:
                response = self.client.models.generate_content(
                    model=self.model_name,
                    contents=prompt,
                    config=self._create_generation_config(max_tokens, temperature)
                )
                return self._extract_text_from_response(response)
            
            result = await asyncio.to_thread(generate)
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
        """시스템 프롬프트와 함께 답변 생성"""
        try:
            if not self._initialized:
                await self.initialize()
            
            # SDK 1.0 supports system instructions in config usually, but content concatenation is safer cross-version
            # Or use config(system_instruction=...) if supported. 
            # For robustness, we'll mimic the prompt structure unless we confirm system_instruction param
            
            # NOTE: google-genai supports `config=types.GenerateContentConfig(system_instruction=...)`
            # Let's use that for "proper" usage.
            
            def generate():
                config = self._create_generation_config(max_tokens, temperature)
                config.system_instruction = system_prompt
                
                response = self.client.models.generate_content(
                    model=self.model_name,
                    contents=user_message,
                    config=config
                )
                return self._extract_text_from_response(response)
            
            return await asyncio.to_thread(generate)
            
        except Exception as e:
            logger.error(f"Gemini 일반 대화 생성 실패: {e}")
            if self._is_quota_exceeded(e):
                return "현재 Gemini API 할당량을 초과했습니다. 잠시 후 다시 시도해주세요."
            return self._get_basic_fallback_response(user_message)
    
    def _build_rag_prompt(self, question: str, context: str) -> str:
        """RAG용 프롬프트 템플릿 구성 (기존 유지)"""
        return f"""당신은 도움이 되는 AI 어시스턴트입니다. 주어진 컨텍스트를 바탕으로 사용자의 질문에 정확하고 완전한 답변을 제공해주세요.
                중요한 규칙:
                1. 컨텍스트에 있는 정보만을 사용해서 답변하세요
                2. 컨텍스트에 없는 정보는 추측하지 말고, 모르겠다고 답변하세요
                3. 답변은 한국어로 작성하세요
                4. 가능하면 구체적인 근거를 제시하세요
                5. 출처 정보가 있다면 언급해주세요
                6. 답변을 완전히 끝까지 작성하세요 - 중간에 끊지 마세요
                
                7. [🚨 CRITICAL - 절대 규칙] 컨텍스트에 비슷한 형식의 데이터가 3개 이상 반복된다면:
                   ✅ 필수 준수: 반드시 Markdown 표(Table) 형식으로 작성하세요
                   
                   표 작성 필수 요구사항:
                   - 상태 컬럼에는 반드시 아이콘과 텍스트 사용: ✅ (정상), ⚠️ (경고/이상), ❌ (불량/에러)
                   - 컨텍스트에 이미 아이콘이 포함되어 있다면 그대로 사용하세요
                   
                   🔑 중요 정보 누락 금지:
                   - 컨텍스트에 있는 핵심 정보를 생략하지 마세요
                   - 컨텍스트에 여러 섹션의 데이터가 있다면 (예: "생산품별 이상 판정", "최근 이상감지 상세 이력" 등) 모든 섹션을 빠짐없이 표로 작성하세요
                   
                   📐 표 정렬 및 가독성:
                   - 너비가 불규칙한 경우, 표의 너비를 넓게 조정하여 가독성을 높이세요
                   - 너무 긴 값은 적절히 줄여서 표시하되 중요 정보는 유지하세요
                   
                8. 상세하고 완성된 답변을 제공하세요
                컨텍스트:
                {context}
                질문: {question}
                답변 (표 형식 필수, 완전하고 상세하게 작성):"""
    
    async def generate_summary(self, text: str, max_length: int = 200) -> str:
        try:
            if not self._initialized:
                await self.initialize()
            
            prompt = f"""다음 텍스트를 {max_length}자 이내로 요약해주세요. 핵심 내용을 놓치지 말고 간결하게 정리하세요.
                        텍스트:
                        {text}
                        요약:"""
            
            def generate():
                response = self.client.models.generate_content(
                    model=self.model_name,
                    contents=prompt,
                    config=self._create_generation_config(max_length * 2, 0.3)
                )
                return self._extract_text_from_response(response)
                
            return await asyncio.to_thread(generate)
            
        except Exception as e:
            logger.error(f"요약 생성 실패: {e}")
            return text[:max_length] + "..."
    
    async def generate_keywords(self, text: str, max_keywords: int = 5) -> list:
        try:
            if not self._initialized:
                await self.initialize()
            
            prompt = f"""다음 텍스트에서 가장 중요한 키워드 {max_keywords}개를 추출해주세요. 각 키워드는 쉼표로 구분하여 나열하세요.
                        텍스트:
                        {text}
                        키워드:"""
            
            def generate():
                response = self.client.models.generate_content(
                    model=self.model_name,
                    contents=prompt
                )
                text = self._extract_text_from_response(response)
                keywords_text = text.strip()
                return [kw.strip() for kw in keywords_text.split(',')]
                
            return await asyncio.to_thread(generate)
            
        except Exception as e:
            logger.error(f"키워드 추출 실패: {e}")
            return []
    
    def _get_fallback_response(self, error: Exception) -> str:
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
        """할당량 초과 시 컨텍스트 포함 답변 (기존 유지)"""
        # (생략: 기존 코드와 동일)
        # 로직이 길어서 여기서는 복원 필요. 기존 코드 그대로 사용.
        if not context or len(context.strip()) == 0:
            return "현재 Gemini API 할당량을 초과했습니다. 잠시 후 다시 시도해주세요."
        
        context_lines = context.strip().split('\n')
        formatted_parts = []
        current_doc = ""
        current_content = []
        
        for line in context_lines:
            if line.startswith('[문서'):
                if current_doc and current_content:
                    content_text = '\n'.join(current_content).strip()
                    if content_text:
                        formatted_parts.append(f"**{current_doc}**\n{content_text}")
                current_doc = line.strip('[]')
                current_content = []
            elif line.strip():
                current_content.append(line)
        
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
        """지능적인 fallback 응답 생성"""
        if quota_exceeded:
            return "현재 Gemini API 할당량을 초과했습니다."
        
        try:
            if self._initialized:
                system_prompt = """당신은 RAG 기반 AI 어시스턴트입니다. 사용자의 메시지 의도를 파악하고 적절한 응답을 생성하세요."""
                return await self.generate_with_system_prompt(
                    system_prompt=system_prompt,
                    user_message=user_message,
                    max_tokens=300,
                    temperature=0.7
                )
        except Exception as e:
            logger.warning(f"LLM 기반 fallback 응답 생성 실패: {e}")
        
        return self._get_basic_fallback_response(user_message)
    
    def _get_basic_fallback_response(self, user_message: str) -> str:
        return "안녕하세요! 저는 RAG 기반 AI 어시스턴트입니다. 도움이 필요하시면 질문해주세요!"

    def get_service_info(self) -> dict:
        """서비스 정보 반환"""
        # _cached_generate_lru는 lru_cache로 래핑되어 있음
        return {
            "service": "Google Gemini (Updated Client)",
            "model": settings.GEMINI_MODEL or "default",
            "initialized": self._initialized,
            "cache_info": self._cached_generate_lru.cache_info()._asdict() if hasattr(self._cached_generate_lru, 'cache_info') else {}
        }
    
    async def cleanup(self):
        """리소스 정리"""
        if hasattr(self._cached_generate_lru, 'cache_clear'):
            self._cached_generate_lru.cache_clear()
        
        self._initialized = False
        self.client = None
        logger.info("Gemini 서비스 리소스 정리 완료")

# 전역 Gemini 서비스 인스턴스
gemini_service = GeminiService()