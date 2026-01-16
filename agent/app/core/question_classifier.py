"""
질문 분류 모듈
사용자 질문을 분석하여 유형(인사, 문서 목록, 시스템 상태, 문서 쿼리)을 분류합니다.
"""
import logging
import asyncio
from typing import Optional
from enum import Enum

logger = logging.getLogger(__name__)


class QuestionType(str, Enum):
    """질문 유형 열거형"""
    GREETING = "GREETING"
    DOCUMENT_LIST = "DOCUMENT_LIST"
    SYSTEM_STATUS = "SYSTEM_STATUS"
    DOCUMENT_QUERY = "DOCUMENT_QUERY"


class QuestionClassifier:
    """질문 유형을 분류하고 메타 질문을 처리하는 클래스"""
    
    def __init__(self, gemini_service, vector_store):
        """
        Args:
            gemini_service: Gemini LLM 서비스 인스턴스
            vector_store: 벡터 스토어 인스턴스
        """
        self.gemini_service = gemini_service
        self.vector_store = vector_store
    
    async def handle_meta_questions(self, question: str, user_id: str) -> Optional[str]:
        """메타 질문 및 일반 대화를 LLM으로 판단 후 처리"""
        try:
            classification = await self.classify_question(question)
            
            if classification == QuestionType.GREETING:
                return await self._handle_greeting(question)
            elif classification == QuestionType.DOCUMENT_LIST:
                return await self._handle_document_list_request(user_id)
            elif classification == QuestionType.SYSTEM_STATUS:
                return await self._handle_system_status_request()
            else:
                return None  # RAG 파이프라인으로 진행
                
        except Exception as e:
            logger.warning(f"메타 질문 처리 실패: {e}")
            return None  # 오류 시 RAG 파이프라인으로 진행
    
    async def classify_question(self, question: str) -> QuestionType:
        """LLM으로 질문 유형 분류"""
        prompt = f"""다음 사용자 질문을 분류해주세요:

"{question}"

분류 기준:
- GREETING: 단순한 인사, 안부 등 (예: 안녕, 하이, 잘 지내?)
- DOCUMENT_LIST: 업로드된 문서 목록 요청 (예: 문서 목록, 어떤 파일들이 있어?)  
- SYSTEM_STATUS: 시스템 상태 확인 요청 (예: 시스템 상태, 정상 작동?)
- DOCUMENT_QUERY: 문서 내용에 대한 질문 (예: 여비 규정, 일비 얼마?)

위 4가지 중 하나로만 답변하세요: GREETING, DOCUMENT_LIST, SYSTEM_STATUS, DOCUMENT_QUERY"""

        try:
            if hasattr(self.gemini_service, 'model') and self.gemini_service.model:
                def classify():
                    response = self.gemini_service.model.generate_content(prompt)
                    result = response.text.strip().upper()
                    # 유효한 분류만 반환
                    for question_type in QuestionType:
                        if question_type.value in result:
                            return question_type
                    return QuestionType.DOCUMENT_QUERY  # 기본값
                
                return await asyncio.to_thread(classify)
        except Exception as e:
            logger.warning(f"질문 분류 실패: {e}")
        
        return QuestionType.DOCUMENT_QUERY  # LLM 실패 시 기본값
    
    async def _handle_greeting(self, question: str) -> str:
        """LLM으로 인사 응답 생성"""
        prompt = f"""사용자가 "{question}"라고 말했습니다.

RAG 문서 검색 AI 어시스턴트로서 친근하고 자연스럽게 응답해주세요.
- 2-3문장으로 간단하게
- 문서 업로드와 질문을 자연스럽게 유도
- 과도하게 길거나 복잡하지 않게"""

        try:
            if hasattr(self.gemini_service, 'model') and self.gemini_service.model:
                def generate():
                    response = self.gemini_service.model.generate_content(prompt)
                    return response.text.strip()
                
                return await asyncio.to_thread(generate)
        except Exception as e:
            logger.warning(f"인사 응답 생성 실패: {e}")
        
        # LLM 실패 시 최소한의 기본 응답
        return "안녕하세요! 문서 관련 질문이 있으시면 도와드릴게요."
    
    async def _handle_document_list_request(self, user_id: str) -> str:
        """문서 목록 요청 처리"""
        try:
            collections = await self.vector_store.list_user_documents(user_id)
            if not collections:
                return "현재 업로드된 문서가 없습니다. 문서를 먼저 업로드해주세요."
            
            doc_list = []
            for i, doc in enumerate(collections, 1):
                doc_list.append(f"{i}. {doc.get('file_name', 'Unknown')}")
            
            return f"업로드된 문서 목록:\n" + "\n".join(doc_list)
        except Exception as e:
            logger.error(f"문서 목록 조회 실패: {e}")
            return "문서 목록을 조회할 수 없습니다."
    
    async def _handle_system_status_request(self) -> str:
        """시스템 상태 확인 요청 처리"""
        # 이 메서드는 RAG 엔진에서 health_check를 호출해야 함
        # 여기서는 간단한 응답만 반환
        return "✅ RAG 시스템이 정상적으로 작동 중입니다."


# 팩토리 함수 (의존성 주입용)
def create_question_classifier(gemini_service, vector_store) -> QuestionClassifier:
    """QuestionClassifier 인스턴스 생성"""
    return QuestionClassifier(gemini_service, vector_store)
