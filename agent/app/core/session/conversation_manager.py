"""
대화 히스토리 관리 모듈
대화 맥락을 분석하고 히스토리를 관리합니다.
"""
import logging
import asyncio
import time
from typing import List, Dict, Any, Tuple

from app.models.schemas import SearchResult

logger = logging.getLogger(__name__)


class ConversationManager:
    """대화 히스토리 관리 및 맥락 분석을 담당하는 클래스"""
    
    def __init__(self, gemini_service=None, max_history_size: int = 10):
        """
        Args:
            gemini_service: Gemini LLM 서비스 인스턴스 (맥락 분석용)
            max_history_size: 유지할 최대 히스토리 개수
        """
        self.gemini_service = gemini_service
        self.max_history_size = max_history_size
        # 대화 히스토리 관리 (메모리 기반)
        self.conversation_history: Dict[str, List[Dict[str, Any]]] = {}
    
    def set_gemini_service(self, gemini_service):
        """Gemini 서비스 설정 (초기화 후 주입용)"""
        self.gemini_service = gemini_service
    
    def get_conversation_key(self, user_id: str, conversation_id: str = None) -> str:
        """대화 키 생성"""
        return f"{user_id}_{conversation_id or 'default'}"
    
    def ensure_history_exists(self, conversation_key: str):
        """대화 히스토리 존재 확인 및 초기화"""
        if conversation_key not in self.conversation_history:
            self.conversation_history[conversation_key] = []
    
    async def analyze_question_context(
        self, 
        question: str, 
        conversation_key: str
    ) -> Tuple[str, bool]:
        """대화 맥락을 분석하여 질문을 보완"""
        try:
            history = self.get_recent_history(conversation_key, limit=3)
            if not history:
                return question, False
            
            if not self.gemini_service:
                return question, False
            
            # LLM으로 맥락 분석 및 질문 보완
            prompt = f"""이전 대화 히스토리와 현재 질문을 보고 질문을 보완해주세요.

                        이전 대화:
                        {self.format_history_for_prompt(history)}

                        현재 질문: "{question}"

                        만약 현재 질문이 이전 대화와 연관된 부가 질문이라면, 맥락을 포함하여 완전한 질문으로 변환해주세요.
                        예: "별표1이 뭔지 모르겠어" → "여비 규정에서 언급된 별표1이 무엇을 의미하는지 알려주세요"

                        만약 독립적인 질문이라면 원래 질문을 그대로 반환해주세요.

                        보완된 질문만 답변하세요:"""

            if hasattr(self.gemini_service, 'model') and self.gemini_service.model:
                def analyze():
                    response = self.gemini_service.model.generate_content(prompt)
                    enhanced_question = self.gemini_service._extract_text_from_response(response).strip()
                    
                    # 빈 응답이면 원본 질문 사용
                    if not enhanced_question:
                        return question, False
                    
                    # 원본과 다르면 맥락적 질문으로 판단
                    is_contextual = enhanced_question != question and len(enhanced_question) > len(question)
                    return enhanced_question, is_contextual
                
                return await asyncio.to_thread(analyze)
        
        except Exception as e:
            logger.warning(f"질문 맥락 분석 실패: {e}")
        
        return question, False
    
    def add_to_history(
        self, 
        conversation_key: str, 
        question: str, 
        answer: str, 
        sources: List[SearchResult],
        confidence: float = 0.0
    ):
        """대화 히스토리에 Q&A 추가"""
        self.ensure_history_exists(conversation_key)
        
        entry = {
            "timestamp": time.time(),
            "question": question,
            "answer": answer,
            "sources_count": len(sources),
            "confidence": confidence
        }
        
        self.conversation_history[conversation_key].append(entry)
        
        # 히스토리 크기 제한
        if len(self.conversation_history[conversation_key]) > self.max_history_size:
            self.conversation_history[conversation_key] = \
                self.conversation_history[conversation_key][-self.max_history_size:]
    
    def get_recent_history(
        self, 
        conversation_key: str, 
        limit: int = 5
    ) -> List[Dict[str, Any]]:
        """최근 대화 히스토리 조회"""
        if conversation_key not in self.conversation_history:
            return []
        
        history = self.conversation_history[conversation_key]
        return history[-limit:] if len(history) > limit else history
    
    def format_history_for_prompt(self, history: List[Dict[str, Any]]) -> str:
        """프롬프트용 히스토리 포맷팅"""
        formatted = []
        for i, entry in enumerate(history, 1):
            formatted.append(f"Q{i}: {entry['question']}")
            # 답변은 200자로 제한
            answer_preview = entry['answer'][:200] + "..." if len(entry['answer']) > 200 else entry['answer']
            formatted.append(f"A{i}: {answer_preview}")
        
        return "\n".join(formatted)
    
    def clear_history(self, conversation_key: str):
        """특정 대화 히스토리 삭제"""
        if conversation_key in self.conversation_history:
            del self.conversation_history[conversation_key]
    
    def clear_all_history(self):
        """모든 대화 히스토리 삭제"""
        self.conversation_history.clear()
    
    def get_history_stats(self) -> Dict[str, Any]:
        """히스토리 통계 반환"""
        total_conversations = len(self.conversation_history)
        total_messages = sum(
            len(history) for history in self.conversation_history.values()
        )
        return {
            "total_conversations": total_conversations,
            "total_messages": total_messages,
            "max_history_size": self.max_history_size
        }


# 전역 인스턴스
conversation_manager = ConversationManager()
