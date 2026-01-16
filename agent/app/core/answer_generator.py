"""
답변 생성 모듈
다양한 전략을 사용하여 RAG 기반 답변을 생성합니다.
"""
import logging
import asyncio
from typing import List, Dict, Any, Optional

from app.core.text_processor import text_processor

logger = logging.getLogger(__name__)


class AnswerGenerator:
    """LLM을 활용한 답변 생성을 담당하는 클래스"""
    
    def __init__(self, gemini_service):
        """
        Args:
            gemini_service: Gemini LLM 서비스 인스턴스
        """
        self.gemini_service = gemini_service
    
    async def generate_intelligent_answer(
        self, 
        original_question: str, 
        context: str, 
        enhanced_question: str = None, 
        history: List[Dict[str, Any]] = None,
        history_formatter=None
    ) -> str:
        """LLM이 대화 맥락을 고려하여 추론 강화된 지능적 답변 생성"""
        
        # 1단계: 맥락이 있는 Gemini API 호출
        try:
            # 히스토리 컨텍스트 구성
            history_context = ""
            if history and history_formatter:
                history_context = f"\n\n이전 대화:\n{history_formatter(history)}"
            
            # 질문 컨텍스트 구성
            question_to_use = enhanced_question or original_question
            context_note = ""
            if enhanced_question and enhanced_question != original_question:
                context_note = f"\n\n원래 질문: '{original_question}'\n맥락 보완된 질문: '{enhanced_question}'"
            
            # 추론 강화 프롬프트 생성
            enhanced_prompt = self._create_reasoning_prompt(
                question_to_use, context, history_context + context_note
            )
            
            answer = await self.gemini_service.generate_answer(
                question=enhanced_prompt,
                context="",  # 프롬프트에 이미 포함됨
                max_tokens=3000,
                temperature=0.2
            )
            
            # 답변 품질 검증
            if self._is_answer_complete(answer, original_question):
                return text_processor.remove_duplicate_content(answer)
            else:
                logger.warning("Gemini 답변이 불완전함, 보완된 답변 생성")
                enhanced_answer = await self._create_enhanced_answer(
                    original_question, context, answer, history_context + context_note
                )
                return text_processor.remove_duplicate_content(enhanced_answer)
                
        except Exception as e:
            logger.warning(f"맥락 인식 Gemini API 실패: {e}")
            
            # 2단계: 단순 프롬프트로 재시도
            question_to_use = enhanced_question or original_question
            try:
                simple_answer = await self._generate_simple_answer(question_to_use, context)
                if len(simple_answer.strip()) > 50:
                    return text_processor.remove_duplicate_content(simple_answer)
            except Exception as e2:
                logger.warning(f"Gemini API 2차 실패: {e2}")
            
            # 3단계: LLM 기반 구조화된 fallback
            fallback_answer = await self._create_llm_guided_fallback(question_to_use, context)
            return text_processor.remove_duplicate_content(fallback_answer)
    
    def _create_reasoning_prompt(
        self, 
        question: str, 
        context: str, 
        additional_context: str = ""
    ) -> str:
        """추론 강화 프롬프트 생성 - 일반인 친화적 답변"""
        return f"""당신은 친절한 회사 규정 안내 도우미입니다. 복잡한 규정을 일반 직원들이 쉽게 이해할 수 있도록 설명해주세요.

다음 단계를 따라 답변해주세요:

1. **정보 수집**: 문서에서 질문과 관련된 모든 정보를 찾아보세요.
2. **연결 분석**: 서로 다른 부분에 있는 정보들을 연결해서 분석해보세요.
3. **쉬운 설명**: 복잡한 규정 용어를 일상 언어로 바꿔서 설명해주세요.
4. **명확한 결론**: 질문자가 원하는 답을 간단명료하게 제시해주세요.

질문: {question}

관련 문서 내용:
{context}

{additional_context}

답변할 때 다음 사항을 지켜주세요:
✅ **쉬운 언어 사용**: 법률 용어나 복잡한 표현 대신 일상 언어로 설명
✅ **구체적인 예시**: 가능하면 구체적인 상황 예시를 들어 설명
✅ **핵심만 간단히**: 불필요한 세부사항은 생략하고 핵심만 전달
✅ **확실한 정보만**: 추측이나 불확실한 내용은 명시
✅ **친근한 톤**: 딱딱한 공식 문서 톤이 아닌 친근한 설명 톤 사용
✅ **중복 방지**: 같은 내용을 반복하지 말고, 한 번에 완전한 답변을 제공하세요. 이전 답변과 중복되는 내용은 피하세요.
✅ **일관성 유지**: 하나의 답변으로 완성하세요. 여러 버전의 답변을 제공하지 마세요.
✅ **완전한 답변**: 질문의 유형에 따라 필요한 모든 세부사항(일수, 절차, 조건, 서류 등)을 포함해서 완전한 답변을 제공하세요.
✅ **표 포맷팅**: 표를 사용할 때는 Markdown 표 형식을 정확히 사용하고, 각 열의 너비를 데이터에 맞게 조정하여 가독성을 높이세요. 헤더와 데이터의 길이를 고려하여 공백을 추가하세요.

답변 형식:
**간단한 답변:**
[질문에 대한 핵심 답변을 1-2문장으로]

**자세한 설명:**
[쉽게 풀어서 설명한 내용]

**예시:** (해당되는 경우)
[구체적인 상황 예시]

**주의사항:** (필요한 경우)
[알아두면 좋을 추가 정보]"""
    
    def _is_answer_complete(self, answer: str, question: str) -> bool:
        """답변 완성도 검증"""
        if not answer or len(answer.strip()) < 100:
            return False
        
        # 중간에 끊어진 것 같은 패턴 체크
        if answer.strip().endswith(('*', ':', '(', '-', ',', '및')):
            return False
        
        # 질문 키워드와 답변 관련성 체크
        question_keywords = question.lower().split()
        answer_lower = answer.lower()
        
        # 주요 키워드 중 일부라도 포함되어 있는지 확인
        keyword_match = any(
            keyword in answer_lower 
            for keyword in question_keywords 
            if len(keyword) > 2
        )
        
        return keyword_match and len(answer.strip()) > 100
    
    async def _generate_simple_answer(self, question: str, context: str) -> str:
        """단순한 프롬프트로 Gemini 답변 생성"""
        simple_prompt = f"""질문: {question}

관련 문서 내용:
{context[:2000]}

위 문서 내용을 바탕으로 질문에 대해 상세하고 완전한 답변을 해주세요."""

        if hasattr(self.gemini_service, 'model') and self.gemini_service.model:
            def generate():
                config = self.gemini_service._create_generation_config(2000, 0.3)
                response = self.gemini_service.model.generate_content(
                    simple_prompt,
                    generation_config=config
                )
                return response.text
            
            result = await asyncio.to_thread(generate)
            return text_processor.remove_duplicate_content(result)
        
        raise Exception("Gemini 모델이 초기화되지 않음")
    
    async def _create_enhanced_answer(
        self, 
        question: str, 
        context: str, 
        partial_answer: str, 
        additional_context: str = ""
    ) -> str:
        """불완전한 Gemini 답변을 보완"""
        try:
            enhancement_prompt = f"""다음은 질문에 대한 부분적인 답변입니다. 이를 완성하고 보완해서 완전한 답변을 만들어주세요:

질문: {question}
부분 답변: {partial_answer}

추가 문서 내용:
{context}

{additional_context}

중요 지시사항:
- 부분 답변의 내용을 반복하지 말고, 부족한 부분만 채워서 완전한 답변을 작성해주세요
- 이미 포함된 정보는 생략하고 새로운 정보만 추가하세요
- 하나의 일관된 답변을 작성하세요. 여러 버전의 답변이나 중복된 소개를 피하세요
- 친절하고 일관된 톤을 유지하세요"""

            enhanced = await self.gemini_service.generate_with_system_prompt(
                system_prompt="""당신은 문서 분석 전문가입니다. 불완전한 답변을 완성하는 것이 임무입니다.
중요: 중복을 피하고, 하나의 완전한 답변을 작성하세요. 부분 답변의 내용을 반복하지 마세요.""",
                user_message=enhancement_prompt,
                max_tokens=2000,
                temperature=0.1
            )
            
            return enhanced
            
        except Exception as e:
            logger.warning(f"답변 보완 실패: {e}")
            return partial_answer
    
    async def _create_llm_guided_fallback(self, question: str, context: str) -> str:
        """LLM 가이드 기반 fallback 답변"""
        try:
            fallback_prompt = f"""문서에서 '{question}' 관련 정보를 찾아 답변하세요.

문서 내용:
{context[:1500]}

답변:"""
            
            if hasattr(self.gemini_service, 'model') and self.gemini_service.model:
                def generate():
                    response = self.gemini_service.model.generate_content(fallback_prompt)
                    return response.text
                
                result = await asyncio.to_thread(generate)
                if len(result.strip()) > 30:
                    return text_processor.remove_duplicate_content(result)
                    
        except Exception as e:
            logger.warning(f"LLM 가이드 fallback 실패: {e}")
        
        # 최종 fallback: LLM이 완전히 실패한 경우
        return text_processor.create_llm_free_summary(question, context)
    
    async def generate_general_conversation_response(
        self, 
        question: str
    ) -> str:
        """일반 대화 처리 - RAG가 아닌 일반적인 질문 응답"""
        system_prompt = """
당신은 RAG(Retrieval Augmented Generation) 기반 문서 검색 AI 어시스턴트입니다.
현재 질문과 관련된 문서가 없지만, 일반적인 대화는 가능합니다.

당신의 주요 기능:
1. 문서 업로드 및 분석 (PDF, Word, 텍스트 파일 등)
2. 업로드된 문서에서 정보 검색 및 질의응답
3. 다국어 문서 처리 (한국어, 영어 등)
4. OCR을 통한 이미지 내 텍스트 추출
5. 벡터 검색을 통한 유사도 기반 문서 매칭
6. 실시간 스트리밍 응답

응답 가이드라인:
1. 인사 질문 (안녕, 안녕하세요 등): 친근하게 인사하고 자신을 RAG 시스템으로 소개
2. 정체성 질문 (너는 누구야, 뭘 하는 AI야 등): RAG 기반 문서 검색 AI라고 구체적으로 소개
3. 기능 질문 (뭘 할 수 있어, 어떤 기능이 있어 등): 위의 주요 기능들을 자세히 설명
4. 사용법 질문 (어떻게 사용해, 문서는 어떻게 올려 등): 문서 업로드 방법과 질문 방법 안내
5. 지원 파일 질문: PDF, Word, 텍스트, 이미지 파일 등 지원 형식 설명
6. 기타 일반 질문: 도움이 되는 답변을 제공하되, 문서 업로드를 통한 더 정확한 답변 가능성 언급

한국어로 자연스럽고 친근하며 도움이 되는 방식으로 답변해주세요.
문서가 없는 상황에서도 최대한 유용한 정보를 제공하세요.
"""
        
        answer = await self.gemini_service.generate_with_system_prompt(
            system_prompt=system_prompt,
            user_message=question,
            max_tokens=500,
            temperature=0.7
        )
        
        return answer


# 팩토리 함수
def create_answer_generator(gemini_service) -> AnswerGenerator:
    """AnswerGenerator 인스턴스 생성"""
    return AnswerGenerator(gemini_service)
