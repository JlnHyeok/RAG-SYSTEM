"""
답변 생성 모듈
다양한 전략을 사용하여 RAG 기반 답변을 생성합니다.
"""
import logging
import asyncio
from typing import List, Dict, Any, Optional

from app.core.processing.text_processor import text_processor

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
        history_formatter=None,
        question_type: str = "DOCUMENT_QUERY"
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
            
            # 추론 강화 프롬프트 생성 (질문 유형 반영)
            enhanced_prompt = self._create_reasoning_prompt(
                question_to_use, context, history_context + context_note, question_type
            )
            
            answer = await self.gemini_service.generate_answer(
                question=enhanced_prompt,
                context="",  # 프롬프트에 이미 포함됨
                max_tokens=4096,  # 한국어 답변을 위해 토큰 증가
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
            
            # [Fail-fast] Quota 초과 시 LLM 재시도 생략하고 즉시 Rule-based 요약 반환
            if "429" in str(e) or "Quota" in str(e):
                logger.warning("Quota 초과 감지 - LLM 재시도 생략")
                return text_processor.create_llm_free_summary(question_to_use, context)
            
            # 2단계: 단순 프롬프트로 재시도
            try:
                question_to_use = enhanced_question or original_question
                simple_answer = await self._generate_simple_answer(question_to_use, context)
                if len(simple_answer.strip()) > 50:
                    return text_processor.remove_duplicate_content(simple_answer)
            except Exception as e2:
                logger.warning(f"Gemini API 2차 실패: {e2}")
            
            # 3단계: LLM 기반 구조화된 fallback
            # (Quota 에러가 아니었을 때만 시도)
            try:
                fallback_answer = await self._create_llm_guided_fallback(question_to_use, context)
                return text_processor.remove_duplicate_content(fallback_answer)
            except Exception as e3:
                logger.warning(f"LLM 가이드 fallback 실패: {e3}")

            return text_processor.create_llm_free_summary(question_to_use, context)
    
    def _create_reasoning_prompt(
        self, 
        question: str, 
        context: str, 
        additional_context: str = "",
        question_type: str = "DOCUMENT_QUERY"
    ) -> str:
        """질문 유형에 따른 추론 강화 프롬프트 생성"""
        
        # 1. 기본 페르소나 및 지침 설정
        if question_type in ["ALARM_QUERY", "ABNORMAL_QUERY"]:
            persona = "당신은 제조 현장의 품질관리 전문가입니다. 이상감지 데이터를 분석하여 정확한 현황과 조치 방안을 제시하세요."
            instructions = """
            1. **데이터 소스별 구분**: MongoDB의 실시간 이상 발생 이력과 InfluxDB의 센서 데이터(부하 등)를 비교 분석하여 정리하세요.
            2. **유형 및 패턴 분석**: 이상 발생 빈도와 시계열 패턴을 통해 문제의 심각성을 진단하세요.
            3. **조치 방안**: 분석 결과에 따른 즉각적인 조치 및 점검 사항을 제안하세요.
            """
            tone = "✅ **전문적/객관적 톤**: 수치와 사실 위주로 명확하게 전달"
            data_source_label = "DB 조회 결과 (이상감지 이력)"
            
        elif question_type in ["SENSOR_QUERY", "RAW_SENSOR_QUERY"]:
            persona = "당신은 제조 현장의 설비 모니터링 전문가입니다. 실시간 센서 데이터를 분석하여 설비 상태를 정확히 보고하세요."
            instructions = """
            1. **데이터 소스별 구분**: InfluxDB의 시계열 통계/트렌드 데이터와 MongoDB의 마스터 설정을 구분하여 분석하세요.
            2. **현재 상태 및 트렌드**: 가동 여부, 현재 부하, 속도 등 핵심 수치와 더불어 최근 트렌드를 보고하세요.
            3. **정상 범위 판단**: 측정값이 설정된 임계치(MongoDB) 대비 정상 범위인지 명확히 판단하세요.
            """
            tone = "✅ **전문적/객관적 톤**: 수치와 사실 위주로 명확하게 전달"
            data_source_label = "DB 조회 결과 (실시간 센서 데이터)"
            
        elif question_type in ["DEVICE_QUERY", "MACHINE_QUERY"]:
            persona = "당신은 제조 현장의 설비관리 전문가입니다. 설비 정보를 명확하게 정리하여 안내하세요."
            instructions = """
            1. **데이터 소스별 구분**: MongoDB의 설비 마스터 정보와 InfluxDB의 최근 가동률(Operating Rate) 정보를 구분하여 보고하세요.
            2. **설비/공구 현황**: 설비별 등록 정보와 현재 장착된 공구의 마스터 정보를 일목요연하게 정리하세요.
            3. **상태 종합**: 가동 이력과 마스터 설정을 종합하여 설비의 전반적인 건전성을 평가하세요.
            """
            tone = "✅ **친절한/명확한 톤**: 정보를 정돈해서 알기 쉽게 전달"
            data_source_label = "DB 조회 결과 (설비 마스터 데이터)"
            
        elif question_type == "PRODUCTION_QUERY":
            persona = "당신은 제조 현장의 생산관리 전문가입니다. 생산 이력과 실적을 분석하여 명확하게 보고하세요."
            instructions = """
            1. **데이터 소스별 구분**: MongoDB의 생산 기록과 InfluxDB의 시계열 분석 데이터를 구분하여 제시하세요.
            2. **생산 현황 및 통계**: 생산량, 평균/최대/최소 CT(사이클타임) 등을 상세히 정리하세요.
            3. **인사이트 제안**: 데이터 기반의 생산성 향상 또는 병목 지점 개선 포인트를 제안하세요.
            """
            tone = "✅ **전문적/분석적 톤**: 데이터 기반으로 명확하게 전달"
            data_source_label = "DB 조회 결과 (생산 이력 데이터)"
            
        elif question_type == "TOOL_QUERY":
            persona = "당신은 제조 현장의 공구관리 전문가입니다. 공구 상태와 수명 정보를 정확하게 안내하세요."
            instructions = """
            1. **데이터 소스별 구분**: InfluxDB의 기간별 누적 사용 이력과 MongoDB의 실시간 수명/현황 데이터를 각각 명확히 구분하여 정리하세요.
            2. **공구 현황 분석**: 현재 장착된 공구의 코드, 명칭, 실시간 사용량 및 수명 한도를 상세히 보고하세요.
            3. **조업 패턴 및 교체 예측**: 이력 데이터를 통해 공구 소모 속도를 분석하고, 실시간 현황과 결합하여 예상 교체 시점을 도출하세요.
            4. **관리 제안**: 분석 결과를 바탕으로 교체 우선순위나 재고 관리 전략을 제안하세요.
            """
            tone = "✅ **전문적/실용적 톤**: 실무에 바로 적용 가능하게 전달"
            data_source_label = "DB 조회 결과 (공구 마스터 데이터)"
            
        elif question_type == "HYBRID_QUERY":
            persona = "당신은 제조 현장의 통합 분석 전문가입니다. 여러 데이터 소스(문서, DB)를 종합하여 분석하세요."
            instructions = """
            1. **데이터 종합**: 문서 내용과 DB 데이터를 함께 분석하세요.
            2. **원인 분석**: 이상 발생 시 문서의 가이드라인과 실제 데이터를 비교하세요.
            3. **결론 도출**: 종합적인 결론과 조치 방안을 제시하세요.
            """
            tone = "✅ **종합적/분석적 톤**: 여러 정보를 연결하여 인사이트 제공"
            data_source_label = "종합 데이터 (문서 + DB)"
            
        else:  # DOCUMENT_QUERY 및 기본값
            persona = "당신은 친절한 회사 규정 및 매뉴얼 안내 도우미입니다. 복잡한 내용을 일반 직원들이 쉽게 이해할 수 있도록 설명해주세요."
            instructions = """
            1. **정보 연결**: 문서 곳곳에 흩어진 정보를 연결하여 종합적인 답변을 만드세요.
            2. **쉬운 번역**: 전문 용어나 법률 용어를 일상 언어로 쉽게 풀어쓰세요.
            3. **예시 활용**: 이해를 돕기 위해 구체적인 상황 예시를 드세요.
            """
            tone = "✅ **친근한/설명적 톤**: '~해요', '~입니다' 등 부드러운 대화체 사용"
            data_source_label = "관련 문서 내용"

        # 2. 통합 프롬프트 구성
        return f"""{persona}
                    다음 단계를 따라 답변해주세요:
                    {instructions}
                    4. **명확한 결론**: 질문자가 원하는 답을 간단명료하게 제시해주세요.
                    
                    질문: {question}
                    {data_source_label}:
                    {context}
                    {additional_context}
                    
                    답변할 때 다음 사항을 지켜주세요:
                    {tone}
                    ✅ **데이터 출처 명시**: 답변의 각 섹션이나 수치 데이터가 어느 소스(MongoDB, InfluxDB, 관련 문서 등)에서 온 것인지 명확히 언급하십시오.
                    ✅ **핵심만 간단히**: 불필요한 서론/결론은 생략하고 핵심 정보를 전달하세요.
                    ✅ **근거 명시**: 답변의 근거가 되는 데이터나 문서를 구분하여 언급하세요. (예: "MongoDB에 등록된 마스터 정보에 따르면...", "InfluxDB의 이력 데이터를 분석한 결과...")
                    ✅ **중복 방지**: 같은 내용을 반복하지 말고, 한 번에 완전한 답변을 제공하세요.
                    ✅ **표 포맷팅**: 데이터 비교가 필요한 경우 Markdown 표를 활용하세요.

                    답변 형식:
                    **요약 답변:**
                    [질문에 대한 핵심 답변을 1-2문장으로]

                    **상세 분석/설명:**
                    [유형에 맞는 상세 내용]

                    **참고/조치사항:** (필요한 경우)
                    [추가 정보나 권장 조치]
                    
                    ⚠️ 중요: 답변을 절대 중간에 끊지 마세요. 모든 분석과 설명을 완전히 작성해야 합니다."""
    
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
                config = self.gemini_service._create_generation_config(4096, 0.3)
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
                                 중요: 이전 답변의 내용을 참고하되, 처음부터 끝까지 완전한 새로운 답변을 작성하세요.
                                 절대 답변을 중간에 끊지 마세요.""",
                user_message=enhancement_prompt,
                max_tokens=4096,  # 완전한 답변을 위해 충분히 확보
                temperature=0.1
            )
            
            return enhanced
            
        except Exception as e:
            logger.warning(f"답변 보완 실패: {e}")
            # 보완 실패 시 단순 재생성 시도
            try:
                return await self._generate_simple_answer(question, context)
            except Exception:
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
            max_tokens=4096,
            temperature=0.7
        )
        
        return answer


# 팩토리 함수
def create_answer_generator(gemini_service) -> AnswerGenerator:
    """AnswerGenerator 인스턴스 생성"""
    return AnswerGenerator(gemini_service)
