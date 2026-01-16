import logging
from typing import List, Dict, Any, Optional
import asyncio
import time

from app.core.embedding_manager import embedding_manager
from app.core.vector_store import vector_store
from app.core.gemini_service import gemini_service
from app.models.schemas import QueryRequest, QueryResponse, SearchResult
from app.models.enums import EmbeddingModelType
from app.core.config import settings

logger = logging.getLogger(__name__)


class RAGEngine:
    """RAG 시스템의 핵심 엔진 - 검색과 생성을 통합 관리"""
    
    def __init__(self):
        self.embedding_manager = embedding_manager
        self.vector_store = vector_store
        self.gemini_service = gemini_service
        self._initialized = False
        # 대화 히스토리 관리 (메모리 기반, 실제 운영시에는 Redis/DB 사용 권장)
        self.conversation_history: Dict[str, List[Dict[str, Any]]] = {}
    
    async def initialize(self):
        """RAG 엔진 초기화"""
        if self._initialized:
            return
        
        logger.info("RAG 엔진 초기화 시작...")
        
        try:
            # 모든 서비스 병렬 초기화 (Gemini는 연결 테스트 비활성화)
            await asyncio.gather(
                self.embedding_manager.initialize(),
                self.vector_store.initialize(),
                self.gemini_service.initialize(test_connection=False)
            )
            
            self._initialized = True
            logger.info("RAG 엔진 초기화 완료!")
            
        except Exception as e:
            logger.error(f"RAG 엔진 초기화 실패: {e}")
            raise
    
    async def query(self, request: QueryRequest) -> QueryResponse:
        """사용자 질문에 대한 RAG 파이프라인 실행"""
        start_time = time.time()
        
        # 대화 히스토리 초기화
        conversation_key = f"{request.user_id}_{request.conversation_id or 'default'}"
        if conversation_key not in self.conversation_history:
            self.conversation_history[conversation_key] = []
        
        try:
            if not self._initialized:
                await self.initialize()
            
            logger.info(f"RAG 쿼리 처리 시작: '{request.question[:50]}...' (사용자: {request.user_id})")
            
            # 1. 대화 맥락 분석 및 질문 보완
            context_aware_question, is_contextual = await self._analyze_question_context(
                request.question, conversation_key
            )
            
            # 2. 메타 질문 감지 및 처리 (맥락 인식된 질문으로)
            meta_response = await self._handle_meta_questions(context_aware_question, request.user_id)
            if meta_response:
                # 대화 히스토리에 추가
                self._add_to_history(conversation_key, request.question, meta_response, [])
                return QueryResponse(
                    answer=meta_response,
                    sources=[],
                    confidence=0.9,
                    processing_time=time.time() - start_time
                )
            
            # 3. 질문을 벡터로 변환 (맥락 인식된 질문 사용)
            question_embedding = await self._embed_question(context_aware_question)
            
            # 4. 벡터 DB에서 유사한 문서 검색
            search_results = await self._vector_search(
                question_embedding=question_embedding,
                user_id=request.user_id,
                limit=request.max_results,
                score_threshold=request.score_threshold
            )
            
            # 5. 검색 결과가 없으면 일반 대화 모드로 전환
            if not search_results:
                return await self._handle_general_conversation(request.question, time.time() - start_time)
            
            # 6. 컨텍스트 구성
            context = self._build_context(search_results)
            
            # 7. Gemini를 활용한 지능적 답변 생성 (원본 질문과 맥락, 히스토리 모두 전달)
            answer = await self._generate_intelligent_answer(
                request.question, 
                context, 
                context_aware_question,
                self._get_recent_history(conversation_key)
            )
            
            # 8. 응답 구성
            processing_time = time.time() - start_time
            confidence = self._calculate_confidence(search_results)
            
            # 대화 히스토리에 추가
            self._add_to_history(conversation_key, request.question, answer, search_results)
            
            response = QueryResponse(
                answer=answer,
                sources=search_results,
                confidence=confidence,
                processing_time=processing_time
            )
            
            logger.info(f"RAG 쿼리 완료: {processing_time:.2f}초, 신뢰도: {confidence:.2f}")
            return response
            
        except Exception as e:
            logger.error(f"RAG 쿼리 처리 실패: {e}")
            return QueryResponse(
                answer="처리 중 오류가 발생했습니다. 다시 시도해주세요.",
                sources=[],
                confidence=0.0,
                processing_time=time.time() - start_time
            )
    
    async def _handle_meta_questions(self, question: str, user_id: str) -> Optional[str]:
        """메타 질문 및 일반 대화를 LLM으로 판단 후 처리"""
        try:
            # LLM으로 질문 유형 분석
            classification = await self._classify_question_with_llm(question)
            
            if classification == "GREETING":
                return await self._handle_greeting_with_llm(question)
            elif classification == "DOCUMENT_LIST":
                return await self._handle_document_list_request(user_id)
            elif classification == "SYSTEM_STATUS":
                return await self._handle_system_status_request()
            else:
                return None  # RAG 파이프라인으로 진행
                
        except Exception as e:
            logger.warning(f"메타 질문 처리 실패: {e}")
            return None  # 오류 시 RAG 파이프라인으로 진행
    
    async def _classify_question_with_llm(self, question: str) -> str:
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
                    valid_types = ["GREETING", "DOCUMENT_LIST", "SYSTEM_STATUS", "DOCUMENT_QUERY"]
                    for valid_type in valid_types:
                        if valid_type in result:
                            return valid_type
                    return "DOCUMENT_QUERY"  # 기본값
                
                return await asyncio.to_thread(classify)
        except Exception as e:
            logger.warning(f"질문 분류 실패: {e}")
        
        return "DOCUMENT_QUERY"  # LLM 실패 시 기본값
    
    async def _handle_greeting_with_llm(self, question: str) -> str:
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
        try:
            status = await self.health_check()
            if status.get('rag_engine') == 'healthy':
                return "✅ RAG 시스템이 정상적으로 작동 중입니다."
            else:
                return f"⚠️ 시스템 상태: {status.get('error', '알 수 없는 오류')}"
        except Exception as e:
            return f"❌ 시스템 상태 확인 실패: {e}"

    async def _embed_question(self, question: str) -> List[float]:
        """질문을 임베딩으로 변환"""
        try:
            return await self.embedding_manager.embed_text(
                question, 
                EmbeddingModelType.KOREAN
            )
        except Exception as e:
            logger.error(f"질문 임베딩 실패: {e}")
            raise
    
    async def _vector_search(
        self,
        question_embedding: List[float],
        user_id: str,
        limit: int,
        score_threshold: float
    ) -> List[SearchResult]:
        """다중 전략 벡터 유사도 검색 수행"""
        try:
            collection_name = f"documents_{user_id}"
            
            # 기본 검색 결과
            primary_results = await self.vector_store.search_similar(
                collection_name=collection_name,
                query_vector=question_embedding,
                limit=limit,
                score_threshold=score_threshold
            )
            
            # 검색 결과가 충분하지 않으면 추가 검색 전략 적용
            if len(primary_results) < 3:
                # 더 낮은 임계값으로 추가 검색
                additional_results = await self.vector_store.search_similar(
                    collection_name=collection_name,
                    query_vector=question_embedding,
                    limit=limit * 2,
                    score_threshold=max(0.3, score_threshold - 0.2)
                )
                
                # 중복 제거하며 결과 합치기
                seen_ids = {r.chunk_id for r in primary_results}
                for result in additional_results:
                    if result.chunk_id not in seen_ids and len(primary_results) < limit:
                        primary_results.append(result)
                        seen_ids.add(result.chunk_id)
            
            return primary_results
            
        except Exception as e:
            logger.error(f"벡터 검색 실패: {e}")
            return []
    
    def _build_context(self, search_results: List[SearchResult]) -> str:
        """검색 결과를 바탕으로 컨텍스트 구성"""
        if not search_results:
            return ""
        
        context_parts = []
        for i, result in enumerate(search_results, 1):
            metadata = result.metadata
            
            # 컨텍스트 헤더 구성
            context_header = f"[문서 {i}"
            if metadata.get("page"):
                context_header += f", 페이지 {metadata['page']}"
            if metadata.get("type"):
                context_header += f", {metadata['type']}"
            if metadata.get("file_path"):
                file_name = metadata["file_path"].split("/")[-1]
                context_header += f", 출처: {file_name}"
            context_header += "]"
            
            # 내용 추가
            content = result.content.strip()
            if len(content) > 1000:  # 긴 내용은 요약
                content = content[:1000] + "..."
            
            context_part = f"{context_header}\n{content}\n"
            context_parts.append(context_part)
        
        return "\n".join(context_parts)
    
    async def _generate_intelligent_answer(
        self, 
        original_question: str, 
        context: str, 
        enhanced_question: str = None, 
        history: List[Dict[str, Any]] = None
    ) -> str:
        """LLM이 대화 맥락을 고려하여 추론 강화된 지능적 답변 생성"""
        
        # 1단계: 맥락이 있는 Gemini API 호출
        try:
            # 히스토리 컨텍스트 구성
            history_context = ""
            if history:
                history_context = f"\n\n이전 대화:\n{self._format_history_for_prompt(history)}"
            
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
                return self._remove_duplicate_content(answer)
            else:
                logger.warning("Gemini 답변이 불완전함, 보완된 답변 생성")
                enhanced_answer = await self._create_enhanced_gemini_answer(
                    original_question, context, answer, history_context + context_note
                )
                return self._remove_duplicate_content(enhanced_answer)
                
        except Exception as e:
            logger.warning(f"맥락 인식 Gemini API 실패: {e}")
            
            # 2단계: 단순 프롬프트로 재시도
            try:
                simple_answer = await self._generate_simple_gemini_answer(question_to_use, context)
                if len(simple_answer.strip()) > 50:
                    return self._remove_duplicate_content(simple_answer)
            except Exception as e2:
                logger.warning(f"Gemini API 2차 실패: {e2}")
            
            # 3단계: LLM 기반 구조화된 fallback
            fallback_answer = await self._create_llm_guided_fallback(question_to_use, context)
            return self._remove_duplicate_content(fallback_answer)
    
    def _create_reasoning_prompt(self, question: str, context: str, additional_context: str = "") -> str:
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
        keyword_match = any(keyword in answer_lower for keyword in question_keywords if len(keyword) > 2)
        
        return keyword_match and len(answer.strip()) > 100
    
    async def _generate_simple_gemini_answer(self, question: str, context: str) -> str:
        """단순한 프롬프트로 Gemini 답변 생성"""
        simple_prompt = f"""질문: {question}

관련 문서 내용:
{context[:2000]}

위 문서 내용을 바탕으로 질문에 대해 상세하고 완전한 답변을 해주세요."""

        def generate():
            response = self.model.generate_content(
                simple_prompt,
                generation_config=self.gemini_service._create_generation_config(2000, 0.3)
            )
            return response.text
        
        return self._remove_duplicate_content(await asyncio.to_thread(generate))
    
    async def _create_enhanced_gemini_answer(
        self, 
        question: str, 
        context: str, 
        partial_answer: str, 
        additional_context: str = ""
    ) -> str:
        """불완전한 Gemini 답변을 보완 (대화 맥락 포함)"""
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
            
            return enhanced  # 보완된 답변만 반환
            
        except Exception as e:
            logger.warning(f"답변 보완 실패: {e}")
            return partial_answer
    
    async def _create_llm_guided_fallback(self, question: str, context: str) -> str:
        """LLM 가이드 기반 fallback 답변"""
        try:
            # 마지막 시도: 매우 간단한 지시문으로
            fallback_prompt = f"""문서에서 '{question}' 관련 정보를 찾아 답변하세요.

문서 내용:
{context[:1500]}

답변:"""
            
            # 직접 모델 호출 (서비스 우회)
            if hasattr(self.gemini_service, 'model') and self.gemini_service.model:
                def generate():
                    response = self.gemini_service.model.generate_content(fallback_prompt)
                    return response.text
                
                result = await asyncio.to_thread(generate)
                if len(result.strip()) > 30:
                    return self._remove_duplicate_content(result)
                    
        except Exception as e:
            logger.warning(f"LLM 가이드 fallback 실패: {e}")
        
        # 최종 fallback: LLM이 완전히 실패한 경우에만 사용되는 단순한 문서 표시
        return await self._create_llm_free_summary(question, context)
    
    async def _create_llm_free_summary(self, question: str, context: str) -> str:
        """LLM 없이 질문에 맞는 문서 요약 (최종 fallback)"""
        if not context.strip():
            return "관련 문서를 찾을 수 없습니다."
        
        # 컨텍스트 파싱
        sections = self._parse_document_sections(context)
        
        answer_parts = [
            f"# 📄 '{question}' 관련 문서 내용\n",
            "⚠️ **AI 답변 생성에 실패하여 원본 문서 내용을 직접 제공합니다.**\n"
        ]
        
        # 모든 섹션을 순서대로 표시 (최대 5개)
        for i, section in enumerate(sections[:5], 1):
            answer_parts.append(f"## {i}. {section['header']}")
            
            # 내용이 너무 길면 적절히 자르기
            content = section['content']
            if len(content) > 1000:
                content = content[:1000] + "\n\n... (내용이 길어 일부만 표시됨)"
            
            answer_parts.append(content)
            answer_parts.append("")
        
        # 더 많은 섹션이 있으면 안내
        if len(sections) > 5:
            answer_parts.append(f"📋 **추가로 {len(sections) - 5}개의 문서 섹션이 더 있습니다.**")
        
        answer_parts.append("---")
        answer_parts.append("💡 **더 정확한 답변을 원하시면 구체적인 키워드로 다시 질문해주세요.**")
        
        return self._remove_duplicate_content("\n".join(answer_parts))
    
    def _parse_document_sections(self, context: str) -> list:
        """문서 섹션 파싱"""
        sections = []
        lines = context.strip().split('\n')
        
        current_header = ""
        current_content = []
        
        for line in lines:
            if line.startswith('[문서'):
                if current_header and current_content:
                    sections.append({
                        'header': current_header,
                        'content': '\n'.join(current_content).strip()
                    })
                current_header = line.strip('[]')
                current_content = []
            elif line.strip():
                current_content.append(line)
        
        if current_header and current_content:
            sections.append({
                'header': current_header,
                'content': '\n'.join(current_content).strip()
            })
        
        return sections

    def _create_enhanced_document_answer(self, question: str, context: str) -> str:
        """향상된 문서 기반 직접 답변 생성 (LLM 실패 시에만 사용)"""
        if not context.strip():
            return "관련 문서를 찾을 수 없습니다. 다른 키워드로 검색해보세요."
        
        # LLM이 완전히 실패한 경우에만 사용되는 간단한 fallback
        return self._create_smart_document_summary(question, context)

    async def _generate_answer_with_fallback(self, question: str, context: str) -> str:
        """Gemini 우선, 실패 시 즉시 문서 내용 기반 답변 제공"""
        try:
            # 먼저 Gemini API로 답변 생성 시도 (더 긴 답변을 위해 토큰 증가)
            gemini_answer = await self.gemini_service.generate_answer(
                question=question,
                context=context,
                max_tokens=2000,
                temperature=0.1
            )
            
            # 답변이 너무 짧거나 끊어진 것 같으면 fallback 사용
            if len(gemini_answer.strip()) < 100 or gemini_answer.strip().endswith(('*', ':', '(', '-', '.')):
                logger.warning(f"Gemini 답변이 불완전함 (길이: {len(gemini_answer)}), fallback 사용")
                return self._create_direct_document_answer(question, context)
            
            return gemini_answer
            
        except Exception as e:
            logger.warning(f"Gemini API 실패, 문서 기반 직접 답변 제공: {e}")
            
            # 즉시 구조화된 문서 내용 제공
            return self._create_direct_document_answer(question, context)
    
    def _create_direct_document_answer(self, question: str, context: str) -> str:
        """문서 내용을 직접 구조화하여 답변 생성"""
        
        # 컨텍스트 파싱
        context_lines = context.strip().split('\n')
        document_sections = []
        
        current_doc = ""
        current_content = []
        
        for line in context_lines:
            if line.startswith('[문서'):
                # 이전 문서 내용 저장
                if current_doc and current_content:
                    content_text = '\n'.join(current_content).strip()
                    if content_text:
                        document_sections.append({
                            'header': current_doc,
                            'content': content_text
                        })
                
                # 새 문서 시작
                current_doc = line.strip('[]')
                current_content = []
            elif line.strip():
                current_content.append(line)
        
        # 마지막 문서 저장
        if current_doc and current_content:
            content_text = '\n'.join(current_content).strip()
            if content_text:
                document_sections.append({
                    'header': current_doc,
                    'content': content_text
                })
        
        # 질문 키워드 기반 관련성 높은 내용 우선 배치
        question_keywords = self._extract_keywords(question)
        scored_sections = []
        
        for section in document_sections:
            relevance_score = self._calculate_text_relevance(
                section['content'], 
                question_keywords
            )
            scored_sections.append((relevance_score, section))
        
        # 관련성 순으로 정렬
        scored_sections.sort(key=lambda x: x[0], reverse=True)
        
        # 답변 구성
        answer_parts = [
            f"**'{question}'** 질문과 관련하여 다음과 같은 정보를 찾았습니다:\n"
        ]
        
        for i, (score, section) in enumerate(scored_sections[:5]):  # 상위 5개로 증가
            answer_parts.append(f"**{section['header']}**")
            # 내용 길이 제한을 늘리고 더 많은 정보 제공
            content = section['content']
            if len(content) > 1200:  # 800에서 1200으로 증가
                # 중요한 부분을 찾아서 더 지능적으로 자르기
                sentences = content.split('.')
                if len(sentences) > 3:
                    content = '. '.join(sentences[:int(len(sentences)*0.7)]) + "...\n\n(추가 내용 있음)"
                else:
                    content = content[:1200] + "..."
            answer_parts.append(content)
            answer_parts.append("")  # 구분선
        
        # 추가 문서가 있으면 표시
        if len(scored_sections) > 3:
            answer_parts.append(f"📋 추가로 {len(scored_sections) - 3}개의 관련 문서가 더 있습니다.")
        
        answer_parts.append("💡 **AI 분석이 일시적으로 제한되어 원본 문서 내용을 직접 제공했습니다.**")
        
        return "\n".join(answer_parts)
    
    def _extract_keywords(self, text: str) -> list:
        """간단한 키워드 추출 (한국어 지원)"""
        import re
        
        # 한국어, 영어, 숫자 조합 추출
        words = re.findall(r'[가-힣a-zA-Z0-9]+', text.lower())
        
        # 불용어 제거
        stopwords = {'은', '는', '이', '가', '을', '를', '에', '의', '로', '와', '과', 
                    'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with'}
        
        keywords = [word for word in words if len(word) > 1 and word not in stopwords]
        return list(set(keywords))  # 중복 제거
    
    def _calculate_text_relevance(self, text: str, keywords: list) -> float:
        """텍스트와 키워드 간 관련성 점수 계산"""
        if not keywords:
            return 0.0
        
        text_lower = text.lower()
        matches = 0
        
        for keyword in keywords:
            if keyword in text_lower:
                matches += text_lower.count(keyword)
        
        # 매치 수를 텍스트 길이로 정규화
        return matches / max(len(text.split()), 1)

    async def _generate_answer(self, question: str, context: str) -> str:
        """Gemini를 사용해 컨텍스트 기반 답변 생성"""
        try:
            return await self.gemini_service.generate_answer(
                question=question,
                context=context,
                max_tokens=1000,
                temperature=0.1  # 일관성 있는 답변을 위해 낮은 값
            )
            
        except Exception as e:
            logger.error(f"답변 생성 실패: {e}")
            
            # Gemini API 실패 시 컨텍스트 기반 직접 답변 생성
            if "할당량" in str(e) or "quota" in str(e).lower():
                return self._create_fallback_answer_with_context(question, context, "API 할당량 초과")
            else:
                return self._create_fallback_answer_with_context(question, context, "답변 생성 오류")
    
    def _create_fallback_answer_with_context(self, question: str, context: str, error_type: str) -> str:
        """Gemini 실패 시 컨텍스트를 포함한 대체 답변 생성"""
        
        # 컨텍스트에서 핵심 정보 추출
        context_lines = context.strip().split('\n')
        formatted_content = []
        
        current_doc = ""
        current_content = []
        
        for line in context_lines:
            if line.startswith('[문서'):
                # 이전 문서 내용 저장
                if current_doc and current_content:
                    clean_content = '\n'.join(current_content).strip()
                    if clean_content:
                        formatted_content.append(f"**{current_doc}**\n{clean_content}")
                
                # 새 문서 시작
                current_doc = line.strip('[]')
                current_content = []
            elif line.strip():
                current_content.append(line)
        
        # 마지막 문서 내용 저장
        if current_doc and current_content:
            clean_content = '\n'.join(current_content).strip()
            if clean_content:
                formatted_content.append(f"**{current_doc}**\n{clean_content}")
        
        # 답변 구성
        answer_parts = []
        
        if error_type == "API 할당량 초과":
            answer_parts.append("현재 Gemini API 할당량을 초과하여 AI 분석은 제한되지만, 관련 문서 내용을 찾아드렸습니다.")
        else:
            answer_parts.append("AI 답변 생성 중 오류가 발생했지만, 관련 문서 내용을 찾아드렸습니다.")
        
        answer_parts.append("")  # 빈 줄
        
        if formatted_content:
            for content in formatted_content:
                answer_parts.append(content)
                answer_parts.append("")  # 문서 간 구분
        else:
            answer_parts.append("관련 문서 내용:")
            answer_parts.append(context[:1500] + ("..." if len(context) > 1500 else ""))
        
        return "\n".join(answer_parts)
    
    def _calculate_confidence(self, search_results: List[SearchResult]) -> float:
        """검색 결과 기반 신뢰도 계산"""
        if not search_results:
            return 0.0
        
        # 최고 점수와 평균 점수를 조합하여 신뢰도 계산
        scores = [result.score for result in search_results]
        max_score = max(scores)
        avg_score = sum(scores) / len(scores)
        
        # 결과 개수도 고려 (더 많은 관련 문서가 있으면 신뢰도 상승)
        result_count_factor = min(len(search_results) / 5.0, 1.0)
        
        confidence = (max_score * 0.6 + avg_score * 0.3 + result_count_factor * 0.1)
        return min(confidence, 1.0)
    
    async def _handle_general_conversation(self, question: str, processing_time: float) -> QueryResponse:
        """일반 대화 처리 - RAG가 아닌 일반적인 질문 응답"""
        try:
            # 일반적인 인사, 소개 등의 질문을 Gemini로 처리
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
                temperature=0.7  # 자연스러운 대화를 위해 높은 값
            )
            
            return QueryResponse(
                answer=answer,
                sources=[],
                confidence=0.8,  # 일반 대화는 높은 신뢰도
                processing_time=processing_time
            )
            
        except Exception as e:
            logger.error(f"일반 대화 처리 실패: {e}")
            return self._create_no_result_response(question, processing_time)
    
    def _create_no_result_response(self, question: str, processing_time: float) -> QueryResponse:
        """검색 결과가 없을 때 기본 응답 (fallback)"""
        return QueryResponse(
            answer="죄송합니다. 질문과 관련된 문서를 찾을 수 없습니다. 다른 방식으로 질문해보시거나, 관련 문서를 먼저 업로드해주세요.",
            sources=[],
            confidence=0.0,
            processing_time=processing_time
        )
    
    async def health_check(self) -> Dict[str, Any]:
        """RAG 엔진 상태 확인"""
        try:
            status = {
                "rag_engine": "healthy",
                "initialized": self._initialized,
                "components": {}
            }
            
            # 각 컴포넌트 상태 확인
            if self._initialized:
                status["components"]["embedding_manager"] = self.embedding_manager.get_model_info()
                status["components"]["gemini_service"] = self.gemini_service.get_service_info()
                
                # 간단한 테스트 수행
                test_embedding = await self.embedding_manager.embed_text("테스트")
                status["components"]["embedding_test"] = {
                    "success": len(test_embedding) > 0,
                    "vector_size": len(test_embedding)
                }
            
            return status
            
        except Exception as e:
            logger.error(f"RAG 엔진 헬스 체크 실패: {e}")
            return {
                "rag_engine": "unhealthy",
                "error": str(e),
                "initialized": self._initialized
            }
    
    async def _analyze_question_context(self, question: str, conversation_key: str) -> tuple[str, bool]:
        """대화 맥락을 분석하여 질문을 보완"""
        try:
            history = self._get_recent_history(conversation_key, limit=3)
            if not history:
                return question, False
            
            # LLM으로 맥락 분석 및 질문 보완
            prompt = f"""이전 대화 히스토리와 현재 질문을 보고 질문을 보완해주세요.

이전 대화:
{self._format_history_for_prompt(history)}

현재 질문: "{question}"

만약 현재 질문이 이전 대화와 연관된 부가 질문이라면, 맥락을 포함하여 완전한 질문으로 변환해주세요.
예: "별표1이 뭔지 모르겠어" → "여비 규정에서 언급된 별표1이 무엇을 의미하는지 알려주세요"

만약 독립적인 질문이라면 원래 질문을 그대로 반환해주세요.

보완된 질문만 답변하세요:"""

            if hasattr(self.gemini_service, 'model') and self.gemini_service.model:
                def analyze():
                    response = self.gemini_service.model.generate_content(prompt)
                    enhanced_question = response.text.strip()
                    # 원본과 다르면 맥락적 질문으로 판단
                    is_contextual = enhanced_question != question and len(enhanced_question) > len(question)
                    return enhanced_question, is_contextual
                
                return await asyncio.to_thread(analyze)
        
        except Exception as e:
            logger.warning(f"질문 맥락 분석 실패: {e}")
        
        return question, False
    
    def _add_to_history(self, conversation_key: str, question: str, answer: str, sources: List[SearchResult]):
        """대화 히스토리에 Q&A 추가"""
        if conversation_key not in self.conversation_history:
            self.conversation_history[conversation_key] = []
        
        entry = {
            "timestamp": time.time(),
            "question": question,
            "answer": answer,
            "sources_count": len(sources),
            "confidence": self._calculate_confidence(sources) if sources else 0.0
        }
        
        self.conversation_history[conversation_key].append(entry)
        
        # 히스토리 크기 제한 (최근 10개만 유지)
        if len(self.conversation_history[conversation_key]) > 10:
            self.conversation_history[conversation_key] = self.conversation_history[conversation_key][-10:]
    
    def _get_recent_history(self, conversation_key: str, limit: int = 5) -> List[Dict[str, Any]]:
        """최근 대화 히스토리 조회"""
        if conversation_key not in self.conversation_history:
            return []
        
        history = self.conversation_history[conversation_key]
        return history[-limit:] if len(history) > limit else history
    
    def _format_history_for_prompt(self, history: List[Dict[str, Any]]) -> str:
        """프롬프트용 히스토리 포맷팅"""
        formatted = []
        for i, entry in enumerate(history, 1):
            formatted.append(f"Q{i}: {entry['question']}")
            formatted.append(f"A{i}: {entry['answer'][:200]}...")
        
        return "\n".join(formatted)
    
    async def cleanup(self):
        """RAG 엔진 리소스 정리"""
        logger.info("RAG 엔진 리소스 정리 시작...")
        
        await asyncio.gather(
            self.embedding_manager.cleanup(),
            self.vector_store.cleanup(),
            self.gemini_service.cleanup(),
            return_exceptions=True
        )
        
        self._initialized = False
        logger.info("RAG 엔진 리소스 정리 완료")

    def _remove_duplicate_content(self, answer: str) -> str:
        """답변에서 중복된 내용을 제거하고 표를 재포맷팅"""
        import re
        
        # 먼저 표 재포맷팅
        answer = self._reformat_markdown_table(answer)
        
        # 섹션별로 나누기 (예: **간단한 답변:**, **자세한 설명:** 등)
        sections = re.split(r'(\*\*.*?\*\*:)', answer)
        
        # 중복 제거를 위한 집합
        seen_lines = set()
        cleaned_sections = []
        
        for section in sections:
            if section.startswith('**') and section.endswith('**:'):
                # 섹션 헤더는 그대로 유지
                cleaned_sections.append(section)
                seen_lines.clear()  # 섹션 바뀔 때마다 초기화
            else:
                # 섹션 내용에서 중복 라인 제거
                lines = section.split('\n')
                unique_lines = []
                for line in lines:
                    line_stripped = line.strip()
                    if line_stripped and line_stripped not in seen_lines:
                        unique_lines.append(line)
                        seen_lines.add(line_stripped)
                    elif not line_stripped:
                        unique_lines.append(line)  # 빈 줄은 유지
                cleaned_sections.append('\n'.join(unique_lines))
        
        return ''.join(cleaned_sections).strip()

    def _reformat_markdown_table(self, text: str) -> str:
        """Markdown 표를 찾아서 정렬된 표로 재포맷팅"""
        import re
        
        # 표 패턴 찾기: |로 시작하는 라인들
        lines = text.split('\n')
        table_start = -1
        table_end = -1
        
        for i, line in enumerate(lines):
            if line.strip().startswith('|') and '|' in line:
                if table_start == -1:
                    table_start = i
                table_end = i
            elif table_start != -1 and not line.strip().startswith('|'):
                break
        
        if table_start == -1 or table_end - table_start < 2:
            return text  # 표가 없거나 너무 작으면 그대로 반환
        
        # 표 라인들 추출
        table_lines = lines[table_start:table_end + 1]
        
        # 각 열의 최대 너비 계산
        columns = []
        for line in table_lines:
            cells = [cell.strip() for cell in line.split('|')[1:-1]]  # | 제거하고 양쪽 공백 제거
            columns.append(cells)
        
        if not columns:
            return text
        
        # 각 열의 최대 너비 계산 (헤더 포함)
        max_widths = []
        for col_idx in range(len(columns[0])):
            max_width = 0
            for row in columns:
                if col_idx < len(row):
                    max_width = max(max_width, len(row[col_idx]))
            max_widths.append(max_width)
        
        # 표 재구성
        formatted_lines = []
        for row_idx, row in enumerate(columns):
            formatted_cells = []
            for col_idx, cell in enumerate(row):
                if col_idx < len(max_widths):
                    # 중앙 정렬로 포맷팅
                    formatted_cells.append(cell.center(max_widths[col_idx]))
                else:
                    formatted_cells.append(cell)
            
            # |로 묶어서 라인 생성
            formatted_line = '| ' + ' | '.join(formatted_cells) + ' |'
            formatted_lines.append(formatted_line)
            
            # 헤더 다음에 구분선 추가 (---|--- 형태)
            if row_idx == 0:
                separator_cells = ['-' * max_widths[col_idx] for col_idx in range(len(max_widths))]
                separator_line = '| ' + ' | '.join(separator_cells) + ' |'
                formatted_lines.append(separator_line)
        
        # 원본 텍스트에 재삽입
        new_lines = lines[:table_start] + formatted_lines + lines[table_end + 1:]
        return '\n'.join(new_lines)


# 전역 RAG 엔진 인스턴스
rag_engine = RAGEngine()