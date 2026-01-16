"""
RAG 시스템의 핵심 엔진
검색과 생성을 통합 관리하며, 분리된 모듈들을 오케스트레이션합니다.
"""
import logging
from typing import List, Dict, Any, Optional
import asyncio
import time

from app.core.embedding_manager import embedding_manager
from app.core.vector_store import vector_store
from app.core.gemini_service import gemini_service
from app.core.text_processor import text_processor
from app.core.conversation_manager import conversation_manager
from app.core.question_classifier import QuestionClassifier
from app.core.answer_generator import AnswerGenerator
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
        self.text_processor = text_processor
        self.conversation_manager = conversation_manager
        self._initialized = False
        
        # 의존성 주입이 필요한 모듈들
        self.question_classifier: Optional[QuestionClassifier] = None
        self.answer_generator: Optional[AnswerGenerator] = None
    
    async def initialize(self):
        """RAG 엔진 초기화"""
        if self._initialized:
            return
        
        logger.info("RAG 엔진 초기화 시작...")
        
        try:
            # 모든 서비스 병렬 초기화
            await asyncio.gather(
                self.embedding_manager.initialize(),
                self.vector_store.initialize(),
                self.gemini_service.initialize(test_connection=False)
            )
            
            # 의존성 주입
            self.question_classifier = QuestionClassifier(
                self.gemini_service, 
                self.vector_store
            )
            self.answer_generator = AnswerGenerator(self.gemini_service)
            self.conversation_manager.set_gemini_service(self.gemini_service)
            
            self._initialized = True
            logger.info("RAG 엔진 초기화 완료!")
            
        except Exception as e:
            logger.error(f"RAG 엔진 초기화 실패: {e}")
            raise
    
    async def query(self, request: QueryRequest) -> QueryResponse:
        """사용자 질문에 대한 RAG 파이프라인 실행"""
        start_time = time.time()
        
        # 대화 키 생성 및 히스토리 초기화
        conversation_key = self.conversation_manager.get_conversation_key(
            request.user_id, 
            request.conversation_id
        )
        self.conversation_manager.ensure_history_exists(conversation_key)
        
        try:
            if not self._initialized:
                await self.initialize()
            
            logger.info(f"RAG 쿼리 처리 시작: '{request.question[:50]}...' (사용자: {request.user_id})")
            
            # 1. 대화 맥락 분석 및 질문 보완
            context_aware_question, is_contextual = await self.conversation_manager.analyze_question_context(
                request.question, conversation_key
            )
            
            # 2. 메타 질문 감지 및 처리
            meta_response = await self.question_classifier.handle_meta_questions(
                context_aware_question, 
                request.user_id
            )
            if meta_response:
                self.conversation_manager.add_to_history(
                    conversation_key, 
                    request.question, 
                    meta_response, 
                    [],
                    0.9
                )
                return QueryResponse(
                    answer=meta_response,
                    sources=[],
                    confidence=0.9,
                    processing_time=time.time() - start_time
                )
            
            # 3. 질문을 벡터로 변환
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
                return await self._handle_general_conversation(
                    request.question, 
                    time.time() - start_time
                )
            
            # 6. 컨텍스트 구성
            context = self.text_processor.build_context(search_results)
            
            # 7. 지능적 답변 생성
            answer = await self.answer_generator.generate_intelligent_answer(
                request.question, 
                context, 
                context_aware_question,
                self.conversation_manager.get_recent_history(conversation_key),
                self.conversation_manager.format_history_for_prompt
            )
            
            # 8. 응답 구성
            processing_time = time.time() - start_time
            confidence = self._calculate_confidence(search_results)
            
            # 대화 히스토리에 추가
            self.conversation_manager.add_to_history(
                conversation_key, 
                request.question, 
                answer, 
                search_results,
                confidence
            )
            
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
    
    def _calculate_confidence(self, search_results: List[SearchResult]) -> float:
        """검색 결과 기반 신뢰도 계산"""
        if not search_results:
            return 0.0
        
        scores = [result.score for result in search_results]
        max_score = max(scores)
        avg_score = sum(scores) / len(scores)
        result_count_factor = min(len(search_results) / 5.0, 1.0)
        
        confidence = (max_score * 0.6 + avg_score * 0.3 + result_count_factor * 0.1)
        return min(confidence, 1.0)
    
    async def _handle_general_conversation(
        self, 
        question: str, 
        processing_time: float
    ) -> QueryResponse:
        """일반 대화 처리"""
        try:
            answer = await self.answer_generator.generate_general_conversation_response(
                question
            )
            
            return QueryResponse(
                answer=answer,
                sources=[],
                confidence=0.8,
                processing_time=processing_time
            )
            
        except Exception as e:
            logger.error(f"일반 대화 처리 실패: {e}")
            return self._create_no_result_response(question, processing_time)
    
    def _create_no_result_response(
        self, 
        question: str, 
        processing_time: float
    ) -> QueryResponse:
        """검색 결과가 없을 때 기본 응답"""
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
            
            if self._initialized:
                status["components"]["embedding_manager"] = self.embedding_manager.get_model_info()
                status["components"]["gemini_service"] = self.gemini_service.get_service_info()
                status["components"]["conversation_manager"] = self.conversation_manager.get_history_stats()
                
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
    
    async def cleanup(self):
        """RAG 엔진 리소스 정리"""
        logger.info("RAG 엔진 리소스 정리 시작...")
        
        # 대화 히스토리 정리
        self.conversation_manager.clear_all_history()
        
        await asyncio.gather(
            self.embedding_manager.cleanup(),
            self.vector_store.cleanup(),
            self.gemini_service.cleanup(),
            return_exceptions=True
        )
        
        self._initialized = False
        logger.info("RAG 엔진 리소스 정리 완료")


# 전역 RAG 엔진 인스턴스
rag_engine = RAGEngine()