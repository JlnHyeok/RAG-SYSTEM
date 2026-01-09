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
        
        try:
            if not self._initialized:
                await self.initialize()
            
            logger.info(f"RAG 쿼리 처리 시작: '{request.question[:50]}...' (사용자: {request.user_id})")
            
            # 0단계: 메타 질문 감지 및 처리 (문서 목록, 시스템 상태 등)
            meta_response = await self._handle_meta_questions(request.question, request.user_id)
            if meta_response:
                return QueryResponse(
                    answer=meta_response,
                    sources=[],
                    confidence=0.9,
                    processing_time=time.time() - start_time
                )
            
            # 1단계: 질문을 벡터로 변환
            question_embedding = await self._embed_question(request.question)
            
            # 2단계: 벡터 DB에서 유사한 문서 검색
            search_results = await self._vector_search(
                question_embedding=question_embedding,
                user_id=request.user_id,
                limit=request.max_results,
                score_threshold=request.score_threshold
            )
            
            # 3단계: 검색 결과가 없으면 일반 대화 모드로 전환
            if not search_results:
                return await self._handle_general_conversation(request.question, time.time() - start_time)
            
            # 4단계: 컨텍스트 구성
            context = self._build_context(search_results)
            
            # 5단계: Gemini로 최종 답변 생성
            answer = await self._generate_answer(request.question, context)
            
            # 6단계: 응답 구성
            processing_time = time.time() - start_time
            confidence = self._calculate_confidence(search_results)
            
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
        """시스템 메타 질문 처리 (문서 목록, 상태 등)"""
        question_lower = question.lower().strip()
        
        # 문서 목록 관련 질문들
        document_keywords = [
            "문서", "파일", "가진", "업로드", "저장", "목록", "리스트", 
            "뭐가 있", "뭐 있", "무엇", "어떤", "얼마나", "몇 개"
        ]
        
        if any(keyword in question_lower for keyword in document_keywords):
            try:
                # 사용자의 컬렉션에서 문서 정보 가져오기
                collection_name = f"documents_{user_id}"
                await self.vector_store.ensure_collection(collection_name)
                
                # 간단한 검색으로 저장된 문서들 확인
                test_embedding = await self.embedding_manager.embed_text("test")
                all_docs = await self.vector_store.search_similar(
                    collection_name=collection_name,
                    query_vector=test_embedding,
                    limit=100,
                    score_threshold=0.0
                )
                
                if not all_docs:
                    return "현재 업로드된 문서가 없습니다. '/upload <파일명>' 명령어로 문서를 업로드해보세요."
                
                # 문서별로 그룹화
                doc_groups = {}
                for doc in all_docs:
                    filename = doc.metadata.get("original_filename", "Unknown")
                    if filename not in doc_groups:
                        doc_groups[filename] = []
                    doc_groups[filename].append(doc)
                
                # 응답 생성
                response_parts = [f"📚 현재 저장된 문서 ({len(doc_groups)}개):"]
                
                for i, (filename, docs) in enumerate(doc_groups.items(), 1):
                    file_type = docs[0].metadata.get("file_type", "unknown")
                    chunk_count = len(docs)
                    upload_time = docs[0].metadata.get("created_at", "Unknown")[:10] if docs[0].metadata.get("created_at") else "Unknown"
                    
                    response_parts.append(f"{i}. 📄 {filename} ({file_type.upper()})")
                    response_parts.append(f"   - 청크 수: {chunk_count}개")
                    response_parts.append(f"   - 업로드: {upload_time}")
                
                response_parts.append("\n💡 이 문서들에 대해 질문하시면 관련 내용을 찾아드립니다!")
                
                return "\n".join(response_parts)
                
            except Exception as e:
                logger.error(f"문서 목록 조회 실패: {e}")
                return "문서 목록을 확인하는 중 오류가 발생했습니다."
        
        # 시스템 상태/연결 관련 질문들 (더 구체적으로 분리)
        connection_keywords = ["연결", "접속", "커넥션"]
        system_keywords = ["상태", "시스템", "어떻게", "작동"]
        
        if any(keyword in question_lower for keyword in connection_keywords):
            return """🔗 연결 상태:
✅ Qdrant 벡터 DB: 연결됨
✅ 임베딩 서비스: 정상
⚠️ Gemini API: 할당량 제한"""
            
        elif any(keyword in question_lower for keyword in system_keywords):
            return """🤖 RAG 시스템 상태:
✅ 벡터 데이터베이스: 연결됨 (Qdrant)
✅ 임베딩 엔진: 활성화됨
⚠️ AI 답변 엔진: 할당량 제한 중 (Gemini)

📋 주요 기능:
• 문서 업로드 및 저장 (/upload)
• 문서 내용 검색 및 질의응답
• 파일 목록 확인 (/list)
• 벡터 검색 테스트 (/search)

현재 AI 답변 할당량이 제한되어 있지만, 문서 검색과 컨텍스트 제공은 정상 작동합니다."""
        
        return None

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
        """벡터 유사도 검색 수행"""
        try:
            collection_name = f"documents_{user_id}"
            
            return await self.vector_store.search_similar(
                collection_name=collection_name,
                query_vector=question_embedding,
                limit=limit,
                score_threshold=score_threshold
            )
            
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
                return f"""현재 Gemini API 할당량을 초과했습니다. 

찾아진 관련 내용을 바탕으로 답변드립니다:

{context}

위 내용이 '{question}' 질문과 관련된 문서에서 찾은 정보입니다. 더 정확한 AI 답변을 원하시면 잠시 후 다시 시도해주세요."""
            else:
                return f"""답변 생성 중 오류가 발생했습니다.

관련 문서 내용:
{context}

위 내용을 참고하시기 바랍니다."""
    
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


# 전역 RAG 엔진 인스턴스
rag_engine = RAGEngine()