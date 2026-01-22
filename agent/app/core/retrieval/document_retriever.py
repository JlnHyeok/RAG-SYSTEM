"""
문서 검색 및 순위 지정 모듈
임베딩 엔진과 벡터 스토어를 사용하여 관련 문서 조각을 검색합니다.
"""
import logging
from typing import List, Optional
from app.core.retrieval.embedding_manager import embedding_manager
from app.core.retrieval.vector_store import vector_store
from app.models.schemas import SearchResult
from app.models.enums import EmbeddingModelType

logger = logging.getLogger(__name__)

class DocumentRetriever:
    """문서 검색 엔진 - 임베딩 및 벡터 유사도 검색 수행"""
    
    def __init__(self):
        self.embedding_manager = embedding_manager
        self.vector_store = vector_store
    
    async def initialize(self):
        """검색 엔진 초기화"""
        # 상위 엔진에서 이미 초기화되었을 수도 있지만 안전을 위해 보장
        await self.embedding_manager.initialize()
        await self.vector_store.initialize()

    async def search(
        self,
        question: str,
        user_id: str,
        limit: int = 5,
        score_threshold: float = 0.5
    ) -> List[SearchResult]:
        """질문에 기반한 관련 문서 검색"""
        try:
            # 1. 질문 임베딩
            embedding = await self.embedding_manager.embed_text(
                question, 
                EmbeddingModelType.KOREAN
            )
            
            # 2. 벡터 검색
            collection_name = f"documents_{user_id}"
            results = await self.vector_store.search_similar(
                collection_name=collection_name,
                query_vector=embedding,
                limit=limit,
                score_threshold=score_threshold
            )
            
            # 3. 검색 결과 부족 시 확장 검색
            if len(results) < 3:
                additional_results = await self.vector_store.search_similar(
                    collection_name=collection_name,
                    query_vector=embedding,
                    limit=limit * 2,
                    score_threshold=max(0.3, score_threshold - 0.2)
                )
                
                seen_ids = {r.chunk_id for r in results}
                for result in additional_results:
                    if result.chunk_id not in seen_ids and len(results) < limit:
                        results.append(result)
                        seen_ids.add(result.chunk_id)
            
            return results
            
        except Exception as e:
            logger.error(f"문서 검색 중 오류 발생: {e}")
            return []

    def calculate_confidence(self, search_results: List[SearchResult]) -> float:
        """검색 결과 기반 답변 신뢰도 계산"""
        if not search_results:
            return 0.0
        
        scores = [result.score for result in search_results]
        max_score = max(scores)
        avg_score = sum(scores) / len(scores)
        result_count_factor = min(len(search_results) / 5.0, 1.0)
        
        confidence = (max_score * 0.6 + avg_score * 0.3 + result_count_factor * 0.1)
        return min(confidence, 1.0)

document_retriever = DocumentRetriever()
