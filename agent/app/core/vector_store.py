import logging
from typing import List, Dict, Any, Optional
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct, Filter, FieldCondition, MatchValue
from qdrant_client.http import models
import uuid
import asyncio

from app.core.config import settings
from app.models.schemas import DocumentChunk, SearchResult

logger = logging.getLogger(__name__)


class VectorStore:
    """Qdrant 벡터 데이터베이스 관리 클래스"""
    
    def __init__(self):
        self.client: Optional[QdrantClient] = None
        self._connected = False
        
    async def initialize(self):
        """Qdrant 클라이언트 초기화 및 연결"""
        try:
            if settings.QDRANT_URL:
                self.client = QdrantClient(url=settings.QDRANT_URL)
            else:
                self.client = QdrantClient(host=settings.QDRANT_HOST, port=settings.QDRANT_PORT)
            
            # 연결 테스트
            await asyncio.to_thread(self.client.get_collections)
            self._connected = True
            
            logger.info(f"Qdrant 연결 성공: {settings.QDRANT_HOST}:{settings.QDRANT_PORT}")
            
        except Exception as e:
            logger.error(f"Qdrant 연결 실패: {e}")
            raise
    
    async def ensure_collection(self, collection_name: str, vector_size: int = 768):
        """컬렉션 존재 확인 및 생성"""
        if not self._connected:
            await self.initialize()
        
        try:
            collections = await asyncio.to_thread(self.client.get_collections)
            collection_names = [col.name for col in collections.collections]
            
            if collection_name not in collection_names:
                logger.info(f"컬렉션 생성: {collection_name}")
                
                await asyncio.to_thread(
                    self.client.create_collection,
                    collection_name=collection_name,
                    vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE)
                )
                
                logger.info(f"컬렉션 생성 완료: {collection_name}")
            
        except Exception as e:
            logger.error(f"컬렉션 확인/생성 실패: {e}")
            raise
    
    async def store_embeddings(
        self, 
        collection_name: str, 
        chunks: List[DocumentChunk],
        batch_size: int = 100
    ) -> int:
        """문서 청크들을 벡터 DB에 저장"""
        try:
            if not chunks:
                return 0
                
            # 컬렉션 존재 확인
            vector_size = len(chunks[0].embedding) if chunks else 768
            await self.ensure_collection(collection_name, vector_size)
            
            # 배치별로 처리
            total_stored = 0
            
            for i in range(0, len(chunks), batch_size):
                batch = chunks[i:i + batch_size]
                points = []
                
                for chunk in batch:
                    point = PointStruct(
                        id=str(uuid.uuid4()),
                        vector=chunk.embedding,
                        payload={
                            "content": chunk.content,
                            "metadata": chunk.metadata,
                            "created_at": chunk.created_at.isoformat()
                        }
                    )
                    points.append(point)
                
                # 벡터 DB에 저장
                await asyncio.to_thread(
                    self.client.upsert,
                    collection_name=collection_name,
                    points=points
                )
                
                total_stored += len(batch)
                logger.info(f"배치 저장 완료: {len(batch)}개 청크 ({total_stored}/{len(chunks)})")
            
            logger.info(f"전체 저장 완료: {total_stored}개 청크")
            return total_stored
            
        except Exception as e:
            logger.error(f"임베딩 저장 실패: {e}")
            raise
    
    async def search_similar(
        self,
        collection_name: str,
        query_vector: List[float],
        limit: int = 5,
        score_threshold: float = 0.7,
        filter_conditions: Optional[Dict[str, Any]] = None
    ) -> List[SearchResult]:
        """유사도 검색 수행"""
        try:
            if not self._connected:
                await self.initialize()
            
            # 필터 조건 설정
            query_filter = None
            if filter_conditions:
                conditions = []
                for key, value in filter_conditions.items():
                    conditions.append(
                        FieldCondition(key=f"metadata.{key}", match=MatchValue(value=value))
                    )
                if conditions:
                    query_filter = Filter(must=conditions)
            
            # 검색 수행
            search_result = await asyncio.to_thread(
                self.client.search,
                collection_name=collection_name,
                query_vector=query_vector,
                limit=limit,
                score_threshold=score_threshold,
                query_filter=query_filter
            )
            
            # 결과 변환
            results = []
            for hit in search_result:
                result = SearchResult(
                    document_id=str(hit.id),
                    content=hit.payload.get("content", ""),
                    score=hit.score,
                    metadata=hit.payload.get("metadata", {})
                )
                results.append(result)
            
            logger.info(f"검색 완료: {len(results)}개 결과 (임계값: {score_threshold})")
            return results
            
        except Exception as e:
            logger.error(f"유사도 검색 실패: {e}")
            return []
    
    async def add_documents(
        self, 
        documents: List[DocumentChunk], 
        user_id: str = "default"
    ) -> int:
        """문서 청크들을 벡터 DB에 추가"""
        try:
            if not documents:
                return 0
                
            collection_name = f"documents_{user_id}"
            return await self.store_embeddings(collection_name, documents)
            
        except Exception as e:
            logger.error(f"문서 추가 실패: {e}")
            return 0
    
    async def get_collection_info(self, collection_name: str) -> Dict[str, Any]:
        """컬렉션 정보 조회"""
        try:
            if not self._connected:
                await self.initialize()
            
            collection_info = await asyncio.to_thread(
                self.client.get_collection,
                collection_name
            )
            
            return {
                "name": collection_name,
                "vectors_count": collection_info.vectors_count,
                "indexed_vectors_count": collection_info.indexed_vectors_count,
                "points_count": collection_info.points_count,
                "status": collection_info.status.name,
                "optimizer_status": collection_info.optimizer_status.ok,
                "disk_data_size": collection_info.disk_data_size
            }
            
        except Exception as e:
            logger.error(f"컬렉션 정보 조회 실패: {e}")
            return {}
    
    async def delete_collection(self, collection_name: str) -> bool:
        """컬렉션 삭제"""
        try:
            if not self._connected:
                await self.initialize()
            
            await asyncio.to_thread(
                self.client.delete_collection,
                collection_name
            )
            
            logger.info(f"컬렉션 삭제 완료: {collection_name}")
            return True
            
        except Exception as e:
            logger.error(f"컬렉션 삭제 실패: {e}")
            return False
    
    async def cleanup(self):
        """리소스 정리"""
        if self.client:
            # Qdrant 클라이언트는 별도 정리가 필요하지 않음
            logger.info("Qdrant 클라이언트 연결 정리")
            self._connected = False
            self.client = None


# 전역 벡터 스토어 인스턴스
vector_store = VectorStore()