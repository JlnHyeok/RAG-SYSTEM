"""
문서 API 엔드포인트
파일 업로드, 처리 상태 조회, 삭제 등의 API를 제공합니다.
"""
from fastapi import APIRouter, HTTPException, UploadFile, File, Form, WebSocket, WebSocketDisconnect
from typing import Dict, Any, Optional
import logging
import time
import hashlib
import os
from pathlib import Path

# TOKENIZERS_PARALLELISM 경고 해결
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from app.services.processing_task import ProcessingTask, ProcessingProgress
from app.services.document_worker import (
    processing_queue,
    ensure_processing_worker_running
)
from app.core import vector_store, settings
from app.models.schemas import DocumentUploadResponse, DocumentDeleteResponse

logger = logging.getLogger(__name__)
router = APIRouter()


# ============================================================================
# 문서 업로드 및 상태 조회
# ============================================================================

@router.get("/upload/{document_id}/status")
async def get_upload_status(document_id: str):
    """문서 처리 상태 확인"""
    try:
        print(f"📊 상태 조회 요청: {document_id}", flush=True)
        
        progress_data = ProcessingProgress.get_progress(document_id)
        
        if not progress_data:
            logger.warning(f"진행률 정보를 찾을 수 없음: {document_id}")
            raise HTTPException(
                status_code=404,
                detail="문서를 찾을 수 없습니다"
            )
        
        return progress_data
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"상태 조회 실패: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"상태 조회 중 오류가 발생했습니다: {str(e)}"
        )


@router.post("/upload", response_model=DocumentUploadResponse)
async def upload_document(
    file: UploadFile = File(...),
    user_id: str = Form("anonymous"),
    document_type: Optional[str] = Form(None)
) -> DocumentUploadResponse:
    """PDF, Word, 텍스트 파일 업로드 - 업로드 완료 후 즉시 응답, 처리는 백그라운드에서"""
    start_time = time.time()
    
    try:
        # 파일 확장자 검증
        allowed_extensions = {'.pdf', '.docx', '.doc', '.txt', '.md'}
        file_extension = Path(file.filename).suffix.lower()
        
        if file_extension not in allowed_extensions:
            raise HTTPException(
                status_code=400, 
                detail=f"지원하지 않는 파일 형식입니다. 지원 형식: {', '.join(allowed_extensions)}"
            )
        
        # 파일 크기 검증 (50MB 제한)
        max_size = getattr(settings, 'MAX_FILE_SIZE', 52428800)
        if hasattr(file, 'size') and file.size > max_size:
            raise HTTPException(
                status_code=413, 
                detail=f"파일이 너무 큽니다. 최대 {max_size // 1024 // 1024}MB까지 지원합니다."
            )
        
        # 고유 문서 ID 생성
        file_hash = hashlib.md5(f"{user_id}_{file.filename}_{time.time()}".encode()).hexdigest()
        
        # 진행 상황 추적 시작
        progress = ProcessingProgress(file_hash, file.filename)
        await progress.start_step_async(0)
        
        # 파일 내용 읽기
        file_content = await file.read()
        await progress.complete_step_async()
        
        print(f"\n✅ 파일 업로드 완료: {file.filename} (크기: {len(file_content):,} bytes)", flush=True)
        logger.info(f"파일 업로드 완료: {file.filename} (크기: {len(file_content)} bytes)")
        
        # 처리 작업을 큐에 추가
        task = ProcessingTask(
            document_id=file_hash,
            file_content=file_content,
            file_extension=file_extension,
            user_id=user_id,
            original_filename=file.filename
        )
        
        await processing_queue.put(task)
        await ensure_processing_worker_running()
        
        print(f"📋 처리 작업 큐에 추가됨: {file.filename}", flush=True)
        
        return DocumentUploadResponse(
            document_id=file_hash,
            filename=file.filename,
            status="processing",
            message=f"파일 '{file.filename}' 업로드 완료. 문서 처리가 시작되었습니다.",
            processing_time=time.time() - start_time
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"파일 업로드 실패: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"파일 업로드 처리 중 오류가 발생했습니다: {str(e)}"
        )


@router.get("/status/{document_id}")
async def get_document_status(document_id: str):
    """문서 처리 상태 조회"""
    progress_data = ProcessingProgress.get_progress(document_id)
    if progress_data:
        return progress_data
    
    return {
        "document_id": document_id,
        "status": "completed",
        "message": "문서 처리가 완료되었습니다"
    }


# ============================================================================
# 문서 목록 조회
# ============================================================================

@router.get("/list")
async def list_documents(user_id: str = "anonymous") -> Dict[str, Any]:
    """업로드된 문서 목록 조회"""
    try:
        await vector_store.initialize()
        collection_name = f"documents_{user_id}"
        
        try:
            await vector_store.ensure_collection(collection_name)
            
            from app.core import embedding_manager
            await embedding_manager.initialize()
            
            test_embedding = await embedding_manager.embed_text("test")
            all_docs = await vector_store.search_similar(
                collection_name=collection_name,
                query_vector=test_embedding,
                limit=1000,
                score_threshold=0.0
            )
            
            # 문서별 그룹화
            doc_groups = {}
            for doc in all_docs:
                doc_id = doc.metadata.get("document_id", "unknown")
                filename = doc.metadata.get("original_filename", "Unknown")
                
                if doc_id not in doc_groups:
                    doc_groups[doc_id] = {
                        "document_id": doc_id,
                        "original_filename": filename,
                        "file_path": doc.metadata.get("file_path", ""),
                        "file_type": doc.metadata.get("file_type", "unknown"),
                        "created_at": doc.metadata.get("created_at", ""),
                        "chunks": 0,
                        "total_content_length": 0
                    }
                
                doc_groups[doc_id]["chunks"] += 1
                doc_groups[doc_id]["total_content_length"] += len(doc.content)
            
            files = []
            for doc_info in doc_groups.values():
                files.append({
                    "document_id": doc_info["document_id"],
                    "original_name": doc_info["original_filename"],
                    "file_type": doc_info["file_type"],
                    "chunks": doc_info["chunks"],
                    "content_length": doc_info["total_content_length"],
                    "uploaded_at": doc_info["created_at"]
                })
            
            return {
                "user_id": user_id,
                "collection_name": collection_name,
                "total_count": len(doc_groups),
                "total_chunks": len(all_docs),
                "files": files
            }
            
        except Exception as db_error:
            logger.warning(f"벡터 DB 조회 실패: {db_error}")
            return {
                "user_id": user_id,
                "collection_name": collection_name,
                "total_count": 0,
                "total_chunks": 0,
                "files": [],
                "message": "아직 업로드된 문서가 없습니다."
            }
            
    except Exception as e:
        logger.error(f"문서 목록 조회 실패: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"문서 목록 조회 중 오류가 발생했습니다: {str(e)}"
        )


# ============================================================================
# 문서 삭제
# ============================================================================

@router.delete("/delete/{document_id}")
async def delete_document(document_id: str, user_id: str = "anonymous") -> DocumentDeleteResponse:
    """문서 삭제"""
    try:
        await vector_store.initialize()
        collection_name = f"documents_{user_id}"
        
        deleted_count = await _delete_document_by_id(collection_name, document_id)
        
        if deleted_count > 0:
            return DocumentDeleteResponse(
                message=f"문서 '{document_id}'가 성공적으로 삭제되었습니다. ({deleted_count}개 청크 삭제됨)",
                deleted_chunks=deleted_count,
                success=True
            )
        else:
            raise HTTPException(
                status_code=404,
                detail=f"문서 '{document_id}'를 찾을 수 없습니다."
            )
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"파일 삭제 실패: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"문서 삭제 중 오류가 발생했습니다: {str(e)}"
        )


@router.delete("/delete-by-name/{filename}")
async def delete_document_by_name(filename: str, user_id: str = "anonymous") -> DocumentDeleteResponse:
    """파일명으로 문서 삭제"""
    try:
        await vector_store.initialize()
        collection_name = f"documents_{user_id}"
        
        deleted_count = await _delete_document_by_filename(collection_name, filename)
        
        if deleted_count > 0:
            return DocumentDeleteResponse(
                message=f"파일 '{filename}'이 성공적으로 삭제되었습니다. ({deleted_count}개 청크 삭제됨)",
                deleted_chunks=deleted_count,
                success=True
            )
        else:
            raise HTTPException(
                status_code=404,
                detail=f"파일 '{filename}'을 찾을 수 없습니다."
            )
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"파일 삭제 실패: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"파일 삭제 중 오류가 발생했습니다: {str(e)}"
        )


@router.delete("/clear-all")
async def clear_all_documents(user_id: str = "anonymous") -> DocumentDeleteResponse:
    """사용자의 모든 문서 삭제"""
    try:
        await vector_store.initialize()
        collection_name = f"documents_{user_id}"
        
        try:
            import asyncio
            collections = await asyncio.to_thread(vector_store.client.get_collections)
            collection_exists = any(col.name == collection_name for col in collections.collections)
            
            if collection_exists:
                await asyncio.to_thread(vector_store.client.delete_collection, collection_name)
                await vector_store.ensure_collection(collection_name)
                
                return DocumentDeleteResponse(
                    message=f"사용자 '{user_id}'의 모든 문서가 성공적으로 삭제되었습니다.",
                    deleted_chunks=0,
                    success=True
                )
            else:
                return DocumentDeleteResponse(
                    message=f"사용자 '{user_id}'의 문서 컬렉션이 존재하지 않습니다.",
                    deleted_chunks=0,
                    success=False
                )
                
        except Exception as collection_error:
            logger.warning(f"컬렉션 삭제 실패: {collection_error}")
            return DocumentDeleteResponse(
                message=f"전체 문서 삭제 실패: {str(collection_error)}",
                deleted_chunks=0,
                success=False
            )
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"전체 파일 삭제 실패: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"전체 문서 삭제 중 오류가 발생했습니다: {str(e)}"
        )


# ============================================================================
# 벡터 DB 상태 및 테스트
# ============================================================================

@router.get("/vector-status")
async def get_vector_status() -> Dict[str, Any]:
    """벡터 DB(Qdrant) 상태 조회"""
    try:
        await vector_store.initialize()
        status = await vector_store.get_collection_info()
        
        return {
            "qdrant_status": "connected",
            "collection_info": status,
            "message": "벡터 DB가 정상적으로 연결되어 있습니다"
        }
        
    except Exception as e:
        logger.error(f"벡터 DB 상태 조회 실패: {e}")
        return {
            "qdrant_status": "error",
            "collection_info": None,
            "message": f"벡터 DB 연결 오류: {str(e)}"
        }


@router.get("/search-test")
async def test_vector_search(query: str = "테스트") -> Dict[str, Any]:
    """벡터 검색 테스트"""
    try:
        from app.core import embedding_manager, hybrid_rag_engine
        
        await hybrid_rag_engine.initialize()
        
        query_embedding = await embedding_manager.embed_text(query)
        
        results = await vector_store.search_similar(
            collection_name="documents_test_user",
            query_vector=query_embedding,
            limit=3,
            score_threshold=0.0
        )
        
        return {
            "query": query,
            "results_count": len(results),
            "results": [
                {
                    "content": result.content[:200] + "..." if len(result.content) > 200 else result.content,
                    "score": result.score,
                    "metadata": result.metadata
                }
                for result in results
            ]
        }
        
    except Exception as e:
        logger.error(f"벡터 검색 테스트 실패: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"벡터 검색 테스트 실패: {str(e)}"
        )


# ============================================================================
# WebSocket 진행률 스트리밍
# ============================================================================

@router.websocket("/ws/progress/{document_id}")
async def websocket_progress(websocket: WebSocket, document_id: str):
    """문서 처리 진행률을 실시간으로 스트리밍"""
    try:
        from app.core import progress_websocket
        
        await progress_websocket.connect(websocket, document_id)
        logger.info(f"WebSocket 연결됨: {document_id}")
        
        try:
            while True:
                data = await websocket.receive_text()
                if data == "ping":
                    await websocket.send_text("pong")
                    
        except WebSocketDisconnect:
            logger.info(f"WebSocket 연결 해제됨: {document_id}")
        finally:
            await progress_websocket.disconnect(websocket, document_id)
            
    except Exception as e:
        logger.error(f"WebSocket 오류: {e}")
        try:
            from app.core import progress_websocket
            await progress_websocket.disconnect(websocket, document_id)
        except:
            pass


# ============================================================================
# 헬퍼 함수
# ============================================================================

async def _delete_document_by_id(collection_name: str, document_id: str) -> int:
    """document_id로 문서 삭제 - 필터 사용하여 최적화"""
    try:
        from qdrant_client.models import Filter, FieldCondition, MatchValue
        import asyncio
        
        # 필터 정의 (metadata.document_id)
        filter_condition = Filter(must=[
            FieldCondition(key="metadata.document_id", match=MatchValue(value=document_id))
        ])
        
        # 해당 조건의 모든 포인트 조회
        search_result = await asyncio.to_thread(
            vector_store.client.scroll,
            collection_name=collection_name,
            scroll_filter=filter_condition,
            limit=10000,
            with_payload=False,
            with_vectors=False
        )
        
        points_found = search_result[0] if search_result else []
        
        if points_found:
            point_ids = [point.id for point in points_found]
            
            # 삭제 수행
            from qdrant_client.models import PointIdsList
            await asyncio.to_thread(
                vector_store.client.delete,
                collection_name=collection_name,
                points_selector=PointIdsList(points=point_ids)
            )
            logger.info(f"벡터 DB에서 {len(point_ids)}개 점 삭제 완료: {document_id}")
            return len(point_ids)
        
        return 0
        
    except Exception as e:
        logger.error(f"벡터 DB ID 기반 삭제 실패: {e}")
        return 0


async def _delete_document_by_filename(collection_name: str, filename: str) -> int:
    """파일명으로 문서 삭제"""
    try:
        from qdrant_client.models import Filter, FieldCondition, MatchValue
        
        filters_to_try = [
            Filter(must=[FieldCondition(key="metadata.original_filename", match=MatchValue(value=filename))]),
            Filter(must=[FieldCondition(key="metadata.filename", match=MatchValue(value=filename))]),
        ]
        
        total_deleted = 0
        
        for i, filter_condition in enumerate(filters_to_try):
            try:
                import asyncio
                search_result = await asyncio.to_thread(
                    vector_store.client.scroll,
                    collection_name=collection_name,
                    scroll_filter=filter_condition,
                    limit=10000,
                    with_payload=True,
                    with_vectors=False
                )
                
                points_found = search_result[0] if search_result else []
                
                if points_found:
                    point_ids = [point.id for point in points_found]
                    
                    from qdrant_client.models import PointIdsList
                    await asyncio.to_thread(
                        vector_store.client.delete,
                        collection_name=collection_name,
                        points_selector=PointIdsList(points=point_ids)
                    )
                    
                    total_deleted = len(point_ids)
                    logger.info(f"'{filename}' 삭제 성공: {total_deleted}개 점 삭제됨")
                    return total_deleted
                    
            except Exception as filter_error:
                logger.warning(f"필터 {i+1} 삭제 시도 실패: {filter_error}")
                continue
        
        # Fallback: 전체 검색
        if total_deleted == 0:
            from app.core import embedding_manager
            await embedding_manager.initialize()
            
            test_embedding = await embedding_manager.embed_text("test")
            all_docs = await vector_store.search_similar(
                collection_name=collection_name,
                query_vector=test_embedding,
                limit=10000,
                score_threshold=0.0
            )
            
            points_to_delete = []
            for doc in all_docs:
                doc_filename = doc.metadata.get("filename", "")
                doc_original_filename = doc.metadata.get("original_filename", "")
                
                if (doc_original_filename == filename or 
                    doc_filename == filename or
                    os.path.basename(doc_original_filename) == filename or
                    os.path.basename(doc_filename) == filename):
                    points_to_delete.append(doc.document_id)
            
            if points_to_delete:
                import asyncio
                from qdrant_client.models import PointIdsList
                await asyncio.to_thread(
                    vector_store.client.delete,
                    collection_name=collection_name,
                    points_selector=PointIdsList(points=points_to_delete)
                )
                total_deleted = len(points_to_delete)
        
        return total_deleted
        
    except Exception as e:
        logger.error(f"벡터 DB 파일명 기반 삭제 실패: {e}")
        return 0