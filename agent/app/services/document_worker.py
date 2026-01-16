"""
문서 처리 워커 모듈
백그라운드에서 문서 처리를 담당합니다.
"""
import asyncio
import logging
import time
import os
from pathlib import Path
from typing import Dict, Any, Tuple, List
from datetime import datetime
import uuid

from app.services.processing_task import ProcessingTask, ProcessingProgress

logger = logging.getLogger(__name__)

# 전역 처리 큐
processing_queue = asyncio.Queue()
processing_worker_running = False


async def ensure_processing_worker_running():
    """처리 워커가 실행 중이 아니면 시작"""
    global processing_worker_running
    
    if not processing_worker_running:
        processing_worker_running = True
        asyncio.create_task(document_processing_worker())
        logger.info("📄 문서 처리 워커 시작됨")


async def document_processing_worker():
    """문서 처리를 담당하는 워커 (큐에서 작업을 가져와 처리)"""
    global processing_worker_running
    
    logger.info("🔄 문서 처리 워커 실행 중...")
    print("🔄 문서 처리 워커 시작!", flush=True)
    
    while processing_worker_running:
        try:
            print("📋 큐에서 작업 대기 중...", flush=True)
            task = await asyncio.wait_for(processing_queue.get(), timeout=300.0)
            
            print(f"📋 처리 시작: {task.original_filename} (문서 ID: {task.document_id})", flush=True)
            logger.info(f"📋 처리 시작: {task.original_filename} (문서 ID: {task.document_id})")
            
            progress = ProcessingProgress(task.document_id, task.original_filename)
            print(f"📊 진행 상황 추적 시작: {task.document_id}", flush=True)
            
            await _process_document_with_progress(task, progress)
            
            processing_queue.task_done()
            print(f"✅ 작업 완료: {task.original_filename}", flush=True)
            
        except asyncio.TimeoutError:
            print("⏰ 처리 워커 타임아웃 - 워커 종료", flush=True)
            logger.info("⏰ 처리 워커 타임아웃 - 워커 종료")
            break
        except Exception as e:
            logger.error(f"처리 워커 오류: {e}")
            try:
                processing_queue.task_done()
            except:
                pass
            continue
    
    processing_worker_running = False
    logger.info("🛑 문서 처리 워커 종료됨")


async def _process_document_with_progress(task: ProcessingTask, progress: ProcessingProgress):
    """진행 상황과 함께 문서 처리"""
    try:
        print(f"\n🔄 문서 처리 시작: {task.original_filename}", flush=True)
        logger.info(f"문서 처리 시작: {task.original_filename}")
        
        await _wait_for_websocket_connection(task.document_id, timeout=10)
        
        processing_result = await process_and_store_document(
            file_content=task.file_content,
            file_extension=task.file_extension,
            user_id=task.user_id,
            document_id=task.document_id,
            original_filename=task.original_filename,
            progress=progress
        )
        
        result_data = {
            "text_chunks": processing_result.get("text_chunks", 0),
            "image_chunks": processing_result.get("image_chunks", 0), 
            "total_embeddings": processing_result.get("total_embeddings", 0),
            "processing_time": time.time() - progress.start_time
        }
        
        await _ensure_completion_message_sent(task.document_id, result_data, progress.filename)
        ProcessingProgress.set_completed(task.document_id, result_data)
        
        print(f"\n✅ 문서 처리 완료: {task.original_filename}", flush=True)
        logger.info(f"문서 처리 완료: {task.original_filename}")
        
    except Exception as e:
        print(f"\n❌ 처리 실패: {task.original_filename} - {e}", flush=True)
        logger.error(f"문서 처리 실패: {task.original_filename} - {e}")
        ProcessingProgress.set_failed(task.document_id, str(e))


async def process_and_store_document(
    file_content: bytes,
    file_extension: str,
    user_id: str, 
    document_id: str, 
    original_filename: str,
    progress: ProcessingProgress
) -> Dict[str, Any]:
    """메모리의 파일 내용을 직접 처리하고 Qdrant 벡터 DB에 저장"""
    from app.core.rag_engine import rag_engine
    from app.models.schemas import DocumentChunk
    
    try:
        if not rag_engine._initialized:
            print("\n🔄 RAG 엔진 초기화 시작...", flush=True)
            await rag_engine.initialize()
            print("✅ RAG 엔진 초기화 완료!", flush=True)
        
        text_chunks = 0
        image_chunks = 0
        chunks = []
        
        if file_extension in ['.txt', '.md']:
            chunks = await _process_text_file(
                file_content, document_id, original_filename, 
                progress, rag_engine.embedding_manager
            )
            text_chunks = len(chunks)
            
        elif file_extension == '.pdf':
            chunks, image_count = await _process_pdf_file(
                file_content, document_id, original_filename,
                progress, rag_engine.embedding_manager
            )
            text_chunks = len([c for c in chunks if c.metadata.get('content_type') == 'text'])
            image_chunks = len([c for c in chunks if c.metadata.get('content_type') == 'image'])
            
        else:
            # 기타 파일은 텍스트로 처리 시도
            chunks = await _process_text_file(
                file_content, document_id, original_filename,
                progress, rag_engine.embedding_manager
            )
            text_chunks = len(chunks)
        
        # 벡터 DB에 저장
        if chunks:
            await progress.start_step_async(5)
            await progress.update_step_progress_async(50.0)
            
            await rag_engine.vector_store.add_documents(chunks, user_id)
            
            await progress.complete_step_async()
            print(f"\n💾 Qdrant에 {len(chunks):,}개 청크 저장 완료", flush=True)
        
        # 최종 진행률 업데이트
        progress.current_step_index = progress.total_steps - 1
        progress.step_progress = 100.0
        await progress._send_websocket_progress_async()
        
        return {
            "text_chunks": text_chunks,
            "image_chunks": image_chunks,
            "total_embeddings": len(chunks)
        }
        
    except Exception as e:
        logger.error(f"문서 처리 및 저장 실패: {e}")
        raise


async def _process_text_file(
    file_content: bytes,
    document_id: str,
    original_filename: str,
    progress: ProcessingProgress,
    embedding_manager
) -> List:
    """텍스트 파일 처리"""
    from app.models.schemas import DocumentChunk
    
    await progress.start_step_async(1)
    await progress.complete_step_async()
    
    await progress.start_step_async(2)
    
    # 인코딩 감지 및 디코딩
    try:
        content = file_content.decode('utf-8')
    except UnicodeDecodeError:
        try:
            content = file_content.decode('cp949')
        except UnicodeDecodeError:
            content = file_content.decode('latin-1', errors='ignore')
    
    await progress.update_step_progress_async(50.0)
    
    # 청킹
    chunks = []
    chunk_size = 1000
    
    for i in range(0, len(content), chunk_size):
        chunk_text = content[i:i+chunk_size].strip()
        if not chunk_text:
            continue
            
        embedding = await embedding_manager.embed_text(chunk_text)
        
        chunk = DocumentChunk(
            id=str(uuid.uuid4()),
            content=chunk_text,
            embedding=embedding,
            metadata={
                "document_id": document_id,
                "original_filename": original_filename,
                "chunk_index": len(chunks),
                "file_type": "text",
                "content_type": "text",
                "created_at": str(datetime.now())
            }
        )
        chunks.append(chunk)
    
    await progress.complete_step_async()
    
    # 이미지 관련 단계 건너뛰기
    await progress.start_step_async(3)
    await progress.complete_step_async()
    await progress.start_step_async(4)
    await progress.complete_step_async()
    
    return chunks


async def _process_pdf_file(
    file_content: bytes,
    document_id: str,
    original_filename: str,
    progress: ProcessingProgress,
    embedding_manager
) -> Tuple[List, int]:
    """PDF 파일 처리"""
    import tempfile
    import fitz  # PyMuPDF
    from app.models.schemas import DocumentChunk
    
    chunks = []
    image_count = 0
    
    await progress.start_step_async(1)
    await progress.update_step_progress_async(20.0)
    
    # 임시 파일로 저장
    with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as temp_file:
        temp_file.write(file_content)
        temp_path = temp_file.name
    
    await progress.update_step_progress_async(60.0)
    
    try:
        pdf_document = fitz.open(temp_path)
        total_pages = len(pdf_document)
        
        await progress.start_step_async(2)
        
        # 텍스트 추출
        for page_num in range(min(total_pages, 1000)):
            try:
                page = pdf_document[page_num]
                
                text_progress = (page_num / min(total_pages, 1000)) * 99.0
                await progress.update_step_progress_async(text_progress)
                
                if page_num % 10 == 0:
                    await asyncio.sleep(0.05)
                else:
                    await asyncio.sleep(0.001)
                
                page_text = page.get_text().strip()
                if page_text and len(page_text) > 20:
                    chunk_size = 2000
                    
                    for i in range(0, len(page_text), chunk_size):
                        chunk_text = page_text[i:i+chunk_size].strip()
                        if chunk_text and len(chunk_text) > 10:
                            await asyncio.sleep(0.001)
                            embedding = await embedding_manager.embed_text(chunk_text)
                            
                            chunk = DocumentChunk(
                                id=str(uuid.uuid4()),
                                content=chunk_text,
                                embedding=embedding,
                                metadata={
                                    "document_id": document_id,
                                    "original_filename": original_filename,
                                    "page": page_num + 1,
                                    "chunk_index": len(chunks),
                                    "content_type": "text",
                                    "file_type": "pdf",
                                    "created_at": str(datetime.now())
                                }
                            )
                            chunks.append(chunk)
                            
            except Exception as page_error:
                logger.warning(f"PDF 페이지 {page_num + 1} 처리 실패: {page_error}")
                continue
        
        await progress.complete_step_async()
        
        # 이미지 처리
        await progress.start_step_async(3)
        
        for page_num in range(min(total_pages, 100)):
            try:
                page = pdf_document[page_num]
                image_list = page.get_images()
                
                for img_index, img in enumerate(image_list[:5]):
                    try:
                        xref = img[0]
                        pix = fitz.Pixmap(pdf_document, xref)
                        
                        if pix.n - pix.alpha < 4:
                            image_chunk = DocumentChunk(
                                id=str(uuid.uuid4()),
                                content=f"Image from page {page_num + 1}, image {img_index + 1}",
                                embedding=[0.0] * 768,
                                metadata={
                                    "document_id": document_id,
                                    "original_filename": original_filename,
                                    "page": page_num + 1,
                                    "image_index": img_index,
                                    "content_type": "image",
                                    "file_type": "pdf",
                                    "created_at": str(datetime.now())
                                }
                            )
                            chunks.append(image_chunk)
                            image_count += 1
                            
                        pix = None
                        
                    except Exception as img_error:
                        logger.warning(f"이미지 처리 실패: {img_error}")
                        continue
                        
                image_progress = (page_num / min(total_pages, 100)) * 100
                await progress.update_step_progress_async(image_progress)
                
            except Exception as page_error:
                logger.warning(f"페이지 이미지 처리 실패: {page_error}")
                continue
        
        await progress.complete_step_async()
        
        await progress.start_step_async(4)
        await progress.complete_step_async()
        
        pdf_document.close()
        
    finally:
        if os.path.exists(temp_path):
            os.unlink(temp_path)
    
    return chunks, image_count


async def _wait_for_websocket_connection(document_id: str, timeout: int = 10):
    """WebSocket 연결을 기다림"""
    from app.core.websocket_manager import progress_websocket
    
    start_time = time.time()
    while time.time() - start_time < timeout:
        if document_id in progress_websocket.connections and len(progress_websocket.connections[document_id]) > 0:
            print(f"✅ WebSocket 연결 확인됨: {document_id}", flush=True)
            return True
        
        print(f"⏳ WebSocket 연결 대기 중... ({int(time.time() - start_time)}초)", flush=True)
        await asyncio.sleep(1)
    
    print(f"⚠️ WebSocket 연결 타임아웃: {document_id}", flush=True)
    return False


async def _ensure_completion_message_sent(document_id: str, result_data: dict, filename: str):
    """완료 메시지가 확실히 전송되도록 보장"""
    from app.core.websocket_manager import progress_websocket
    
    max_retries = 5
    retry_count = 0
    
    while retry_count < max_retries:
        try:
            if document_id in progress_websocket.connections and len(progress_websocket.connections[document_id]) > 0:
                print(f"📡 완료 메시지 전송 시도 {retry_count + 1}/{max_retries}", flush=True)
                
                await progress_websocket.send_completion(
                    document_id,
                    "completed", 
                    f"문서 처리가 완료되었습니다: {filename}",
                    result_data
                )
                
                print(f"✅ 완료 메시지 전송 성공!", flush=True)
                return True
            else:
                print(f"📡 WebSocket 연결 없음, 재연결 대기...", flush=True)
                
            await asyncio.sleep(2)
            retry_count += 1
            
        except Exception as e:
            print(f"⚠️ 완료 메시지 전송 실패: {e}", flush=True)
            retry_count += 1
            await asyncio.sleep(1)
    
    print(f"❌ 완료 메시지 전송 최종 실패: {document_id}", flush=True)
    return False
