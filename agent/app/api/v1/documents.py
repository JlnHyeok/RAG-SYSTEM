from fastapi import APIRouter, HTTPException, UploadFile, File, Form, WebSocket, WebSocketDisconnect
from typing import Dict, Any, Optional, Tuple
import logging
import time
import hashlib
import shutil
import os
import threading
import asyncio
from pathlib import Path
from dataclasses import dataclass
from queue import Queue
import uuid

# TOKENIZERS_PARALLELISM 경고 해결
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# 전역 처리 큐
processing_queue = asyncio.Queue()
processing_worker_running = False

@dataclass
class ProcessingTask:
    """처리할 작업 정의"""
    task_id: str
    document_id: str
    file_content: bytes
    file_extension: str
    user_id: str
    original_filename: str
    created_at: float
    
    def __post_init__(self):
        if not self.task_id:
            self.task_id = str(uuid.uuid4())

class ProcessingProgress:
    """문서 처리 진행 상황을 추적하는 클래스"""
    
    # 전역 처리 상태 저장소
    _progress_store = {}
    
    def __init__(self, document_id: str, filename: str):
        self.document_id = document_id
        self.filename = filename
        self.current_step = ""
        self.step_progress = 0.0
        self.total_steps = 6
        self.current_step_index = 0
        self.status = "processing"  # processing, completed, failed
        self.start_time = time.time()
        self.result_data = {}
        
        self.steps = [
            "📤 파일 업로드",
            "📖 PDF 파싱", 
            "✂️ 텍스트 추출 및 청킹",
            "🖼️ 이미지 추출",
            "👁️ OCR 처리", 
            "🧠 임베딩 생성 및 벡터 저장"
        ]
        
        # 전역 저장소에 저장
        ProcessingProgress._progress_store[document_id] = self
        
    @staticmethod
    def get_progress(document_id: str) -> Optional[Dict[str, Any]]:
        """문서 처리 상태 조회"""
        progress = ProcessingProgress._progress_store.get(document_id)
        if not progress:
            return None
            
        overall_progress = (progress.current_step_index + progress.step_progress / 100.0) / progress.total_steps * 100
        
        return {
            "document_id": progress.document_id,
            "filename": progress.filename,
            "status": progress.status,
            "current_step": progress.current_step,
            "current_step_index": progress.current_step_index,
            "step_progress": progress.step_progress,
            "overall_progress": overall_progress,
            "total_steps": progress.total_steps,
            "elapsed_time": time.time() - progress.start_time,
            "result_data": progress.result_data
        }
    
    @staticmethod
    def set_completed(document_id: str, result_data: Dict[str, Any]):
        """처리 완료 상태로 변경"""
        progress = ProcessingProgress._progress_store.get(document_id)
        if progress:
            progress.status = "completed"
            progress.result_data = result_data
            progress.current_step_index = progress.total_steps - 1
            progress.step_progress = 100.0
            
            # WebSocket 완료 알림 전송 (간소화)
            try:
                import asyncio
                from app.core.websocket_manager import progress_websocket
                
                try:
                    loop = asyncio.get_running_loop()
                    print(f"📡 완료 메시지 전송 준비: {document_id} -> {result_data}", flush=True)
                    task = loop.create_task(progress_websocket.send_completion(
                        document_id,
                        "completed",
                        f"문서 처리가 완료되었습니다: {progress.filename}",
                        result_data
                    ))
                    print(f"📡 완료 메시지 전송 태스크 생성됨", flush=True)
                except RuntimeError:
                    pass
            except Exception:
                pass
    
    @staticmethod  
    def set_failed(document_id: str, error_message: str):
        """처리 실패 상태로 변경"""
        progress = ProcessingProgress._progress_store.get(document_id)
        if progress:
            progress.status = "failed"
            progress.result_data = {"error": error_message}
            
            # WebSocket 실패 알림 전송
            try:
                import asyncio
                from app.core.websocket_manager import progress_websocket
                
                try:
                    loop = asyncio.get_running_loop()
                    loop.create_task(progress_websocket.send_completion(
                        document_id,
                        "failed",
                        f"문서 처리 실패: {error_message}",
                        {"error": error_message}
                    ))
                    print(f"❌ 문서 처리 실패 WebSocket 알림 전송: {document_id}", flush=True)
                except RuntimeError:
                    print(f"⚠️ 이벤트 루프가 없어 WebSocket 알림 건너뛰: {document_id}", flush=True)
            except Exception as e:
                print(f"⚠️ WebSocket 실패 알림 실패: {e}", flush=True)
        
    def start_step(self, step_index: int):
        """단계 시작 (동기)"""
        self.current_step_index = step_index
        self.current_step = self.steps[step_index]
        self.step_progress = 0.0
        self._log_progress()
        self._send_websocket_progress()

    async def start_step_async(self, step_index: int):
        """단계 시작 (비동기)"""
        self.current_step_index = step_index
        self.current_step = self.steps[step_index]
        self.step_progress = 0.0
        self._log_progress()
        await self._send_websocket_progress_async()
        
    def update_step_progress(self, progress: float):
        """현재 단계 진행률 업데이트 (동기)"""
        self.step_progress = min(100.0, max(0.0, progress))
        self._log_progress()
        self._send_websocket_progress()

    async def update_step_progress_async(self, progress: float):
        """현재 단계 진행률 업데이트 (비동기)"""
        self.step_progress = min(100.0, max(0.0, progress))
        self._log_progress()
        await self._send_websocket_progress_async()
        
    def complete_step(self):
        """현재 단계 완료 (동기)"""
        self.step_progress = 100.0
        self._log_progress()
        self._send_websocket_progress()

    async def complete_step_async(self):
        """현재 단계 완료 (비동기)"""
        import asyncio
        print(f"🏁 단계 완료 시작: {self.current_step}", flush=True)
        self.step_progress = 100.0
        self._log_progress()
        
        # WebSocket 메시지 전송
        await self._send_websocket_progress_async()
        
        # 완료 후 약간의 대기 (메시지 전송 보장)
        await asyncio.sleep(0.1)
        print(f"✅ 단계 완료됨: {self.current_step}", flush=True)
    
    def _send_websocket_progress(self):
        """WebSocket으로 진행률 전송 (동기 - 백워드 호환성)"""
        try:
            import asyncio
            
            # 현재 이벤트 루프가 있는지 확인
            try:
                loop = asyncio.get_running_loop()
                # 즉시 실행되도록 태스크 생성
                task = loop.create_task(self._send_websocket_progress_async())
                
            except RuntimeError:
                # 이벤트 루프가 없으면 무시
                pass
        except Exception as e:
            # WebSocket 에러는 무시하고 계속 진행
            pass

    async def _send_websocket_progress_async(self):
        """WebSocket으로 진행률 전송 (즉시 실행)"""
        try:
            from app.core.websocket_manager import progress_websocket
            
            # 전체 진행률 계산
            overall_progress = (self.current_step_index + self.step_progress / 100.0) / self.total_steps * 100
            
            print(f"🔍 [DEBUG] WebSocket 전송 시도: document_id={self.document_id}", flush=True)
            print(f"🔍 [DEBUG] 단계: {self.current_step}, 진행률: {self.step_progress:.1f}%, 전체: {overall_progress:.1f}%", flush=True)
            
            await progress_websocket.send_progress(
                self.document_id,
                self.current_step,
                self.step_progress,
                overall_progress,
                f"{self.current_step_index + 1}/{self.total_steps} - {self.step_progress:.1f}%"
            )
            print(f"📡 WebSocket 전송: {self.current_step} - {self.step_progress:.1f}% (전체: {overall_progress:.1f}%)", flush=True)
                
        except Exception as e:
            print(f"⚠️ WebSocket 전송 실패: {e}", flush=True)
            import traceback
            traceback.print_exc()
            pass

    def _log_progress(self):
        """진행 상황을 로그로 출력"""
        overall_progress = (self.current_step_index + self.step_progress / 100.0) / self.total_steps * 100
        
        progress_bar = self._create_progress_bar(self.step_progress)
        
        # 터미널과 로그 모두에 출력
        progress_msg = f"📋 처리 중: {self.filename}"
        step_msg = f"🔄 {self.current_step}"
        step_progress_msg = f"📊 단계 진행률: {progress_bar} {self.step_progress:.1f}%"
        overall_progress_msg = f"📈 전체 진행률: {overall_progress:.1f}% ({self.current_step_index + 1}/{self.total_steps})"
        separator = "=" * 70
        
        # 터미널 실시간 출력
        print(f"\n{progress_msg}", flush=True)
        print(step_msg, flush=True)
        print(step_progress_msg, flush=True)
        print(overall_progress_msg, flush=True)
        print(separator, flush=True)
        
        # 로그 파일에도 기록
        logger.info(f"\n{progress_msg}")
        logger.info(step_msg)
        logger.info(step_progress_msg)
        logger.info(overall_progress_msg)
        logger.info(separator)
        
    def _create_progress_bar(self, progress: float, width: int = 30) -> str:
        """진행률 바 생성"""
        filled = int(width * progress / 100)
        bar = "█" * filled + "░" * (width - filled)
        return f"[{bar}]"

from app.core.document_processor import document_processor
from app.core.vector_store import vector_store
from app.models.schemas import ProcessingResult, DocumentUploadResponse, DocumentDeleteResponse
from app.core.config import settings

logger = logging.getLogger(__name__)
router = APIRouter()

# 문서 처리기 (전역 인스턴스)
processor = None


@router.get("/upload/{document_id}/status")
async def get_upload_status(document_id: str):
    """문서 처리 상태 확인"""
    try:
        print(f"📊 상태 조회 요청: {document_id}", flush=True)
        print(f"📊 저장된 진행률 개수: {len(ProcessingProgress._progress_store)}", flush=True)
        print(f"📊 저장된 키들: {list(ProcessingProgress._progress_store.keys())}", flush=True)
        
        # ProcessingProgress에서 현재 상태 조회
        progress_data = ProcessingProgress.get_progress(document_id)
        
        if not progress_data:
            logger.warning(f"진행률 정보를 찾을 수 없음: {document_id}")
            raise HTTPException(
                status_code=404,
                detail="문서를 찾을 수 없습니다"
            )
        
        print(f"📊 상태 조회 성공: {progress_data}", flush=True)
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
        max_size = getattr(settings, 'MAX_FILE_SIZE', 52428800)  # 50MB
        if hasattr(file, 'size') and file.size > max_size:
            raise HTTPException(
                status_code=413, 
                detail=f"파일이 너무 큽니다. 최대 {max_size // 1024 // 1024}MB까지 지원합니다."
            )
        
        # 고유 문서 ID 생성
        file_hash = hashlib.md5(f"{user_id}_{file.filename}_{time.time()}".encode()).hexdigest()
        
        # 진행 상황 추적 시작
        progress = ProcessingProgress(file_hash, file.filename)
        await progress.start_step_async(0)  # 파일 업로드
        
        # 파일 내용을 메모리에서 직접 읽기
        file_content = await file.read()
        await progress.complete_step_async()
        
        print(f"\n✅ 파일 업로드 완료: {file.filename} (크기: {len(file_content):,} bytes)", flush=True)
        logger.info(f"파일 업로드 완료: {file.filename} (크기: {len(file_content)} bytes, 사용자: {user_id})")
        
        # 처리 작업을 큐에 추가
        task = ProcessingTask(
            task_id="",  # __post_init__에서 자동 생성
            document_id=file_hash,
            file_content=file_content,
            file_extension=file_extension,
            user_id=user_id,
            original_filename=file.filename,
            created_at=time.time()
        )
        
        # 큐에 작업 추가
        await processing_queue.put(task)
        
        # 처리 워커가 실행 중이 아니면 시작
        await ensure_processing_worker_running()
        
        print(f"📋 처리 작업 큐에 추가됨: {file.filename} (작업 ID: {task.task_id})", flush=True)
        
        # 업로드 완료 후 즉시 응답 반환 (처리는 워커에서 진행)
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
            # 큐에서 작업 가져오기 (타임아웃 설정 - 더 길게)
            task = await asyncio.wait_for(processing_queue.get(), timeout=300.0)  # 5분으로 연장
            
            print(f"📋 처리 시작: {task.original_filename} (문서 ID: {task.document_id})", flush=True)
            logger.info(f"📋 처리 시작: {task.original_filename} (문서 ID: {task.document_id})")
            
            # 진행 상황 추적 시작
            progress = ProcessingProgress(task.document_id, task.original_filename)
            print(f"📊 진행 상황 추적 시작: {task.document_id}", flush=True)
            
            # 문서 처리 실행
            await _process_document_with_progress(task, progress)
            
            # 작업 완료 표시
            processing_queue.task_done()
            print(f"✅ 작업 완료: {task.original_filename}", flush=True)
            
        except asyncio.TimeoutError:
            # 5분 동안 새 작업이 없으면 워커 종료
            print("⏰ 처리 워커 타임아웃 - 워커 종료", flush=True)
            logger.info("⏰ 처리 워커 타임아웃 - 워커 종료")
            break
        except Exception as e:
            logger.error(f"처리 워커 오류: {e}")
            # 작업 완료 표시 (오류 발생해도)
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
        print(f"📊 진행 상황 객체 생성: {progress.document_id}", flush=True)
        logger.info(f"문서 처리 시작: {task.original_filename}")
        
        # WebSocket 연결 확인 및 대기
        await _wait_for_websocket_connection(task.document_id, timeout=10)
        
        # 기존 문서 처리 로직 실행
        print(f"🚀 _process_and_store_document_from_memory 호출 시작", flush=True)
        processing_result = await _process_and_store_document_from_memory(
            file_content=task.file_content,
            file_extension=task.file_extension,
            user_id=task.user_id,
            document_id=task.document_id,
            original_filename=task.original_filename,
            progress=progress
        )
        print(f"🚀 _process_and_store_document_from_memory 완료: {processing_result}", flush=True)
        
        # 완료 상태 업데이트
        result_data = {
            "text_chunks": processing_result.get("text_chunks", 0),
            "image_chunks": processing_result.get("image_chunks", 0), 
            "total_embeddings": processing_result.get("total_embeddings", 0),
            "processing_time": time.time() - progress.start_time
        }
        print(f"📊 완료 상태 업데이트: {result_data}", flush=True)
        
        # WebSocket 연결을 다시 확인하고 완료 메시지 전송
        await _ensure_completion_message_sent(task.document_id, result_data, progress.filename)
        
        ProcessingProgress.set_completed(task.document_id, result_data)
        
        print(f"\n✅ 문서 처리 완료: {task.original_filename}", flush=True)
        logger.info(f"문서 처리 완료: {task.original_filename}")
        
    except Exception as e:
        print(f"\n❌ 처리 실패: {task.original_filename} - {e}", flush=True)
        logger.error(f"문서 처리 실패: {task.original_filename} - {e}")
        
        # 실패 상태 업데이트
        ProcessingProgress.set_failed(task.document_id, str(e))


@router.post("/process-document", response_model=Dict[str, Any])
async def process_document(
    file_path: str,
    user_id: str
) -> Dict[str, Any]:
    """Backend에서 호출하는 문서 처리 엔드포인트"""
    global processor
    
    start_time = time.time()
    
    try:
        # 입력 검증
        if not file_path.strip():
            raise HTTPException(status_code=400, detail="파일 경로가 필요합니다")
        
        if not user_id.strip():
            raise HTTPException(status_code=400, detail="사용자 ID가 필요합니다")
        
        # 파일 존재 확인
        if not Path(file_path).exists():
            raise HTTPException(status_code=404, detail=f"파일을 찾을 수 없습니다: {file_path}")
        
        logger.info(f"문서 처리 시작: {file_path} (사용자: {user_id})")
        
        # 문서 처리기 초기화 (지연 로딩)
        if processor is None:
            # TODO: 실제 MultiModalDocumentProcessor 구현 후 사용
            # processor = MultiModalDocumentProcessor()
            # await processor.initialize()
            pass
        
        # 임시 구현: 간단한 텍스트 처리
        result = await _process_simple_document(file_path, user_id)
        
        # 벡터 DB에 저장
        collection_name = f"documents_{user_id}"
        
        # 임베딩이 있는 청크들만 저장
        all_chunks = result.text_chunks + result.image_chunks
        if all_chunks:
            stored_count = await vector_store.store_embeddings(collection_name, all_chunks)
        else:
            stored_count = 0
        
        processing_time = time.time() - start_time
        
        # 문서 ID 생성
        document_id = hashlib.md5(f"{file_path}_{user_id}".encode()).hexdigest()
        
        response = {
            "document_id": document_id,
            "status": "processed",
            "text_chunks": len(result.text_chunks),
            "image_chunks": len(result.image_chunks),
            "total_embeddings": stored_count,
            "processing_time": processing_time
        }
        
        logger.info(f"문서 처리 완료: {response}")
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"문서 처리 실패: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"문서 처리 중 오류가 발생했습니다: {str(e)}"
        )


async def _process_and_store_document_from_memory(
    file_content: bytes,
    file_extension: str,
    user_id: str, 
    document_id: str, 
    original_filename: str,
    progress: ProcessingProgress
) -> Dict[str, Any]:
    """메모리의 파일 내용을 직접 처리하고 Qdrant 벡터 DB에 저장"""
    from app.core.embedding_manager import embedding_manager
    from app.models.schemas import DocumentChunk
    import uuid
    from datetime import datetime
    
    try:
        # 이미 초기화된 서비스들 사용
        from app.core.rag_engine import rag_engine
        
        # RAG 엔진이 초기화되어 있는지 확인
        if not rag_engine._initialized:
            print("\n🔄 RAG 엔진 초기화 시작...", flush=True)
            logger.info("RAG 엔진이 초기화되지 않음. 초기화 시작...")
            await rag_engine.initialize()
            print("✅ RAG 엔진 초기화 완료!", flush=True)
        else:
            print("✅ 이미 초기화된 RAG 엔진 사용", flush=True)
            logger.info("이미 초기화된 RAG 엔진 사용")
        
        text_chunks = 0
        image_chunks = 0
        chunks = []
        
        if file_extension in ['.txt', '.md']:
            # 텍스트 파일 처리
            await progress.start_step_async(1)  # PDF 파싱 (텍스트는 건너뛴)
            await progress.complete_step_async()
            
            await progress.start_step_async(2)  # 텍스트 추출 및 청킹
            
            try:
                content = file_content.decode('utf-8')
            except UnicodeDecodeError:
                # UTF-8 디코딩 실패 시 다른 인코딩 시도
                try:
                    content = file_content.decode('cp949')  # 한국어 인코딩
                except UnicodeDecodeError:
                    content = file_content.decode('latin-1', errors='ignore')
            
            await progress.update_step_progress_async(50.0)
            await progress._send_websocket_progress_async()
            
            chunks = await _process_text_content_from_string(
                content, document_id, original_filename, rag_engine.embedding_manager
            )
            text_chunks = len(chunks)
            
            await progress.complete_step_async()
            await progress._send_websocket_progress_async()
            
            # 이미지 추출 단계는 건너뛴
            await progress.start_step_async(3)  # 이미지 추출
            await progress.complete_step_async()
            await progress.start_step_async(4)  # OCR 처리
            await progress.complete_step_async()
            
        elif file_extension == '.pdf':
            # PDF 파일을 임시로 저장해서 처리 (PyMuPDF 등이 파일 경로 필요)
            import tempfile
            import os
            
            await progress.start_step_async(1)  # PDF 파싱
            await progress.update_step_progress_async(20.0)
            await progress._send_websocket_progress_async()
            
            # 이벤트 루프 양보
            import asyncio
            await asyncio.sleep(0.01)
            
            with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as temp_file:
                temp_file.write(file_content)
                temp_path = temp_file.name
            
            await progress.update_step_progress_async(60.0)
            await progress._send_websocket_progress_async()
            
            try:
                chunks, image_count = await _process_pdf_with_images(
                    temp_path, document_id, original_filename, progress, rag_engine.embedding_manager
                )
                text_chunks = len([c for c in chunks if c.metadata.get('content_type') == 'text'])
                image_chunks = len([c for c in chunks if c.metadata.get('content_type') == 'image'])
            finally:
                # 임시 파일 삭제
                if os.path.exists(temp_path):
                    os.unlink(temp_path)
            
            progress.complete_step()
            await progress._send_websocket_progress_async()
                    
        else:
            # 기타 파일은 텍스트로 처리 시도
            await progress.start_step_async(1)  # PDF 파싱 (건너뛴)
            await progress.complete_step_async()
            
            await progress.start_step_async(2)  # 텍스트 추출 및 청킹
            
            try:
                content = file_content.decode('utf-8', errors='ignore')
                chunks = await _process_text_content_from_string(
                    content, document_id, original_filename, rag_engine.embedding_manager
                )
                text_chunks = len(chunks)
            except Exception as e:
                logger.warning(f"지원하지 않는 파일 형식 처리 실패: {file_extension}, {e}")
                chunks = []
            
            await progress.complete_step_async()
            
            # 이미지 추출 단계는 건너뛴
            await progress.start_step_async(3)  # 이미지 추출
            await progress.complete_step_async()
            await progress.start_step_async(4)  # OCR 처리
            await progress.complete_step_async()
        
        # 벡터 DB에 저장
        if chunks:
            await progress.start_step_async(5)  # 벡터 저장
            await progress.update_step_progress_async(50.0)
            await progress._send_websocket_progress_async()
            
            # 이벤트 루프 양보
            await asyncio.sleep(0.01)
            
            await rag_engine.vector_store.add_documents(chunks, user_id)
            
            await progress.complete_step_async()
            await progress._send_websocket_progress_async()
            print(f"\n💾 Qdrant에 {len(chunks):,}개 청크 저장 완료: {original_filename}", flush=True)
            logger.info(f"Qdrant에 {len(chunks)}개 청크 저장 완료: {original_filename}")
        
        # 최종 WebSocket 메시지 전송을 위한 이벤트 루프 양보
        import asyncio
        await asyncio.sleep(0.01)
        
        # 모든 단계 완료 - 전체 진행률 100%로 설정
        progress.current_step_index = progress.total_steps - 1
        progress.step_progress = 100.0
        await progress._send_websocket_progress_async()
        
        # 최종 완료 메시지 전송을 위한 추가 시간
        await asyncio.sleep(0.02)
        
        return {
            "text_chunks": text_chunks,
            "image_chunks": image_chunks,
            "total_embeddings": len(chunks)
        }
        
    except Exception as e:
        logger.error(f"메모리 파일 처리 및 저장 실패: {e}")
        raise


async def _process_text_content_from_string(
    content: str, 
    document_id: str, 
    original_filename: str,
    embedding_manager
) -> list:
    """문자열 콘텐츠를 청크로 나누고 임베딩 생성 (파일 경로 없이)"""
    from app.models.schemas import DocumentChunk
    import uuid
    from datetime import datetime
    
    chunks = []
    chunk_size = 1000  # 1000자 단위로 청킹
    
    for i in range(0, len(content), chunk_size):
        chunk_text = content[i:i+chunk_size].strip()
        if not chunk_text:
            continue
            
        # 임베딩 생성
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
                "created_at": str(datetime.now())
            }
        )
        chunks.append(chunk)
    
    return chunks


async def _process_and_store_document(
    file_path: str, 
    user_id: str, 
    document_id: str, 
    original_filename: str
) -> Dict[str, Any]:
    """문서를 처리하고 Qdrant 벡터 DB에 저장"""
    from app.core.embedding_manager import embedding_manager
    from app.models.schemas import DocumentChunk
    import uuid
    from datetime import datetime
    
    try:
        # 임베딩 매니저와 벡터 스토어 초기화
        await embedding_manager.initialize()
        await vector_store.initialize()
        
        # 파일 확장자에 따른 처리
        file_extension = Path(file_path).suffix.lower()
        text_chunks = 0
        image_chunks = 0
        
        if file_extension == '.txt' or file_extension == '.md':
            # 텍스트 파일 처리
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            chunks = await _process_text_content(
                content, file_path, document_id, original_filename
            )
            text_chunks = len(chunks)
            
        elif file_extension == '.pdf':
            # PDF 파일 처리 (텍스트 + 이미지 + OCR)
            chunks, image_count = await _process_pdf_with_images(
                file_path, document_id, original_filename
            )
            text_chunks = len([c for c in chunks if c.metadata.get('content_type') == 'text'])
            image_chunks = len([c for c in chunks if c.metadata.get('content_type') == 'image'])
                    
        else:
            # 기타 파일은 텍스트로 읽기 시도
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                chunks = await _process_text_content(
                    content, file_path, document_id, original_filename
                )
                text_chunks = len(chunks)
            except:
                logger.warning(f"지원하지 않는 파일 형식: {file_extension}")
                chunks = []
        
        # 벡터 DB에 저장
        if chunks:
            await vector_store.add_documents(chunks, user_id)
            logger.info(f"Qdrant에 {len(chunks)}개 청크 저장 완료: {original_filename}")
        
        return {
            "text_chunks": text_chunks,
            "image_chunks": image_chunks,
            "total_embeddings": len(chunks)
        }
        
    except Exception as e:
        logger.error(f"문서 처리 및 저장 실패: {e}")
        raise


async def _process_text_content(
    content: str, 
    file_path: str, 
    document_id: str, 
    original_filename: str
) -> list:
    """텍스트 콘텐츠를 청크로 나누고 임베딩 생성"""
    from app.core.embedding_manager import embedding_manager
    from app.models.schemas import DocumentChunk
    import uuid
    from datetime import datetime
    
    chunks = []
    chunk_size = 1000  # 1000자 단위로 청킹
    
    for i in range(0, len(content), chunk_size):
        chunk_text = content[i:i+chunk_size].strip()
        if not chunk_text:
            continue
            
        # 임베딩 생성
        embedding = await embedding_manager.embed_text(chunk_text)
        
        chunk = DocumentChunk(
            id=str(uuid.uuid4()),
            content=chunk_text,
            embedding=embedding,
            metadata={
                "document_id": document_id,
                "file_path": file_path,
                "original_filename": original_filename,
                "chunk_index": len(chunks),
                "file_type": "text",
                "created_at": str(datetime.now())
            }
        )
        chunks.append(chunk)
    
    return chunks


async def _process_pdf_with_images(
    file_path: str, 
    document_id: str, 
    original_filename: str,
    progress: ProcessingProgress,
    embedding_manager
) -> Tuple[list, int]:
    """고급 PDF 처리: 텍스트와 이미지를 모두 처리 (OCR 포함)"""
    from app.core.document_processor import document_processor
    from app.models.schemas import DocumentChunk
    import uuid
    from datetime import datetime
    import fitz  # PyMuPDF
    
    chunks = []
    image_count = 0
    
    try:
        # PyMuPDF를 사용해서 PDF 열기 (이미지 추출 가능)
        pdf_document = fitz.open(file_path)
        total_pages = len(pdf_document)
        
        progress.start_step(2)  # 텍스트 추출 및 청킹
        
        # 텍스트 추출 및 청킹 단계 - 대용량 파일 대응
        processed_pages = 0
        max_chunks_per_batch = 20  # 배치 크기 증가 (대용량 처리용)
        batch_chunks = []
        
        for page_num in range(min(total_pages, 1000)):  # 최대 1000페이지로 제한
            try:
                page = pdf_document[page_num]
                
                # 페이지별 진행률 계산 및 업데이트 (마지막 페이지 고려)
                total_to_process = min(total_pages, 1000)
                if page_num == total_to_process - 1:  # 마지막 페이지
                    text_progress = 99.0  # 마지막은 99%로 설정
                else:
                    text_progress = (page_num / total_to_process) * 99.0  # 99%까지만 진행
                await progress.update_step_progress_async(text_progress)
                
                # 이벤트 루프 양보 - 대용량 파일 처리 시 더 자주 양보
                import asyncio
                if page_num % 10 == 0:  # 10페이지마다 더 긴 대기
                    await asyncio.sleep(0.05)
                else:
                    await asyncio.sleep(0.001)  # 기본 대기 시간 단축
                
                # 1. 텍스트 추출
                page_text = page.get_text().strip()
                if page_text and len(page_text) > 20:  # 최소 길이 체크
                    # 텍스트를 청크로 나누기 (더 큰 청크 사용)
                    chunk_size = 2000
                    page_chunks_text = []
                    
                    for i in range(0, len(page_text), chunk_size):
                        chunk_text = page_text[i:i+chunk_size].strip()
                        if chunk_text and len(chunk_text) > 10:
                            page_chunks_text.append(chunk_text)
                    
                    # 임베딩 생성 (배치로 처리)
                    for chunk_text in page_chunks_text[:5]:  # 페이지당 최대 5개 청크
                        try:
                            # 임베딩 생성 전 이벤트 루프 양보 (블로킹 방지)
                            await asyncio.sleep(0.001)
                            embedding = await embedding_manager.embed_text(chunk_text)
                            
                            chunk = DocumentChunk(
                                id=str(uuid.uuid4()),
                                content=chunk_text,
                                embedding=embedding,
                                metadata={
                                    "document_id": document_id,
                                    "file_path": file_path,
                                    "original_filename": original_filename,
                                    "page": page_num + 1,
                                    "chunk_index": len(chunks),
                                    "content_type": "text",
                                    "file_type": "pdf",
                                    "created_at": str(datetime.now())
                                }
                            )
                            batch_chunks.append(chunk)
                            
                            # 배치 처리
                            if len(batch_chunks) >= max_chunks_per_batch:
                                chunks.extend(batch_chunks)
                                batch_chunks = []
                                print(f"📄 PDF 텍스트 처리: {len(chunks)}개 청크 완료", flush=True)
                                await asyncio.sleep(0.05)  # 배치 처리 후 대기 시간 단축
                            
                        except Exception as embed_error:
                            logger.warning(f"텍스트 임베딩 실패 (페이지 {page_num + 1}): {embed_error}")
                            continue
                
                processed_pages += 1
                
                # 50페이지마다 진행 상황 출력 (1000페이지 처리 시 너무 많은 로그 방지)
                if processed_pages % 50 == 0:
                    print(f"📖 PDF 처리 진행: {processed_pages}/{min(total_pages, 1000)} 페이지 완료 ({text_progress:.1f}%)", flush=True)
                    
            except Exception as page_error:
                logger.warning(f"PDF 페이지 {page_num + 1} 처리 실패: {page_error}")
                continue
        
        # 남은 배치 처리
        if batch_chunks:
            chunks.extend(batch_chunks)
        
        # 텍스트 처리 완료 전에 100% 진행률 업데이트
        print(f"🔄 텍스트 처리 완료 중... (100%)", flush=True)
        await progress.update_step_progress_async(100.0)
        await asyncio.sleep(0.1)  # 진행률 업데이트 전송 시간 확보
        
        print(f"✅ 텍스트 처리 단계 완료!", flush=True)
        await progress.complete_step_async()  # 텍스트 추출 완료
        
        # 이미지 처리 시작
        await progress.start_step_async(3)  # 이미지 추출
        print(f"📷 이미지 처리 시작...", flush=True)
        
        # 이미지 추출 및 OCR 처리
        for page_num in range(min(total_pages, 1000)):  # 최대 1000페이지의 이미지 처리
            try:
                page = pdf_document[page_num]
                image_list = page.get_images()
                
                if image_list:
                    print(f"📷 페이지 {page_num + 1}: {len(image_list)}개 이미지 발견", flush=True)
                
                for img_index, img in enumerate(image_list[:10]):  # 페이지당 최대 10개 이미지
                    try:
                        # 이미지 추출
                        xref = img[0]
                        pix = fitz.Pixmap(pdf_document, xref)
                        
                        if pix.n - pix.alpha < 4:  # GRAY 또는 RGB
                            # 이미지를 PIL Image로 변환
                            img_data = pix.tobytes("png")
                            
                            # 도면/기술 문서용 이미지 품질 개선
                            enhanced_img_data = await _enhance_technical_image(img_data, page_num + 1, img_index + 1)
                            
                            # 이미지 메타데이터만 저장 (간소화)
                            image_metadata = {
                                "document_id": document_id,
                                "original_filename": original_filename,
                                "page": page_num + 1,
                                "image_index": img_index,
                                "chunk_index": len(chunks),
                                "content_type": "image",
                                "file_type": "pdf",
                                "image_size": len(enhanced_img_data),
                                "original_size": len(img_data),
                                "enhanced": True,
                                "created_at": str(datetime.now())
                            }
                            
                            # 간단한 이미지 청크 생성 (임베딩 없이)
                            image_content = f"Enhanced technical image from page {page_num + 1}, image {img_index + 1}"
                            if enhanced_img_data != img_data:
                                image_content += " (upscaled and enhanced)"
                            
                            image_chunk = DocumentChunk(
                                id=str(uuid.uuid4()),
                                content=image_content,
                                embedding=[0.0] * 768,  # 더미 임베딩
                                metadata=image_metadata
                            )
                            chunks.append(image_chunk)
                            image_count += 1
                            
                            if enhanced_img_data != img_data:
                                print(f"🔍 이미지 품질 개선 완료: 페이지 {page_num + 1}, 이미지 {img_index + 1}", flush=True)
                            else:
                                print(f"🖼️ 이미지 추가: 페이지 {page_num + 1}, 이미지 {img_index + 1}", flush=True)
                        
                        # 메모리 정리 (대용량 파일 처리 시 중요)
                        if pix:
                            pix = None
                        
                        # 100개 이미지마다 가비지 컬렉션 (메모리 최적화)
                        if image_count % 100 == 0 and image_count > 0:
                            import gc
                            gc.collect()
                            print(f"🗑️ 메모리 정리 완료 ({image_count}개 이미지 처리됨)", flush=True)
                        
                    except Exception as img_error:
                        logger.warning(f"이미지 처리 실패 (페이지 {page_num + 1}, 이미지 {img_index + 1}): {img_error}")
                        continue
                        
                # 진행률 업데이트
                image_progress = (page_num / min(total_pages, 1000)) * 100
                await progress.update_step_progress_async(image_progress)
                        
            except Exception as page_error:
                logger.warning(f"페이지 {page_num + 1} 이미지 처리 실패: {page_error}")
                continue
                
        await progress.complete_step_async()  # 이미지 처리 완료
        
        await progress.start_step_async(4)  # OCR 처리
        print(f"🔍 OCR 처리 건너뛰기 (성능 최적화)", flush=True)
        await progress.complete_step_async()
        
        pdf_document.close()
        
        print(f"📋 PDF 처리 완료: {len(chunks)}개 청크 생성 (텍스트: {len(chunks) - image_count}, 이미지: {image_count})")
        logger.info(f"PDF 처리 완료: {len(chunks)}개 청크 (텍스트: {len(chunks) - image_count}, 이미지: {image_count}개 포함)")
        return chunks, image_count
        
        
    except Exception as e:
        logger.warning(f"PDF 처리 실패, 간단한 텍스트 추출로 대체: {e}")
        # Fallback: 기본 텍스트만 추출
        try:
            simple_chunks = await _process_pdf_file_simple(file_path, document_id, original_filename, embedding_manager)
            logger.info(f"PDF 간단 텍스트 처리 완료: {len(simple_chunks)}개 청크")
            return simple_chunks, 0
        except Exception as fallback_e:
            logger.error(f"PDF 텍스트 처리 완전 실패: {fallback_e}")
            return [], 0


async def _process_pdf_file_simple(file_path: str, document_id: str, original_filename: str, embedding_manager) -> list:
    """간단한 PDF 텍스트 추출"""
    import PyPDF2
    from app.models.schemas import DocumentChunk
    import uuid
    from datetime import datetime
    
    chunks = []
    
    try:
        with open(file_path, 'rb') as pdf_file:
            pdf_reader = PyPDF2.PdfReader(pdf_file)
            
            for page_num, page in enumerate(pdf_reader.pages[:20]):  # 최대 20페이지
                try:
                    page_text = page.extract_text().strip()
                    if page_text and len(page_text) > 50:
                        # 큰 청크 단위로 처리
                        chunk_size = 2000
                        for i in range(0, len(page_text), chunk_size):
                            chunk_text = page_text[i:i+chunk_size].strip()
                            if chunk_text and len(chunk_text) > 20:
                                
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
                                        "file_type": "pdf",
                                        "created_at": str(datetime.now())
                                    }
                                )
                                chunks.append(chunk)
                                
                except Exception as e:
                    logger.warning(f"PDF 페이지 {page_num + 1} 처리 실패: {e}")
                    continue
                    
        return chunks
        
    except Exception as e:
        logger.error(f"PDF 파일 읽기 실패: {e}")
        return []


async def _enhance_technical_image(img_data: bytes, page_num: int, img_index: int) -> bytes:
    """기술 도면/도표용 이미지 품질 향상"""
    try:
        from PIL import Image, ImageEnhance, ImageFilter
        import io
        
        # PIL Image로 변환
        image = Image.open(io.BytesIO(img_data))
        original_size = image.size
        
        # 도면/기술 문서 특성 감지
        is_technical_drawing = _detect_technical_drawing(image)
        
        # 이미지가 너무 작으면 업스케일링 (도면 특성상 해상도 중요)
        min_dimension = 1200 if is_technical_drawing else 800  # 도면이면 더 높은 해상도 요구
        max_dimension = max(image.size)
        
        if max_dimension < min_dimension:
            # 업스케일링 비율 계산
            scale_factor = min_dimension / max_dimension
            new_size = (int(image.width * scale_factor), int(image.height * scale_factor))
            
            print(f"🔍 이미지 업스케일링: {original_size} → {new_size} (페이지 {page_num}, 이미지 {img_index})", flush=True)
            
            # 고품질 업스케일링 (LANCZOS 사용)
            image = image.resize(new_size, Image.Resampling.LANCZOS)
        
        # 도면/기술 문서에 최적화된 후처리
        enhancement_factor = 1.4 if is_technical_drawing else 1.2
        
        # 1. 대비 향상 (도면의 선명도 개선)
        if image.mode in ['L', 'RGB']:  # 그레이스케일 또는 컬러 이미지
            enhancer = ImageEnhance.Contrast(image)
            image = enhancer.enhance(enhancement_factor)  # 도면이면 더 강한 대비
        
        # 2. 선명도 향상 (도면 라인 강화)
        if image.mode in ['L', 'RGB']:
            enhancer = ImageEnhance.Sharpness(image)
            sharpness_factor = 1.5 if is_technical_drawing else 1.3
            image = enhancer.enhance(sharpness_factor)  # 도면이면 더 강한 선명도
        
        # 3. 도면용 에지 강화
        if is_technical_drawing and max_dimension > 400:
            # 도면의 라인을 더욱 선명하게
            image = image.filter(ImageFilter.UnsharpMask(radius=1, percent=150, threshold=3))
        
        # 4. 노이즈 제거 (스캔된 도면의 잡음 제거)
        elif max_dimension > 300:  # 일반 이미지
            image = image.filter(ImageFilter.MedianFilter(size=3))
        
        # 결과를 바이트로 변환
        output_buffer = io.BytesIO()
        # PNG로 저장 (무손실, 도면에 적합)
        if image.mode in ['RGBA', 'LA']:
            image.save(output_buffer, format='PNG')
        else:
            image.save(output_buffer, format='PNG')
        
        enhanced_data = output_buffer.getvalue()
        
        # 개선 결과 로그
        improvement_ratio = len(enhanced_data) / len(img_data) if len(img_data) > 0 else 1
        drawing_type = "기술도면" if is_technical_drawing else "일반이미지"
        print(f"📈 {drawing_type} 품질 개선 완료: {len(img_data):,} → {len(enhanced_data):,} bytes (x{improvement_ratio:.1f})", flush=True)
        
        return enhanced_data
        
    except Exception as e:
        print(f"⚠️ 이미지 품질 개선 실패 (페이지 {page_num}, 이미지 {img_index}): {e}", flush=True)
        # 실패 시 원본 이미지 데이터 반환
        return img_data


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
            # 연결 상태 확인
            if document_id in progress_websocket.connections and len(progress_websocket.connections[document_id]) > 0:
                print(f"📡 완료 메시지 전송 시도 {retry_count + 1}/{max_retries}: {document_id}", flush=True)
                
                # 완료 메시지 전송
                await progress_websocket.send_completion(
                    document_id,
                    "completed", 
                    f"문서 처리가 완료되었습니다: {filename}",
                    result_data
                )
                
                print(f"✅ 완료 메시지 전송 성공!", flush=True)
                return True
            else:
                print(f"📡 WebSocket 연결 없음, 재연결 대기... ({retry_count + 1}/{max_retries})", flush=True)
                
            await asyncio.sleep(2)  # 2초 대기 후 재시도
            retry_count += 1
            
        except Exception as e:
            print(f"⚠️ 완료 메시지 전송 실패 ({retry_count + 1}/{max_retries}): {e}", flush=True)
            retry_count += 1
            await asyncio.sleep(1)
    
    print(f"❌ 완료 메시지 전송 최종 실패: {document_id}", flush=True)
    return False


def _detect_technical_drawing(image) -> bool:
    """이미지가 기술 도면인지 감지"""
    try:
        # 그레이스케일로 변환
        if image.mode != 'L':
            gray_image = image.convert('L')
        else:
            gray_image = image
        
        # 이미지 크기
        width, height = gray_image.size
        
        # 너무 작은 이미지는 도면이 아닐 가능성이 높음
        if width < 200 or height < 200:
            return False
        
        # 히스토그램 분석
        histogram = gray_image.histogram()
        
        # 흑백 픽셀의 비율 계산
        total_pixels = width * height
        black_pixels = sum(histogram[0:50])  # 어두운 픽셀
        white_pixels = sum(histogram[200:256])  # 밝은 픽셀
        
        # 도면 특징: 대부분 흰색 배경에 검은색 선
        white_ratio = white_pixels / total_pixels
        black_ratio = black_pixels / total_pixels
        
        # 도면 판별 조건
        # 1. 흰색 배경이 60% 이상
        # 2. 검은색 선이 10% 이상
        # 3. 중간 톤이 적음 (선명한 대비)
        middle_tones = sum(histogram[50:200]) / total_pixels
        
        is_drawing = (white_ratio > 0.6 and 
                     black_ratio > 0.05 and 
                     middle_tones < 0.3)
        
        if is_drawing:
            print(f"🏗️ 기술도면 감지됨: 백색 {white_ratio:.2f}, 흑색 {black_ratio:.2f}, 중간톤 {middle_tones:.2f}", flush=True)
        
        return is_drawing
        
    except Exception as e:
        print(f"⚠️ 도면 감지 실패: {e}", flush=True)
        return False



async def _delete_document_from_vector_db(collection_name: str, document_id: str) -> int:
    """벡터 DB에서 특정 document_id를 가진 모든 점들 삭제"""
    try:
        # 먼저 해당 document_id를 가진 모든 점들 찾기
        from app.core.embedding_manager import embedding_manager
        await embedding_manager.initialize()
        
        # 더미 검색으로 모든 점 가져오기
        test_embedding = await embedding_manager.embed_text("test")
        all_docs = await vector_store.search_similar(
            collection_name=collection_name,
            query_vector=test_embedding,
            limit=10000,  # 충분히 큰 수
            score_threshold=0.0
        )
        
        # document_id가 일치하는 점들의 ID 수집
        points_to_delete = []
        for doc in all_docs:
            if doc.metadata.get("document_id") == document_id:
                points_to_delete.append(doc.document_id)  # Qdrant point ID
        
        # 점들 삭제
        if points_to_delete:
            # Qdrant에서 점들 삭제
            await vector_store.client.delete(
                collection_name=collection_name,
                points_selector={"points": points_to_delete}
            )
            logger.info(f"벡터 DB에서 {len(points_to_delete)}개 점 삭제 완료: {document_id}")
        
        return len(points_to_delete)
        
    except Exception as e:
        logger.error(f"벡터 DB 삭제 실패: {e}")
        return 0


async def _delete_document_by_filename(collection_name: str, filename: str) -> int:
    """벡터 DB에서 특정 filename을 가진 모든 점들 삭제"""
    try:
        # Qdrant에서 직접 필터링으로 찾기 (더 효율적)
        from qdrant_client.models import Filter, FieldCondition, MatchValue
        
        # 다양한 필터 조건 시도
        filters_to_try = [
            # original_filename으로 정확히 매칭
            Filter(must=[FieldCondition(key="original_filename", match=MatchValue(value=filename))]),
            # filename으로 정확히 매칭  
            Filter(must=[FieldCondition(key="filename", match=MatchValue(value=filename))]),
        ]
        
        total_deleted = 0
        
        for i, filter_condition in enumerate(filters_to_try):
            try:
                logger.info(f"삭제 시도 {i+1}: 필터 조건으로 '{filename}' 검색")
                
                # 조건에 맞는 점들 검색
                search_result = await vector_store.client.scroll(
                    collection_name=collection_name,
                    scroll_filter=filter_condition,
                    limit=10000,
                    with_payload=True,
                    with_vectors=False
                )
                
                points_found = search_result[0] if search_result else []
                logger.info(f"필터 {i+1}로 찾은 점 수: {len(points_found)}")
                
                if points_found:
                    # 점들의 ID 수집
                    point_ids = [point.id for point in points_found]
                    
                    # 실제 삭제 실행
                    delete_result = await vector_store.client.delete(
                        collection_name=collection_name,
                        points_selector={"points": point_ids}
                    )
                    
                    deleted_count = len(point_ids)
                    total_deleted += deleted_count
                    
                    logger.info(f"필터 {i+1}로 {deleted_count}개 점 삭제 완료")
                    
                    # 첫 번째 성공한 필터로 삭제됐으면 나머지는 시도하지 않음
                    if deleted_count > 0:
                        logger.info(f"'{filename}' 삭제 성공: 총 {total_deleted}개 점 삭제됨")
                        return total_deleted
                        
            except Exception as filter_error:
                logger.warning(f"필터 {i+1} 삭제 시도 실패: {filter_error}")
                continue
        
        # 필터링으로 안 되면 전체 검색 후 매칭 (fallback)
        if total_deleted == 0:
            logger.info("필터링 실패, 전체 검색으로 fallback")
            
            from app.core.embedding_manager import embedding_manager
            await embedding_manager.initialize()
            
            # 더미 검색으로 모든 점 가져오기
            test_embedding = await embedding_manager.embed_text("test")
            all_docs = await vector_store.search_similar(
                collection_name=collection_name,
                query_vector=test_embedding,
                limit=10000,
                score_threshold=0.0
            )
            
            logger.info(f"전체 검색으로 {len(all_docs)}개 문서 확인")
            
            # filename이 일치하는 점들 찾기
            points_to_delete = []
            
            for doc in all_docs:
                doc_filename = doc.metadata.get("filename", "")
                doc_original_filename = doc.metadata.get("original_filename", "")
                
                # 단순하고 확실한 매칭
                if (doc_original_filename == filename or 
                    doc_filename == filename or
                    os.path.basename(doc_original_filename) == filename or
                    os.path.basename(doc_filename) == filename):
                    
                    points_to_delete.append(doc.document_id)
                    logger.info(f"매칭 발견: original='{doc_original_filename}', filename='{doc_filename}'")
            
            # 삭제 실행
            if points_to_delete:
                await vector_store.client.delete(
                    collection_name=collection_name,
                    points_selector={"points": points_to_delete}
                )
                total_deleted = len(points_to_delete)
                logger.info(f"Fallback으로 {total_deleted}개 점 삭제 완료")
            else:
                logger.warning(f"'{filename}' 파일을 찾을 수 없음")
        
        return total_deleted
        
    except Exception as e:
        logger.error(f"벡터 DB 파일명 기반 삭제 실패: {e}")
        return 0


async def _process_pdf_file(
    file_path: str, 
    document_id: str, 
    original_filename: str
) -> list:
    """PDF 파일 처리"""
    import PyPDF2
    from app.core.embedding_manager import embedding_manager
    from app.models.schemas import DocumentChunk
    import uuid
    from datetime import datetime
    
    chunks = []
    
    with open(file_path, 'rb') as pdf_file:
        pdf_reader = PyPDF2.PdfReader(pdf_file)
        
        for page_num, page in enumerate(pdf_reader.pages):
            try:
                page_text = page.extract_text().strip()
                if not page_text:
                    continue
                
                # 페이지별로 청킹 (큰 페이지는 더 나눌 수 있음)
                chunk_size = 1500
                for i in range(0, len(page_text), chunk_size):
                    chunk_text = page_text[i:i+chunk_size].strip()
                    if not chunk_text:
                        continue
                    
                    # 임베딩 생성
                    embedding = await embedding_manager.embed_text(chunk_text)
                    
                    chunk = DocumentChunk(
                        id=str(uuid.uuid4()),
                        content=chunk_text,
                        embedding=embedding,
                        metadata={
                            "document_id": document_id,
                            "file_path": file_path,
                            "original_filename": original_filename,
                            "page": page_num + 1,
                            "chunk_index": len(chunks),
                            "file_type": "pdf",
                            "created_at": str(datetime.now())
                        }
                    )
                    chunks.append(chunk)
                    
            except Exception as e:
                logger.warning(f"PDF 페이지 {page_num + 1} 처리 실패: {e}")
                continue
    
    return chunks


async def _process_simple_document(file_path: str, user_id: str) -> ProcessingResult:
    """임시 구현: 간단한 문서 처리"""
    from app.core.embedding_manager import embedding_manager
    from app.models.schemas import DocumentChunk
    import uuid
    from datetime import datetime
    
    try:
        # 임베딩 매니저 초기화
        await embedding_manager.initialize()
        
        # 파일 확장자에 따른 간단한 처리
        file_extension = Path(file_path).suffix.lower()
        
        if file_extension == '.txt':
            # 텍스트 파일 읽기
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # 간단한 청킹 (1000자 단위)
            chunks = []
            for i in range(0, len(content), 1000):
                chunk_text = content[i:i+1000]
                if chunk_text.strip():
                    # 임베딩 생성
                    embedding = await embedding_manager.embed_text(chunk_text)
                    
                    chunk = DocumentChunk(
                        id=str(uuid.uuid4()),
                        content=chunk_text,
                        embedding=embedding,
                        metadata={
                            "file_path": file_path,
                            "chunk_index": len(chunks),
                            "file_type": "text"
                        }
                    )
                    chunks.append(chunk)
            
            return ProcessingResult(
                text_chunks=chunks,
                image_chunks=[],
                total_embeddings=len(chunks)
            )
        
        else:
            # 다른 파일 형식은 아직 미구현
            raise HTTPException(
                status_code=400,
                detail=f"아직 지원하지 않는 파일 형식입니다: {file_extension}"
            )
            
    except Exception as e:
        logger.error(f"간단 문서 처리 실패: {e}")
        raise


@router.get("/status/{document_id}")
async def get_document_status(document_id: str):
    """문서 처리 상태 조회"""
    # TODO: 실제 문서 상태 추적 시스템 구현
    return {
        "document_id": document_id,
        "status": "completed",
        "message": "문서 처리가 완료되었습니다"
    }


@router.get("/list")
async def list_documents(user_id: str = "anonymous") -> Dict[str, Any]:
    """업로드된 문서 목록 조회 - 벡터 DB에서 실제 저장된 문서 확인"""
    try:
        # 벡터 DB 초기화
        await vector_store.initialize()
        
        # 사용자별 컬렉션명
        collection_name = f"documents_{user_id}"
        
        try:
            # 컬렉션 존재 확인
            await vector_store.ensure_collection(collection_name)
            
            # 벡터 DB에서 모든 문서 정보 가져오기
            from app.core.embedding_manager import embedding_manager
            await embedding_manager.initialize()
            
            # 더미 임베딩으로 모든 문서 검색 (score_threshold=0.0으로 모든 결과 반환)
            test_embedding = await embedding_manager.embed_text("test")
            all_docs = await vector_store.search_similar(
                collection_name=collection_name,
                query_vector=test_embedding,
                limit=1000,  # 충분히 큰 수
                score_threshold=0.0  # 모든 문서 반환
            )
            
            # 문서별로 그룹화
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
            
            # 응답 형식으로 변환
            uploaded_files = []
            for doc_info in doc_groups.values():
                uploaded_files.append({
                    "filename": f"{doc_info['document_id']}_{doc_info['original_filename']}",
                    "original_name": doc_info["original_filename"],
                    "document_id": doc_info["document_id"],
                    "chunks": doc_info["chunks"],
                    "content_length": doc_info["total_content_length"],
                    "uploaded_at": doc_info["created_at"][:19] if doc_info["created_at"] else "",
                    "file_type": doc_info["file_type"],
                    "stored_in_vector_db": True
                })
            
            return {
                "files": uploaded_files,
                "total_count": len(uploaded_files),
                "collection_name": collection_name,
                "total_chunks": sum(doc["chunks"] for doc in uploaded_files)
            }
            
        except Exception as vector_error:
            logger.error(f"벡터 DB에서 문서 조회 실패: {vector_error}")
            raise HTTPException(
                status_code=500,
                detail=f"문서 목록 조회 중 오류가 발생했습니다: {str(vector_error)}"
            )
        
    except Exception as e:
        logger.error(f"파일 목록 조회 실패: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"파일 목록 조회 중 오류가 발생했습니다: {str(e)}"
        )


@router.delete("/delete/{document_id}")
async def delete_document(document_id: str, user_id: str = "anonymous") -> DocumentDeleteResponse:
    """업로드된 문서 삭제 - 벡터 DB에서만 삭제 (파일 시스템 사용 안함)"""
    try:
        # 벡터 DB에서 문서 삭제
        deleted_from_vector_db = False
        deleted_count = 0
        
        try:
            # 벡터 DB 초기화
            await vector_store.initialize()
            collection_name = f"documents_{user_id}"
            
            # 해당 document_id를 가진 모든 점들 삭제
            deleted_count = await _delete_document_from_vector_db(collection_name, document_id)
            
            if deleted_count > 0:
                deleted_from_vector_db = True
                logger.info(f"벡터 DB에서 {deleted_count}개 청크 삭제: {document_id}")
            else:
                logger.warning(f"삭제할 문서를 찾을 수 없음: {document_id}")
                
        except Exception as vector_error:
            logger.error(f"벡터 DB 삭제 실패: {vector_error}")
            raise HTTPException(
                status_code=500,
                detail=f"문서 삭제 중 오류가 발생했습니다: {str(vector_error)}"
            )
        
        if deleted_from_vector_db:
            message = f"문서 '{document_id}'가 성공적으로 삭제되었습니다. ({deleted_count}개 청크 삭제됨)"
            success = True
        else:
            message = f"문서 '{document_id}'를 찾을 수 없어 삭제되지 않았습니다."
            success = False
            
        return DocumentDeleteResponse(
            message=message, 
            deleted_chunks=deleted_count,
            success=success
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"파일 삭제 실패: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"문서 삭제 중 오류가 발생했습니다: {str(e)}"
        )


@router.delete("/clear-all")
async def clear_all_documents(user_id: str = "anonymous") -> DocumentDeleteResponse:
    """사용자의 모든 문서를 벡터 DB에서 삭제"""
    try:
        deleted_count = 0
        
        try:
            # 벡터 DB 초기화
            await vector_store.initialize()
            collection_name = f"documents_{user_id}"
            
            # 컬렉션 전체 삭제 시도
            try:
                # 컬렉션이 존재하는지 확인
                import asyncio
                collections = await asyncio.to_thread(vector_store.client.get_collections)
                collection_exists = any(col.name == collection_name for col in collections.collections)
                
                if collection_exists:
                    # 컬렉션 삭제
                    await asyncio.to_thread(vector_store.client.delete_collection, collection_name)
                    logger.info(f"컬렉션 '{collection_name}' 전체 삭제 완료")
                    
                    # 새로운 빈 컬렉션 생성
                    await vector_store._create_collection(collection_name)
                    logger.info(f"새로운 빈 컬렉션 '{collection_name}' 생성 완료")
                    
                    deleted_count = "전체"  # 전체 삭제를 표시
                    message = f"사용자 '{user_id}'의 모든 문서가 성공적으로 삭제되었습니다."
                    success = True
                else:
                    message = f"사용자 '{user_id}'의 문서 컬렉션이 존재하지 않습니다."
                    success = False
                    
            except Exception as collection_error:
                logger.warning(f"컬렉션 삭제 실패, 개별 점 삭제로 전환: {collection_error}")
                
                # 컬렉션 삭제가 실패하면 모든 점 개별 삭제
                from app.core.embedding_manager import embedding_manager
                await embedding_manager.initialize()
                
                # 더미 검색으로 모든 점 가져오기
                test_embedding = await embedding_manager.embed_text("test")
                all_docs = await vector_store.search_similar(
                    collection_name=collection_name,
                    query_vector=test_embedding,
                    limit=50000,  # 매우 큰 수
                    score_threshold=0.0
                )
                
                if all_docs:
                    # 모든 점의 ID 수집
                    all_point_ids = [doc.document_id for doc in all_docs]
                    
                    # 배치로 삭제 (Qdrant 제한 고려)
                    batch_size = 1000
                    total_deleted = 0
                    
                    for i in range(0, len(all_point_ids), batch_size):
                        batch_ids = all_point_ids[i:i + batch_size]
                        from qdrant_client.models import PointIdsList
                        await asyncio.to_thread(
                            vector_store.client.delete,
                            collection_name=collection_name,
                            points_selector=PointIdsList(points=batch_ids)
                        )
                        total_deleted += len(batch_ids)
                        logger.info(f"배치 삭제 진행: {total_deleted}/{len(all_point_ids)}")
                    
                    deleted_count = total_deleted
                    message = f"사용자 '{user_id}'의 모든 문서가 성공적으로 삭제되었습니다. ({total_deleted}개 청크 삭제됨)"
                    success = True
                else:
                    message = f"사용자 '{user_id}'의 문서가 이미 비어있습니다."
                    success = False
                
        except Exception as vector_error:
            logger.error(f"벡터 DB 전체 삭제 실패: {vector_error}")
            raise HTTPException(
                status_code=500,
                detail=f"전체 문서 삭제 중 오류가 발생했습니다: {str(vector_error)}"
            )
        
        return DocumentDeleteResponse(
            message=message,
            deleted_chunks=deleted_count if isinstance(deleted_count, int) else 0,
            success=success
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"전체 파일 삭제 실패: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"전체 문서 삭제 중 오류가 발생했습니다: {str(e)}"
        )


@router.delete("/delete-by-name/{filename}")
async def delete_document_by_name(filename: str, user_id: str = "anonymous") -> DocumentDeleteResponse:
    """파일명으로 문서 삭제 - 벡터 DB에서만 삭제"""
    try:
        # 벡터 DB에서 해당 파일명을 가진 문서들 검색
        deleted_from_vector_db = False
        deleted_count = 0
        
        try:
            # 벡터 DB 초기화
            await vector_store.initialize()
            collection_name = f"documents_{user_id}"
            
            # filename을 기준으로 삭제
            deleted_count = await _delete_document_by_filename(collection_name, filename)
            
            if deleted_count > 0:
                deleted_from_vector_db = True
                logger.info(f"벡터 DB에서 {deleted_count}개 청크 삭제 (파일명: {filename})")
            else:
                logger.warning(f"삭제할 파일을 찾을 수 없음: {filename}")
                
        except Exception as vector_error:
            logger.error(f"벡터 DB 삭제 실패: {vector_error}")
            raise HTTPException(
                status_code=500,
                detail=f"문서 삭제 중 오류가 발생했습니다: {str(vector_error)}"
            )
        
        if deleted_from_vector_db:
            message = f"파일 '{filename}'이 성공적으로 삭제되었습니다. ({deleted_count}개 청크 삭제됨)"
            success = True
        else:
            message = f"파일 '{filename}'을 찾을 수 없어 삭제되지 않았습니다."
            success = False
            
        return DocumentDeleteResponse(
            message=message, 
            deleted_chunks=deleted_count,
            success=success
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"파일 삭제 실패: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"파일 삭제 중 오류가 발생했습니다: {str(e)}"
        )


@router.get("/vector-status")
async def get_vector_status() -> Dict[str, Any]:
    """벡터 DB(Qdrant) 상태 조회"""
    try:
        await vector_store.initialize()
        
        # Qdrant 컬렉션 정보 조회
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
        from app.core.embedding_manager import embedding_manager
        from app.core.rag_engine import rag_engine
        
        # RAG 엔진 초기화 및 검색 테스트
        await rag_engine.initialize()
        
        # 임베딩 생성
        query_embedding = await embedding_manager.embed_text(query)
        
        # 벡터 검색
        results = await vector_store.search_similar(
            collection_name="documents_test_user",
            query_vector=query_embedding,
            limit=3,
            score_threshold=0.0  # 모든 결과 반환
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


@router.websocket("/ws/progress/{document_id}")
async def websocket_progress(websocket: WebSocket, document_id: str):
    """문서 처리 진행률을 실시간으로 스트리밍"""
    try:
        from app.core.websocket_manager import progress_websocket
        
        await progress_websocket.connect(websocket, document_id)
        logger.info(f"WebSocket 연결됨: {document_id}")
        
        try:
            while True:
                # 클라이언트로부터 메시지 대기 (연결 유지)
                data = await websocket.receive_text()
                
                # 핑/폰 메시지 처리
                if data == "ping":
                    await websocket.send_text("pong")
                    
        except WebSocketDisconnect:
            logger.info(f"WebSocket 연결 해제됨: {document_id}")
        finally:
            await progress_websocket.disconnect(websocket, document_id)
            
    except Exception as e:
        logger.error(f"WebSocket 오류: {e}")
        try:
            await progress_websocket.disconnect(websocket, document_id)
        except:
            pass