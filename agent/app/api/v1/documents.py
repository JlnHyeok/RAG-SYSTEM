from fastapi import APIRouter, HTTPException, UploadFile, File, Form
from typing import Dict, Any, Optional, Tuple
import logging
import time
import hashlib
import shutil
import os
from pathlib import Path

# TOKENIZERS_PARALLELISM 경고 해결
os.environ["TOKENIZERS_PARALLELISM"] = "false"

class ProcessingProgress:
    """문서 처리 진행 상황을 추적하는 클래스"""
    def __init__(self, document_id: str, filename: str):
        self.document_id = document_id
        self.filename = filename
        self.current_step = ""
        self.step_progress = 0.0
        self.total_steps = 6
        self.current_step_index = 0
        
        self.steps = [
            "📤 파일 업로드",
            "📖 PDF 파싱", 
            "✂️ 텍스트 추출 및 청킹",
            "🖼️ 이미지 추출",
            "👁️ OCR 처리", 
            "🧠 임베딩 생성 및 벡터 저장"
        ]
        
    def start_step(self, step_index: int):
        """단계 시작"""
        self.current_step_index = step_index
        self.current_step = self.steps[step_index]
        self.step_progress = 0.0
        self._log_progress()
        self._send_websocket_progress()
        
    def update_step_progress(self, progress: float):
        """현재 단계 진행률 업데이트"""
        self.step_progress = min(100.0, max(0.0, progress))
        self._log_progress()
        self._send_websocket_progress()
        
    def complete_step(self):
        """현재 단계 완료"""
        self.step_progress = 100.0
        self._log_progress()
        self._send_websocket_progress()
    
    def _send_websocket_progress(self):
        """WebSocket으로 진행률 전송 (비동기)"""
        try:
            import asyncio
            from app.core.websocket_manager import progress_websocket
            
            # 현재 이벤트 루프가 있는지 확인
            try:
                loop = asyncio.get_running_loop()
                # 백그라운드 태스크로 실행
                loop.create_task(progress_websocket.send_progress(
                    self.document_id,
                    self.current_step,
                    self.step_progress,
                    f"{self.current_step_index + 1}/{self.total_steps} - {self.step_progress:.1f}%"
                ))
            except RuntimeError:
                # 이벤트 루프가 없으면 무시
                pass
        except Exception as e:
            # WebSocket 에러는 무시 (로그 출력은 계속)
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


@router.post("/upload", response_model=DocumentUploadResponse)
async def upload_document(
    file: UploadFile = File(...),
    user_id: str = Form("anonymous"),
    document_type: Optional[str] = Form(None)
) -> DocumentUploadResponse:
    """PDF, Word, 텍스트 파일 업로드 및 처리"""
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
        progress.start_step(0)  # 파일 업로드
        
        # 파일 내용을 메모리에서 직접 읽기
        file_content = await file.read()
        progress.complete_step()
        
        print(f"\n✅ 파일 업로드 완료: {file.filename} (크기: {len(file_content):,} bytes)", flush=True)
        logger.info(f"파일 업로드 완료: {file.filename} (크기: {len(file_content)} bytes, 사용자: {user_id})")
        
        # 메모리에서 직접 문서 처리 및 벡터 DB 저장
        print(f"\n🚀 문서 처리 시작: {file.filename}", flush=True)
        logger.info(f"문서 처리 시작: {file.filename}")
        try:
            processing_result = await _process_and_store_document_from_memory(
                file_content=file_content,
                file_extension=file_extension,
                user_id=user_id,
                document_id=file_hash,
                original_filename=file.filename,
                progress=progress
            )
            
            print(f"\n✅ 문서 처리 및 벡터 저장 완료: {file.filename}", flush=True)
            logger.info(f"문서 처리 및 벡터 저장 완료: {file.filename}")
            
            return DocumentUploadResponse(
                document_id=file_hash,
                status="completed",
                text_chunks=processing_result.get("text_chunks", 0),
                image_chunks=processing_result.get("image_chunks", 0),
                total_embeddings=processing_result.get("total_embeddings", 0),
                processing_time=time.time() - start_time
            )
            
        except Exception as processing_error:
            logger.error(f"문서 처리 실패: {processing_error}")
            
            # 처리 실패 시에도 파일은 업로드된 상태로 유지 (재처리 가능)
            error_detail = str(processing_error)
            if "embedding" in error_detail.lower():
                error_msg = "임베딩 생성 실패"
            elif "vector" in error_detail.lower() or "qdrant" in error_detail.lower():
                error_msg = "벡터 DB 저장 실패"
            elif "datetime" in error_detail.lower():
                error_msg = "날짜 처리 오류"
            else:
                error_msg = "문서 처리 중 오류"
            
            return DocumentUploadResponse(
                document_id=file_hash,
                status="failed",
                text_chunks=0,
                image_chunks=0,
                total_embeddings=0,
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
            try:
                content = file_content.decode('utf-8')
            except UnicodeDecodeError:
                # UTF-8 디코딩 실패 시 다른 인코딩 시도
                try:
                    content = file_content.decode('cp949')  # 한국어 인코딩
                except UnicodeDecodeError:
                    content = file_content.decode('latin-1', errors='ignore')
            
            chunks = await _process_text_content_from_string(
                content, document_id, original_filename, rag_engine.embedding_manager
            )
            text_chunks = len(chunks)
            
        elif file_extension == '.pdf':
            # PDF 파일을 임시로 저장해서 처리 (PyMuPDF 등이 파일 경로 필요)
            import tempfile
            import os
            
            progress.start_step(1)  # PDF 파싱
            
            with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as temp_file:
                temp_file.write(file_content)
                temp_path = temp_file.name
            
            progress.complete_step()
            
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
                    
        else:
            # 기타 파일은 텍스트로 처리 시도
            try:
                content = file_content.decode('utf-8', errors='ignore')
                chunks = await _process_text_content_from_string(
                    content, document_id, original_filename, rag_engine.embedding_manager
                )
                text_chunks = len(chunks)
            except Exception as e:
                logger.warning(f"지원하지 않는 파일 형식 처리 실패: {file_extension}, {e}")
                chunks = []
        
        # 벡터 DB에 저장
        if chunks:
            progress.start_step(5)  # 벡터 저장
            progress.update_step_progress(50.0)
            
            await rag_engine.vector_store.add_documents(chunks, user_id)
            
            progress.complete_step()
            print(f"\n💾 Qdrant에 {len(chunks):,}개 청크 저장 완료: {original_filename}", flush=True)
            logger.info(f"Qdrant에 {len(chunks)}개 청크 저장 완료: {original_filename}")
        
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
        
        # 텍스트 추출 및 청킹 단계
        for page_num in range(total_pages):
            page = pdf_document[page_num]
            
            # 페이지별 진행률 계산
            text_progress = (page_num / total_pages) * 100
            progress.update_step_progress(text_progress)
            
            # 1. 텍스트 추출
            page_text = page.get_text().strip()
            if page_text:
                # 텍스트를 청크로 나누기
                chunk_size = 1500
                for i in range(0, len(page_text), chunk_size):
                    chunk_text = page_text[i:i+chunk_size].strip()
                    if not chunk_text:
                        continue
                    
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
                    chunks.append(chunk)
        
        progress.complete_step()  # 텍스트 추출 완료
        
        # 이미지 추출 단계
        progress.start_step(3)  # 이미지 추출
        
        total_images = 0
        # 전체 이미지 수 계산
        for page_num in range(total_pages):
            page = pdf_document[page_num]
            total_images += len(page.get_images())
        
        processed_images = 0
        
        # OCR 처리 단계
        if total_images > 0:
            progress.complete_step()  # 이미지 추출 완료
            progress.start_step(4)  # OCR 처리 시작
            
            for page_num in range(total_pages):
                page = pdf_document[page_num]
                image_list = page.get_images()
                
                for img_index, img in enumerate(image_list):
                    try:
                        # OCR 진행률 계산
                        ocr_progress = (processed_images / total_images) * 100
                        progress.update_step_progress(ocr_progress)
                        
                        # 이미지 추출
                        xref = img[0]
                        pix = fitz.Pixmap(pdf_document, xref)
                        
                        if pix.n - pix.alpha < 4:  # GRAY 또는 RGB
                            # PIL Image로 변환
                            img_data = pix.tobytes("png")
                            
                            # OCR 수행
                            ocr_text = await _perform_ocr_on_image(img_data)
                            
                            if ocr_text and ocr_text.strip():  # 빈 문자열이 아닌 모든 결과 저장
                                # OCR 결과를 청크로 저장
                                try:
                                    embedding = await embedding_manager.embed_text(ocr_text)
                                    
                                    chunk = DocumentChunk(
                                        id=str(uuid.uuid4()),
                                        content=ocr_text,
                                        embedding=embedding,
                                        metadata={
                                            "document_id": document_id,
                                            "original_filename": original_filename,
                                            "page": page_num + 1,
                                            "image_index": img_index,
                                            "chunk_index": len(chunks),
                                            "content_type": "image",
                                            "ocr_engine": "tesseract",
                                            "file_type": "pdf",
                                            "created_at": str(datetime.now())
                                        }
                                    )
                                    chunks.append(chunk)
                                    image_count += 1
                                    
                                    logger.info(f"PDF 이미지 OCR 성공: 페이지 {page_num + 1}, 이미지 {img_index + 1} - {len(ocr_text)} 문자")
                                except Exception as embed_error:
                                    logger.warning(f"OCR 텍스트 임베딩 실패: {embed_error}")
                            else:
                                logger.debug(f"이미지에서 텍스트 미발견: 페이지 {page_num + 1}, 이미지 {img_index + 1}")
                        
                        pix = None
                        processed_images += 1
                        
                    except Exception as e:
                        logger.warning(f"PDF 이미지 처리 실패 (페이지 {page_num + 1}, 이미지 {img_index + 1}): {e}")
                        processed_images += 1
                        continue
            
            progress.complete_step()  # OCR 처리 완료
        else:
            progress.complete_step()  # 이미지 추출 완료 (이미지 없음)
            progress.start_step(4)  # OCR 단계 건너뛰기
            progress.complete_step()
        
        pdf_document.close()
        
        logger.info(f"PDF 처리 완료: {len(chunks)}개 청크 (이미지 {image_count}개 포함)")
        return chunks, image_count
        
    except Exception as e:
        logger.warning(f"PDF 이미지 처리 실패, 텍스트만 처리: {e}")
        # Fallback: 기본 텍스트만 추출
        try:
            chunks = await _process_pdf_file(file_path, document_id, original_filename)
            logger.info(f"PDF 텍스트 처리 완료: {len(chunks)}개 청크")
            return chunks, 0
        except Exception as fallback_e:
            logger.error(f"PDF 텍스트 처리도 실패: {fallback_e}")
            return [], 0


async def _perform_ocr_on_image(image_data: bytes) -> str:
    """이미지 데이터에 OCR을 수행 - 모든 이미지 처리"""
    try:
        # pytesseract 직접 사용
        import pytesseract
        from PIL import Image
        import io
        
        # 이미지 로드
        image = Image.open(io.BytesIO(image_data))
        
        # 이미지 전처리 (OCR 성능 향상)
        # 이미지가 너무 작으면 확대
        if image.width < 100 or image.height < 100:
            # 2배 확대
            new_size = (image.width * 2, image.height * 2)
            image = image.resize(new_size, Image.Resampling.LANCZOS)
        
        # 이미지 모드 최적화
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # OCR 수행 (다양한 설정으로 시도)
        ocr_configs = [
            '--psm 3',  # 기본 자동 페이지 분할
            '--psm 6',  # 단일 블록
            '--psm 8',  # 단일 단어
            '--psm 7',  # 단일 텍스트 라인
            '--psm 13', # 원시 라인, Tesseract-specific 처리 없음
            ''          # 기본 설정
        ]
        
        # 언어별 시도
        languages = ['kor+eng', 'eng', 'kor']
        
        best_text = ""
        max_length = 0
        
        for lang in languages:
            for config in ocr_configs:
                try:
                    if lang and config:
                        text = pytesseract.image_to_string(image, lang=lang, config=config)
                    elif lang:
                        text = pytesseract.image_to_string(image, lang=lang)
                    elif config:
                        text = pytesseract.image_to_string(image, config=config)
                    else:
                        text = pytesseract.image_to_string(image)
                    
                    # 가장 긴 결과를 선택
                    if text and len(text.strip()) > max_length:
                        max_length = len(text.strip())
                        best_text = text.strip()
                        
                    # 충분히 긴 텍스트를 찾으면 조기 종료
                    if len(text.strip()) > 50:
                        return text.strip()
                        
                except Exception as e:
                    continue
        
        return best_text if best_text else ""
        
    except ImportError:
        logger.warning("pytesseract 미설치. 이미지 OCR을 건너뜁니다.")
        return ""
    except Exception as e:
        logger.debug(f"이미지 OCR 처리 중 오류: {e}")
        # 오류가 발생해도 빈 문자열 반환 (처리 계속)
        return ""


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