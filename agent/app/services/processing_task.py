"""
문서 처리 작업 정의 및 진행 상황 관리 모듈
"""
import time
import asyncio
import logging
import uuid
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List

logger = logging.getLogger(__name__)


@dataclass
class ProcessingTask:
    """처리할 작업 정의"""
    task_id: str = ""
    document_id: str = ""
    file_content: bytes = b""
    file_extension: str = ""
    user_id: str = ""
    original_filename: str = ""
    created_at: float = 0.0
    
    def __post_init__(self):
        if not self.task_id:
            self.task_id = str(uuid.uuid4())
        if not self.created_at:
            self.created_at = time.time()


class ProcessingProgress:
    """문서 처리 진행 상황을 추적하는 클래스"""
    
    # 전역 처리 상태 저장소
    _progress_store: Dict[str, 'ProcessingProgress'] = {}
    
    # 처리 단계 정의
    STEPS = [
        "📤 파일 업로드",
        "📖 PDF 파싱", 
        "✂️ 텍스트 추출 및 청킹",
        "🖼️ 이미지 추출",
        "👁️ OCR 처리", 
        "🧠 임베딩 생성 및 벡터 저장"
    ]
    
    def __init__(self, document_id: str, filename: str):
        self.document_id = document_id
        self.filename = filename
        self.current_step = ""
        self.step_progress = 0.0
        self.total_steps = len(self.STEPS)
        self.current_step_index = 0
        self.status = "processing"  # processing, completed, failed
        self.start_time = time.time()
        self.result_data: Dict[str, Any] = {}
        self.steps = self.STEPS
        
        # 전역 저장소에 저장
        ProcessingProgress._progress_store[document_id] = self
        
    @classmethod
    def get_progress(cls, document_id: str) -> Optional[Dict[str, Any]]:
        """문서 처리 상태 조회"""
        progress = cls._progress_store.get(document_id)
        if not progress:
            return None
            
        overall_progress = (
            progress.current_step_index + progress.step_progress / 100.0
        ) / progress.total_steps * 100
        
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
    
    @classmethod
    def set_completed(cls, document_id: str, result_data: Dict[str, Any]):
        """처리 완료 상태로 변경"""
        progress = cls._progress_store.get(document_id)
        if progress:
            progress.status = "completed"
            progress.result_data = result_data
            progress.current_step_index = progress.total_steps - 1
            progress.step_progress = 100.0
            
            # WebSocket 완료 알림 전송
            try:
                from app.core.websocket_manager import progress_websocket
                
                try:
                    loop = asyncio.get_running_loop()
                    print(f"📡 완료 메시지 전송 준비: {document_id} -> {result_data}", flush=True)
                    loop.create_task(progress_websocket.send_completion(
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
    
    @classmethod
    def set_failed(cls, document_id: str, error_message: str):
        """처리 실패 상태로 변경"""
        progress = cls._progress_store.get(document_id)
        if progress:
            progress.status = "failed"
            progress.result_data = {"error": error_message}
            
            # WebSocket 실패 알림 전송
            try:
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
                    print(f"⚠️ 이벤트 루프가 없어 WebSocket 알림 건너뛰기: {document_id}", flush=True)
            except Exception as e:
                print(f"⚠️ WebSocket 실패 알림 실패: {e}", flush=True)
    
    @classmethod
    def remove(cls, document_id: str):
        """진행 상황 제거"""
        if document_id in cls._progress_store:
            del cls._progress_store[document_id]
    
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
        print(f"🏁 단계 완료 시작: {self.current_step}", flush=True)
        self.step_progress = 100.0
        self._log_progress()
        
        await self._send_websocket_progress_async()
        
        # 완료 후 약간의 대기 (메시지 전송 보장)
        await asyncio.sleep(0.1)
        print(f"✅ 단계 완료됨: {self.current_step}", flush=True)
    
    def _send_websocket_progress(self):
        """WebSocket으로 진행률 전송 (동기)"""
        try:
            try:
                loop = asyncio.get_running_loop()
                loop.create_task(self._send_websocket_progress_async())
            except RuntimeError:
                pass
        except Exception:
            pass

    async def _send_websocket_progress_async(self):
        """WebSocket으로 진행률 전송 (비동기)"""
        try:
            from app.core.websocket_manager import progress_websocket
            
            overall_progress = (
                self.current_step_index + self.step_progress / 100.0
            ) / self.total_steps * 100
            
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

    def _log_progress(self):
        """진행 상황을 로그로 출력"""
        overall_progress = (
            self.current_step_index + self.step_progress / 100.0
        ) / self.total_steps * 100
        
        progress_bar = self._create_progress_bar(self.step_progress)
        
        progress_msg = f"📋 처리 중: {self.filename}"
        step_msg = f"🔄 {self.current_step}"
        step_progress_msg = f"📊 단계 진행률: {progress_bar} {self.step_progress:.1f}%"
        overall_progress_msg = f"📈 전체 진행률: {overall_progress:.1f}% ({self.current_step_index + 1}/{self.total_steps})"
        separator = "=" * 70
        
        print(f"\n{progress_msg}", flush=True)
        print(step_msg, flush=True)
        print(step_progress_msg, flush=True)
        print(overall_progress_msg, flush=True)
        print(separator, flush=True)
        
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
