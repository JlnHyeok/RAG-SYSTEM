from fastapi import WebSocket, WebSocketDisconnect
from typing import Dict, Set
import json
import asyncio
import logging

logger = logging.getLogger(__name__)

class ProgressWebSocket:
    """실시간 진행률을 WebSocket으로 전송하는 클래스"""
    
    def __init__(self):
        self.connections: Dict[str, Set[WebSocket]] = {}
        
    async def connect(self, websocket: WebSocket, document_id: str):
        """WebSocket 연결"""
        await websocket.accept()
        
        if document_id not in self.connections:
            self.connections[document_id] = set()
        
        self.connections[document_id].add(websocket)
        print(f"[WebSocket] Client connected for document {document_id}. Total connections: {len(self.connections[document_id])}")
        logger.info(f"WebSocket 연결됨: {document_id}")
        
    async def disconnect(self, websocket: WebSocket, document_id: str):
        """WebSocket 연결 해제"""
        if document_id in self.connections:
            self.connections[document_id].discard(websocket)
            remaining_connections = len(self.connections[document_id])
            
            if not self.connections[document_id]:
                del self.connections[document_id]
                
        print(f"[WebSocket] Client disconnected from document {document_id}. Remaining connections: {remaining_connections if document_id in self.connections else 0}")
        logger.info(f"WebSocket 연결 해제됨: {document_id}")
    
    async def send_progress(self, document_id: str, step: str, step_progress: float, overall_progress: float = 0, message: str = ""):
        """진행률을 모든 연결된 클라이언트에게 전송"""
        print(f"[WebSocket] Attempting to send progress - ID: {document_id}, Connections: {len(self.connections.get(document_id, []))}")
        
        if document_id not in self.connections:
            print(f"[WebSocket] No connections found for document_id: {document_id}")
            return
            
        progress_data = {
            "type": "progress",
            "document_id": document_id,
            "step": step,
            "current_step": step,  # 호환성을 위해 추가
            "progress": step_progress,
            "step_progress": step_progress,  # 호환성을 위해 추가
            "overall_progress": overall_progress,
            "message": message,
            "timestamp": asyncio.get_event_loop().time()
        }
        
        disconnected = set()
        sent_count = 0
        
        for websocket in self.connections[document_id]:
            try:
                await websocket.send_text(json.dumps(progress_data))
                sent_count += 1
                print(f"[WebSocket] Progress message sent successfully to client {sent_count}")
            except Exception as e:
                logger.warning(f"WebSocket 전송 실패: {e}")
                print(f"[WebSocket] Failed to send to client: {e}")
                disconnected.add(websocket)
        
        print(f"[WebSocket] Sent progress to {sent_count}/{len(self.connections[document_id])} clients")
        
        # 끊어진 연결 제거
        for ws in disconnected:
            await self.disconnect(ws, document_id)
    
    async def send_completion(self, document_id: str, status: str, message: str, result: dict):
        """완료 상태를 전송"""
        print(f"[WebSocket] 완료 메시지 전송 시작: {document_id}, status: {status}", flush=True)
        print(f"[WebSocket] 전송할 결과 데이터: {result}", flush=True)
        
        if document_id not in self.connections:
            print(f"[WebSocket] 연결이 없음: {document_id}", flush=True)
            return
            
        completion_data = {
            "type": "completion",
            "document_id": document_id,
            "status": status,
            "message": message,
            "result": result,
            "timestamp": asyncio.get_event_loop().time()
        }
        
        print(f"[WebSocket] 전송할 완료 데이터: {completion_data}", flush=True)
        
        disconnected = set()
        sent_count = 0
        
        for websocket in self.connections[document_id]:
            try:
                await websocket.send_text(json.dumps(completion_data))
                sent_count += 1
            except Exception as e:
                logger.warning(f"WebSocket 전송 실패: {e}")
                disconnected.add(websocket)
        
        print(f"[WebSocket] 완료 메시지 전송됨: {sent_count}개 클라이언트", flush=True)
        # 끊어진 연결 제거
        for ws in disconnected:
            await self.disconnect(ws, document_id)

    async def broadcast_status(self, message: str, status_type: str = "status"):
        """모든 연결된 클라이언트에게 상태 메시지 브로드캐스트"""
        data = {
            "type": status_type,
            "message": message,
            "timestamp": asyncio.get_event_loop().time()
        }
        
        for doc_id in list(self.connections.keys()):
            disconnected = set()
            for websocket in self.connections[doc_id]:
                try:
                    await websocket.send_text(json.dumps(data, ensure_ascii=False))
                except:
                    disconnected.add(websocket)
            for ws in disconnected:
                await self.disconnect(ws, doc_id)

# 전역 WebSocket 매니저
progress_websocket = ProgressWebSocket()