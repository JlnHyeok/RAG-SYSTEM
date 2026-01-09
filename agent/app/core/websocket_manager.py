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
        logger.info(f"WebSocket 연결됨: {document_id}")
        
    async def disconnect(self, websocket: WebSocket, document_id: str):
        """WebSocket 연결 해제"""
        if document_id in self.connections:
            self.connections[document_id].discard(websocket)
            
            if not self.connections[document_id]:
                del self.connections[document_id]
                
        logger.info(f"WebSocket 연결 해제됨: {document_id}")
    
    async def send_progress(self, document_id: str, step: str, progress: float, message: str = ""):
        """진행률을 모든 연결된 클라이언트에게 전송"""
        if document_id not in self.connections:
            return
            
        progress_data = {
            "type": "progress",
            "document_id": document_id,
            "step": step,
            "progress": progress,
            "message": message,
            "timestamp": asyncio.get_event_loop().time()
        }
        
        disconnected = set()
        
        for websocket in self.connections[document_id]:
            try:
                await websocket.send_text(json.dumps(progress_data))
            except Exception as e:
                logger.warning(f"WebSocket 전송 실패: {e}")
                disconnected.add(websocket)
        
        # 끊어진 연결 제거
        for ws in disconnected:
            await self.disconnect(ws, document_id)
    
    async def send_completion(self, document_id: str, status: str, result: dict):
        """완료 상태를 전송"""
        if document_id not in self.connections:
            return
            
        completion_data = {
            "type": "completion",
            "document_id": document_id,
            "status": status,
            "result": result,
            "timestamp": asyncio.get_event_loop().time()
        }
        
        disconnected = set()
        
        for websocket in self.connections[document_id]:
            try:
                await websocket.send_text(json.dumps(completion_data))
            except Exception as e:
                logger.warning(f"WebSocket 전송 실패: {e}")
                disconnected.add(websocket)
        
        # 끊어진 연결 제거
        for ws in disconnected:
            await self.disconnect(ws, document_id)

# 전역 WebSocket 매니저
progress_websocket = ProgressWebSocket()