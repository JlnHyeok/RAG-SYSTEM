from typing import List, Optional, Dict, Any, Union
from pydantic import BaseModel, Field
from datetime import datetime
from .enums import DocumentType, ProcessingStatus, OCREngine, EmbeddingModelType


class DocumentChunk(BaseModel):
    """문서 청크 데이터 모델"""
    id: str
    content: str
    embedding: List[float]
    metadata: Dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=datetime.now)


class ProcessingResult(BaseModel):
    """문서 처리 결과"""
    text_chunks: List[DocumentChunk] = Field(default_factory=list)
    image_chunks: List[DocumentChunk] = Field(default_factory=list)
    total_embeddings: int = 0
    processing_time: float = 0.0
    status: ProcessingStatus = ProcessingStatus.COMPLETED
    error_message: Optional[str] = None


class QueryRequest(BaseModel):
    """RAG 쿼리 요청"""
    question: str = Field(..., min_length=1, max_length=1000)
    user_id: str
    conversation_id: Optional[str] = None
    max_results: int = Field(default=5, ge=1, le=20)
    score_threshold: float = Field(default=0.3, ge=0.0, le=1.0)


class SearchResult(BaseModel):
    """검색 결과 아이템"""
    document_id: str
    chunk_id: Optional[str] = None  # 청크 고유 ID (중복 제거용)
    content: str
    score: float
    metadata: Dict[str, Any] = Field(default_factory=dict)


class QueryResponse(BaseModel):
    """RAG 쿼리 응답"""
    answer: str
    sources: List[SearchResult]
    confidence: float
    processing_time: float = 0.0
    timestamp: datetime = Field(default_factory=datetime.now)


class DocumentUploadRequest(BaseModel):
    """문서 업로드 요청"""
    file_path: str
    user_id: str
    document_type: Optional[DocumentType] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class DocumentUploadResponse(BaseModel):
    """문서 업로드 응답"""
    document_id: str
    filename: Optional[str] = None
    status: Union[ProcessingStatus, str] = ProcessingStatus.PENDING
    message: Optional[str] = None
    text_chunks: int = 0
    image_chunks: int = 0
    total_embeddings: int = 0
    processing_time: float = 0.0


class OCRResult(BaseModel):
    """OCR 처리 결과"""
    text: str
    confidence: float
    engine_used: OCREngine
    bounding_boxes: List[Dict[str, Any]] = Field(default_factory=list)


class HealthResponse(BaseModel):
    """헬스 체크 응답"""
    status: str
    service: str
    version: str
    timestamp: datetime = Field(default_factory=datetime.now)
    models_loaded: Dict[str, Any] = Field(default_factory=dict)  # boolean에서 Any로 변경
    error: Optional[str] = None


class DocumentDeleteResponse(BaseModel):
    """문서 삭제 응답"""
    message: str
    deleted_chunks: int = 0
    success: bool = True