from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field
from datetime import datetime
from enum import Enum


class FileType(str, Enum):
    """파일 타입 열거형"""
    PDF = "pdf"
    TEXT = "txt"
    MARKDOWN = "md"
    IMAGE_PNG = "png"
    IMAGE_JPEG = "jpeg"
    IMAGE_JPG = "jpg"
    IMAGE_TIFF = "tiff"
    IMAGE_BMP = "bmp"
    DOCX = "docx"


class DocumentType(str, Enum):
    """문서 타입 열거형"""
    PDF = "pdf"
    IMAGE = "image"
    TEXT = "text"
    DOCX = "docx"


class ProcessingStatus(str, Enum):
    """처리 상태 열거형"""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


class OCREngine(str, Enum):
    """OCR 엔진 타입"""
    TESSERACT = "tesseract"
    PADDLE_OCR = "paddleocr"
    EASY_OCR = "easyocr"
    AWS_TEXTRACT = "aws_textract"


class EmbeddingModelType(str, Enum):
    """임베딩 모델 타입"""
    TEXT = "text"
    KOREAN = "korean"
    MULTIMODAL = "multimodal"