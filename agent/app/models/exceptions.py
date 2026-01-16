"""
RAG 시스템 커스텀 예외 클래스

이 모듈은 RAG 시스템 전반에서 사용되는 표준화된 예외 클래스들을 정의합니다.
각 예외는 명확한 에러 메시지와 추가 컨텍스트를 제공합니다.
"""
from typing import Optional, Dict, Any


class RAGBaseException(Exception):
    """
    RAG 시스템 기본 예외 클래스
    
    모든 커스텀 예외의 베이스 클래스로, 일관된 에러 메시지 형식과
    추가 상세 정보를 제공합니다.
    
    Attributes:
        message: 사용자에게 표시할 에러 메시지
        details: 디버깅을 위한 추가 정보
        error_code: 에러 분류를 위한 코드 (선택적)
    """
    
    def __init__(
        self, 
        message: str, 
        details: Optional[Dict[str, Any]] = None,
        error_code: Optional[str] = None
    ) -> None:
        self.message = message
        self.details = details or {}
        self.error_code = error_code or "RAG_ERROR"
        super().__init__(self.message)
    
    def to_dict(self) -> Dict[str, Any]:
        """예외 정보를 딕셔너리로 변환"""
        return {
            "error_code": self.error_code,
            "message": self.message,
            "details": self.details
        }
    
    def __str__(self) -> str:
        if self.details:
            return f"[{self.error_code}] {self.message} | Details: {self.details}"
        return f"[{self.error_code}] {self.message}"


# ============================================================================
# Gemini API 관련 예외
# ============================================================================

class GeminiAPIError(RAGBaseException):
    """Gemini API 호출 중 발생하는 일반적인 오류"""
    
    def __init__(
        self, 
        message: str = "Gemini API 호출 중 오류가 발생했습니다",
        details: Optional[Dict[str, Any]] = None
    ) -> None:
        super().__init__(message, details, "GEMINI_API_ERROR")


class QuotaExceededError(GeminiAPIError):
    """Gemini API 할당량 초과 오류"""
    
    def __init__(
        self, 
        message: str = "Gemini API 할당량을 초과했습니다. 잠시 후 다시 시도해주세요.",
        details: Optional[Dict[str, Any]] = None
    ) -> None:
        super().__init__(message, details)
        self.error_code = "QUOTA_EXCEEDED"


class ModelNotInitializedError(GeminiAPIError):
    """Gemini 모델이 초기화되지 않은 상태에서 호출"""
    
    def __init__(
        self, 
        message: str = "Gemini 모델이 초기화되지 않았습니다. initialize()를 먼저 호출하세요.",
        details: Optional[Dict[str, Any]] = None
    ) -> None:
        super().__init__(message, details)
        self.error_code = "MODEL_NOT_INITIALIZED"


# ============================================================================
# 벡터 저장소 관련 예외
# ============================================================================

class VectorStoreError(RAGBaseException):
    """벡터 저장소(Qdrant) 관련 일반 오류"""
    
    def __init__(
        self, 
        message: str = "벡터 저장소 작업 중 오류가 발생했습니다",
        details: Optional[Dict[str, Any]] = None
    ) -> None:
        super().__init__(message, details, "VECTOR_STORE_ERROR")


class CollectionNotFoundError(VectorStoreError):
    """지정된 컬렉션을 찾을 수 없음"""
    
    def __init__(
        self, 
        collection_name: str,
        details: Optional[Dict[str, Any]] = None
    ) -> None:
        message = f"컬렉션 '{collection_name}'을 찾을 수 없습니다"
        super().__init__(message, details)
        self.error_code = "COLLECTION_NOT_FOUND"
        self.collection_name = collection_name


class DocumentNotFoundError(VectorStoreError):
    """지정된 문서를 찾을 수 없음"""
    
    def __init__(
        self, 
        document_id: str,
        details: Optional[Dict[str, Any]] = None
    ) -> None:
        message = f"문서 '{document_id}'를 찾을 수 없습니다"
        super().__init__(message, details)
        self.error_code = "DOCUMENT_NOT_FOUND"
        self.document_id = document_id


# ============================================================================
# 문서 처리 관련 예외
# ============================================================================

class DocumentProcessingError(RAGBaseException):
    """문서 처리 중 발생하는 일반 오류"""
    
    def __init__(
        self, 
        message: str = "문서 처리 중 오류가 발생했습니다",
        details: Optional[Dict[str, Any]] = None
    ) -> None:
        super().__init__(message, details, "DOCUMENT_PROCESSING_ERROR")


class UnsupportedFileTypeError(DocumentProcessingError):
    """지원하지 않는 파일 형식"""
    
    def __init__(
        self, 
        file_type: str,
        supported_types: Optional[list] = None,
        details: Optional[Dict[str, Any]] = None
    ) -> None:
        supported = supported_types or [".pdf", ".docx", ".txt", ".md"]
        message = f"지원하지 않는 파일 형식입니다: {file_type}. 지원 형식: {', '.join(supported)}"
        super().__init__(message, details)
        self.error_code = "UNSUPPORTED_FILE_TYPE"
        self.file_type = file_type


class FileTooLargeError(DocumentProcessingError):
    """파일 크기 초과"""
    
    def __init__(
        self, 
        file_size: int,
        max_size: int = 52428800,  # 50MB
        details: Optional[Dict[str, Any]] = None
    ) -> None:
        max_mb = max_size // (1024 * 1024)
        file_mb = file_size / (1024 * 1024)
        message = f"파일이 너무 큽니다: {file_mb:.1f}MB. 최대 {max_mb}MB까지 지원합니다."
        super().__init__(message, details)
        self.error_code = "FILE_TOO_LARGE"
        self.file_size = file_size
        self.max_size = max_size


# ============================================================================
# 임베딩 관련 예외
# ============================================================================

class EmbeddingError(RAGBaseException):
    """임베딩 생성 중 발생하는 오류"""
    
    def __init__(
        self, 
        message: str = "임베딩 생성 중 오류가 발생했습니다",
        details: Optional[Dict[str, Any]] = None
    ) -> None:
        super().__init__(message, details, "EMBEDDING_ERROR")


class EmptyTextError(EmbeddingError):
    """빈 텍스트에 대한 임베딩 시도"""
    
    def __init__(
        self, 
        details: Optional[Dict[str, Any]] = None
    ) -> None:
        message = "빈 텍스트는 임베딩할 수 없습니다"
        super().__init__(message, details)
        self.error_code = "EMPTY_TEXT"


# ============================================================================
# 입력 검증 관련 예외
# ============================================================================

class ValidationError(RAGBaseException):
    """입력 유효성 검증 오류"""
    
    def __init__(
        self, 
        message: str = "입력값이 올바르지 않습니다",
        field: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None
    ) -> None:
        super().__init__(message, details, "VALIDATION_ERROR")
        self.field = field


class QuestionTooLongError(ValidationError):
    """질문이 너무 김"""
    
    def __init__(
        self, 
        length: int,
        max_length: int = 1000,
        details: Optional[Dict[str, Any]] = None
    ) -> None:
        message = f"질문이 너무 깁니다: {length}자. 최대 {max_length}자까지 허용됩니다."
        super().__init__(message, "question", details)
        self.error_code = "QUESTION_TOO_LONG"


class InvalidUserIdError(ValidationError):
    """잘못된 사용자 ID"""
    
    def __init__(
        self, 
        user_id: str,
        details: Optional[Dict[str, Any]] = None
    ) -> None:
        message = f"유효하지 않은 사용자 ID: {user_id}"
        super().__init__(message, "user_id", details)
        self.error_code = "INVALID_USER_ID"


# ============================================================================
# 연결 관련 예외
# ============================================================================

class ConnectionError(RAGBaseException):
    """외부 서비스 연결 오류"""
    
    def __init__(
        self, 
        service_name: str,
        message: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None
    ) -> None:
        msg = message or f"{service_name} 연결에 실패했습니다"
        super().__init__(msg, details, "CONNECTION_ERROR")
        self.service_name = service_name


class QdrantConnectionError(ConnectionError):
    """Qdrant 연결 오류"""
    
    def __init__(
        self, 
        host: str = "localhost",
        port: int = 6333,
        details: Optional[Dict[str, Any]] = None
    ) -> None:
        message = f"Qdrant({host}:{port})에 연결할 수 없습니다"
        super().__init__("Qdrant", message, details)
        self.error_code = "QDRANT_CONNECTION_ERROR"
