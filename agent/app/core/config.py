import os
from typing import Optional
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """애플리케이션 설정"""
    
    # 기본 설정
    APP_NAME: str = "RAG Agent Service"
    APP_VERSION: str = "1.0.0"
    DEBUG: bool = False
    
    # 서버 설정
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    
    # API 키 설정
    GEMINI_API_KEY: Optional[str] = None
    OPENAI_API_KEY: Optional[str] = None  # 백업용
    
    # Gemini 모델 설정
    GEMINI_MODEL: Optional[str] = None
    
    # 벡터 DB 설정
    QDRANT_HOST: str = "localhost"
    QDRANT_PORT: int = 6333
    QDRANT_URL: Optional[str] = None
    
    # 임베딩 모델 설정
    DEFAULT_EMBEDDING_MODEL: str = "jhgan/ko-sroberta-multitask"
    MULTIMODAL_EMBEDDING_MODEL: str = "clip-ViT-B-32"
    TEXT_EMBEDDING_MODEL: str = "all-MiniLM-L6-v2"
    
    # 문서 처리 설정
    MAX_FILE_SIZE: int = 50 * 1024 * 1024  # 50MB
    CHUNK_SIZE: int = 1000
    CHUNK_OVERLAP: int = 200
    
    # OCR 설정
    OCR_CONFIDENCE_THRESHOLD: float = 0.7
    IMAGE_ENHANCEMENT: bool = True
    
    # 검색 설정
    DEFAULT_SEARCH_LIMIT: int = 5
    SEARCH_SCORE_THRESHOLD: float = 0.7
    
    # Redis 설정 (캐싱용)
    REDIS_HOST: str = "localhost"
    REDIS_PORT: int = 6379
    REDIS_DB: int = 0
    REDIS_PASSWORD: Optional[str] = None
    
    # MongoDB 설정 (민감 정보는 .env에서 설정)
    MONGODB_URL: str = ""  # .env에서 설정 필요 (mongodb://host:port)
    MONGODB_DB_NAME: str = ""  # .env에서 설정 필요
    MONGODB_USER: str = ""  # .env에서 설정 필요
    MONGODB_PASSWORD: str = ""  # .env에서 설정 필요
    
    # InfluxDB 설정 (민감 정보는 .env에서 설정)
    INFLUXDB_URL: str = ""  # .env에서 설정 필요
    INFLUXDB_TOKEN: str = ""  # .env에서 설정 필요 (필수)
    INFLUXDB_ORG: str = ""  # .env에서 설정 필요
    INFLUXDB_BUCKET: str = ""  # .env에서 설정 필요
    
    # MongoDB 컬렉션 설정
    MONGODB_COLLECTION_PRODUCTS: str = "products"
    MONGODB_COLLECTION_ABNORMALS: str = "abnormals"
    MONGODB_COLLECTION_ABNORMAL_SUMMARY: str = "abnormalSummary"
    MONGODB_COLLECTION_MACHINES: str = "machineMaster"
    MONGODB_COLLECTION_TOOLS: str = "toolMaster"
    MONGODB_COLLECTION_THRESHOLDS: str = "thresholdMaster"
    MONGODB_COLLECTION_LINES: str = "lineMaster"
    MONGODB_COLLECTION_OPERATIONS: str = "operationMaster"
    MONGODB_COLLECTION_WORKSHOPS: str = "workshopMaster"
    MONGODB_COLLECTION_TOOL_CHANGE: str = "tool_change"
    MONGODB_COLLECTION_TOOL_HISTORY: str = "tool_history"
    MONGODB_COLLECTION_USERS: str = "userMaster"

    
    # InfluxDB Measurement 설정
    INFLUXDB_MEASUREMENT_RAW: str = "cnc_analyze"
    INFLUXDB_MEASUREMENT_PRODUCT: str = "cnc_product"
    INFLUXDB_MEASUREMENT_TOOL_HISTORY: str = "cnc_tool"
    
    # 기본 필터 설정 (개발/테스트용 - 빈 값이면 모든 데이터 조회)
    DEFAULT_WORKSHOP_ID: str = "F01"
    DEFAULT_LINE_ID: str = "OR-T3"
    DEFAULT_OP_CODE: str = "OP10-1"
    
    # 로깅 설정
    LOG_LEVEL: str = "INFO"
    LOG_FORMAT: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


# 전역 설정 인스턴스
settings = Settings()