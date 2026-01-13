from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi import WebSocket, WebSocketDisconnect
from contextlib import asynccontextmanager
import logging

from app.core.config import settings
from app.core.rag_engine import rag_engine
from app.core.websocket_manager import progress_websocket
from app.api.v1 import health, query, documents

# 로깅 설정
logging.basicConfig(
    level=getattr(logging, settings.LOG_LEVEL),
    format=settings.LOG_FORMAT
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """앱 생명주기 관리"""
    # 시작 시
    logger.info(f"{settings.APP_NAME} 시작...")
    
    try:
        # RAG 엔진 초기화
        await rag_engine.initialize()
        logger.info("RAG 엔진 초기화 완료")
        
        # 앱 상태에 저장
        app.state.rag_engine = rag_engine
        
        logger.info(f"{settings.APP_NAME} 준비 완료!")
        
    except Exception as e:
        logger.error(f"앱 초기화 실패: {e}")
        raise
    
    yield
    
    # 종료 시
    logger.info(f"{settings.APP_NAME} 종료 중...")
    
    try:
        await rag_engine.cleanup()
        logger.info("리소스 정리 완료")
    except Exception as e:
        logger.error(f"리소스 정리 실패: {e}")


# FastAPI 앱 생성
app = FastAPI(
    title="RAG Agent Service",  # settings 대신 하드코딩
    description="멀티모달 RAG 시스템의 AI 처리 엔진",
    version="1.0.0",  # settings 대신 하드코딩
    lifespan=lifespan,
    docs_url="/docs",  # 항상 활성화
    redoc_url="/redoc",  # 항상 활성화
    # ReDoc 안정적인 버전 지정
    redoc_js_url="https://cdn.jsdelivr.net/npm/redoc@2.0.0/bundles/redoc.standalone.js",
    redoc_favicon_url="https://fastapi.tiangolo.com/img/favicon.png"
)

# CORS 미들웨어 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # TODO: 프로덕션에서는 특정 도메인으로 제한
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# API 라우터 등록
app.include_router(health.router, prefix="/api/v1", tags=["Health"])
app.include_router(query.router, prefix="/api/v1", tags=["Query"])  
app.include_router(documents.router, prefix="/api/v1/documents", tags=["Documents"])

# WebSocket 엔드포인트
@app.websocket("/ws/progress/{document_id}")
async def websocket_progress(websocket: WebSocket, document_id: str):
    """문서 처리 진행률을 실시간으로 스트리밍"""
    try:
        await progress_websocket.connect(websocket, document_id)
        logger.info(f"WebSocket 연결됨: {document_id}")
        
        try:
            while True:
                # 클라이언트로부터 메시지 대기 (연결 유지)
                data = await websocket.receive_text()
                
                # 핀/폁 메시지 처리
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


@app.get("/")
async def root():
    """루트 엔드포인트"""
    return {
        "service": settings.APP_NAME,
        "version": settings.APP_VERSION,
        "status": "running",
        "docs": "/docs" if settings.DEBUG else "disabled"
    }


# 전역 예외 핸들러
@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """전역 예외 처리"""
    logger.error(f"전역 예외 발생: {exc}", exc_info=True)
    return {
        "error": "Internal Server Error",
        "message": "서버 내부 오류가 발생했습니다",
        "detail": str(exc) if settings.DEBUG else "자세한 정보는 로그를 확인하세요"
    }


if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.DEBUG,
        log_level=settings.LOG_LEVEL.lower()
    )