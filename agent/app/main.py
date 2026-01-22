from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi import WebSocket, WebSocketDisconnect
from contextlib import asynccontextmanager
import logging
import sys

# 로그 컬러 포맷터 정의
class ColouredFormatter(logging.Formatter):
    grey = "\x1b[38;20m"
    green = "\x1b[32;20m"
    yellow = "\x1b[33;20m"
    red = "\x1b[31;20m"
    bold_red = "\x1b[31;1m"
    reset = "\x1b[0m"
    # 포맷 분리: [시간] [레벨] (색상) + 파일:라인 - 메시지 (기본색)
    prefix_fmt = "%(asctime)s [%(levelname)s]"
    suffix_fmt = " %(filename)s:%(lineno)d - %(message)s"

    FORMATS = {
        logging.DEBUG: grey + prefix_fmt + reset + suffix_fmt,
        logging.INFO: green + prefix_fmt + reset + suffix_fmt,
        logging.WARNING: yellow + prefix_fmt + reset + suffix_fmt,
        logging.ERROR: red + prefix_fmt + reset + suffix_fmt,
        logging.CRITICAL: bold_red + prefix_fmt + reset + suffix_fmt
    }

    def format(self, record):
        log_fmt = self.FORMATS.get(record.levelno)
        # 시간 포맷은 밀리초 포함하여 이전과 유사하게
        formatter = logging.Formatter(log_fmt, datefmt="%Y-%m-%d %H:%M:%S")
        return formatter.format(record)


from app.core import settings, hybrid_rag_engine, progress_websocket
from app.api.v1 import health, query, documents

# 로깅 설정
# 로깅 설정 (커스텀 컬러 포맷 적용)
root_logger = logging.getLogger()
root_logger.setLevel(getattr(logging, settings.LOG_LEVEL))

# 기존 핸들러 제거 후 커스텀 핸들러 추가
if root_logger.handlers:
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)

console_handler = logging.StreamHandler(sys.stdout)
console_handler.setFormatter(ColouredFormatter())
root_logger.addHandler(console_handler)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """앱 생명주기 관리"""
    try:
        # 하이브리드 RAG 엔진 초기화 (내부에서 상세 리포트 출력)
        await hybrid_rag_engine.initialize()
        
        # 앱 상태에 저장
        app.state.hybrid_rag_engine = hybrid_rag_engine
        
    except Exception as e:
        logger.error(f"앱 초기화 실패: {e}")
        raise
    
    yield
    
    # 종료 시
    logger.info(f"{settings.APP_NAME} 종료 중...")
    
    try:
        await hybrid_rag_engine.cleanup()
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