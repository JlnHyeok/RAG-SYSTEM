from fastapi import APIRouter
from app.models.schemas import HealthResponse
from app.core.rag_engine import rag_engine
from app.core.embedding_manager import embedding_manager
from app.core.vector_store import vector_store
from app.core.gemini_service import gemini_service


router = APIRouter()


@router.get("/", response_model=HealthResponse)
async def health_check():
    """서비스 헬스 체크"""
    
    try:
        # RAG 엔진 상태 확인
        rag_status = await rag_engine.health_check()
        
        return HealthResponse(
            status="healthy" if rag_status.get("rag_engine") == "healthy" else "unhealthy",
            service="RAG Agent Service",
            version="1.0.0",
            models_loaded=rag_status.get("components", {}),
            error=rag_status.get("error")  # 에러 정보도 포함
        )
    except Exception as e:
        return HealthResponse(
            status="unhealthy",
            service="RAG Agent Service", 
            version="1.0.0",
            models_loaded={},
            error=f"Health check failed: {str(e)}"
        )


@router.get("/detailed")
async def detailed_health_check():
    """상세 헬스 체크 - 각 컴포넌트별 상태 확인"""
    
    components = {}
    
    # 1. 임베딩 모델 체크
    try:
        embedding_status = embedding_manager._initialized
        components["embedding_models"] = {
            "status": "healthy" if embedding_status else "unhealthy",
            "initialized": embedding_status,
            "models": list(getattr(embedding_manager, 'models', {}).keys()) if embedding_status else []
        }
    except Exception as e:
        components["embedding_models"] = {
            "status": "error",
            "error": str(e)
        }
    
    # 2. Qdrant 벡터스토어 체크  
    try:
        vector_status = await vector_store.health_check()
        components["vector_store"] = vector_status
    except Exception as e:
        components["vector_store"] = {
            "status": "error", 
            "error": str(e)
        }
    
    # 3. Gemini API 체크
    try:
        gemini_status = gemini_service._initialized
        components["gemini_api"] = {
            "status": "healthy" if gemini_status else "unhealthy",
            "initialized": gemini_status,
            "model": "gemini-2.5-flash" if gemini_status else "not_loaded"
        }
    except Exception as e:
        components["gemini_api"] = {
            "status": "error",
            "error": str(e)
        }
    
    # 전체 상태 결정
    all_healthy = all(
        comp.get("status") == "healthy" 
        for comp in components.values()
    )
    
    return {
        "overall_status": "healthy" if all_healthy else "unhealthy",
        "components": components,
        "timestamp": "2026-01-09"
    }