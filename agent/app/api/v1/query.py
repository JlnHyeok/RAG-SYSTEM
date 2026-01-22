from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from typing import Dict, Any, AsyncGenerator
import logging
import time
import json
import asyncio

from app.core import hybrid_rag_engine
from app.models.schemas import QueryRequest, QueryResponse

logger = logging.getLogger(__name__)
router = APIRouter()


def _format_sources(sources) -> list:
    """소스 데이터를 일관된 형식으로 변환"""
    if not sources:
        return []
    return [
        {
            "document_id": source.document_id,
            "file_path": source.metadata.get("file_path", ""),
            "relevance_score": source.score,
            "content_preview": source.content[:200] + "..." if len(source.content) > 200 else source.content
        }
        for source in sources
    ]


@router.post("/query", response_model=Dict[str, Any])
async def query(
    request: QueryRequest
) -> Dict[str, Any]:
    """모든 질문 처리 엔드포인트 (문서 + 하이브리드 자동 판별)"""
    try:
        if not request.question.strip():
            raise HTTPException(status_code=400, detail="질문이 비어있습니다")
        
        logger.info(f"질문 요청: '{request.question[:50]}...'")
        
        # 하이브리드 RAG 엔진으로 처리
        response = await hybrid_rag_engine.query(request)
        
        return {
            "answer": response.answer,
            "sources": _format_sources(response.sources),
            "processing_time": response.processing_time,
            "confidence": response.confidence,
            "timestamp": response.timestamp.isoformat(),
            "metadata": getattr(response, 'metadata', {})
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"질문 처리 실패: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"질문 처리 중 오류: {str(e)}"
        )


@router.post("/query/stream")
async def query_stream(
    request: QueryRequest
):
    """스트리밍 질문 처리 - DB 탐색 상태와 답변을 실시간 스트리밍"""
    
    async def generate_stream() -> AsyncGenerator[str, None]:
        try:
            yield f"data: {json.dumps({'type': 'start', 'message': '검색을 시작합니다...'}, ensure_ascii=False)}\n\n"
            
            queue = asyncio.Queue()
            
            async def on_status_handler(msg: str):
                await queue.put({'type': 'progress', 'message': msg})

            # 백그라운드에서 엔진 실행
            async def run_engine():
                try:
                    response = await hybrid_rag_engine.query(request, on_status=on_status_handler)
                    await queue.put({'type': 'result', 'data': response})
                except Exception as e:
                    await queue.put({'type': 'error', 'message': str(e)})
                finally:
                    await queue.put(None)

            asyncio.create_task(run_engine())
            
            while True:
                item = await queue.get()
                if item is None: break
                
                if item['type'] == 'progress':
                    yield f"data: {json.dumps(item, ensure_ascii=False)}\n\n"
                elif item['type'] == 'error':
                    yield f"data: {json.dumps(item, ensure_ascii=False)}\n\n"
                    return
                elif item['type'] == 'result':
                    response = item['data']
                    
                    # 답변 스트리밍
                    answer_chunks = response.answer.split()
                    for i in range(0, len(answer_chunks), 3):
                        chunk = ' '.join(answer_chunks[i:i + 3])
                        yield f"data: {json.dumps({'type': 'chunk', 'content': chunk + ' '}, ensure_ascii=False)}\n\n"
                        await asyncio.sleep(0.05)
                    
                    completion_data = {
                        'type': 'complete', 
                        'confidence': response.confidence, 
                        'sources': _format_sources(response.sources),
                        'metadata': getattr(response, 'metadata', {})
                    }
                    yield f"data: {json.dumps(completion_data, ensure_ascii=False)}\n\n"

        except Exception as e:
            logger.error(f"스트리밍 실패: {e}")
            yield f"data: {json.dumps({'type': 'error', 'message': str(e)}, ensure_ascii=False)}\n\n"
        finally:
            yield "data: [DONE]\n\n"

    return StreamingResponse(
        generate_stream(),
        media_type="text/plain",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "Content-Type": "text/plain; charset=utf-8"
        }
    )


# 하위 호환성을 위해 유지하거나 제거 가능
@router.post("/hybrid")
async def hybrid_query(request: QueryRequest):
    return await query(request)

@router.post("/hybrid/stream")
async def hybrid_stream(request: QueryRequest):
    return await query_stream(request)