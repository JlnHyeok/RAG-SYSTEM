from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from typing import Dict, Any, AsyncGenerator
import logging
import time
import json
import asyncio

from app.core.rag_engine import rag_engine
from app.models.schemas import QueryRequest, QueryResponse

logger = logging.getLogger(__name__)
router = APIRouter()


def _format_sources(sources) -> list:
    """소스 데이터를 일관된 형식으로 변환"""
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
    request: QueryRequest  # JSON body로 받도록 변경
) -> Dict[str, Any]:
    """Backend에서 호출하는 질문 처리 엔드포인트"""
    start_time = time.time()
    
    try:
        # 입력 검증
        if not request.question.strip():
            raise HTTPException(status_code=400, detail="질문이 비어있습니다")
        
        # user_id 검증 제거 (기본값 사용)
        logger.info(f"질문 처리 요청: '{request.question[:50]}...' (사용자: {request.user_id})")
        
        # RAG 엔진으로 질문 처리
        response = await rag_engine.query(request)
        
        # Backend가 기대하는 형태로 응답 변환
        return {
            "answer": response.answer,
            "sources": _format_sources(response.sources),
            "processing_time": response.processing_time,
            "confidence": response.confidence,
            "timestamp": response.timestamp.isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"질문 처리 실패: {e}")
        raise HTTPException(
            status_code=500, 
            detail=f"질문 처리 중 오류가 발생했습니다: {str(e)}"
        )


@router.post("/query/stream")
async def query_stream(
    request: QueryRequest  # JSON body로 통일
):
    """스트리밍 질문 처리 - 실시간으로 답변을 스트리밍합니다"""
    
    async def generate_stream() -> AsyncGenerator[str, None]:
        """답변을 청크 단위로 스트리밍하는 제너레이터"""
        try:
            # 시작 메시지
            yield f"data: {json.dumps({'type': 'start', 'message': '질문을 처리하고 있습니다...'}, ensure_ascii=False)}\n\n"
            
            # 입력 검증
            if not request.question.strip():
                yield f"data: {json.dumps({'type': 'error', 'message': '질문이 비어있습니다'}, ensure_ascii=False)}\n\n"
                return
            
            logger.info(f"스트리밍 질문 처리: '{request.question[:50]}...' (사용자: {request.user_id})")
            
            # 벡터 검색 진행 알림
            yield f"data: {json.dumps({'type': 'progress', 'message': '관련 문서를 검색 중...'}, ensure_ascii=False)}\n\n"
            await asyncio.sleep(0.1)  # 짧은 대기로 스트리밍 효과
            
            # RAG 엔진에서 답변 생성
            response = await rag_engine.query(request)
            
            # 답변 생성 시작 알림
            yield f"data: {json.dumps({'type': 'progress', 'message': 'AI가 답변을 생성 중...'}, ensure_ascii=False)}\n\n"
            await asyncio.sleep(0.1)
            
            # 답변을 청크로 나누어 스트리밍
            answer_chunks = response.answer.split()
            chunk_size = 3  # 단어 3개씩 묶어서 전송
            
            for i in range(0, len(answer_chunks), chunk_size):
                chunk = ' '.join(answer_chunks[i:i + chunk_size])
                yield f"data: {json.dumps({'type': 'chunk', 'content': chunk + ' '}, ensure_ascii=False)}\n\n"
                await asyncio.sleep(0.05)  # 스트리밍 효과를 위한 짧은 대기
            
            # 소스 정보 전송
            sources_data = _format_sources(response.sources)
            
            yield f"data: {json.dumps({'type': 'sources', 'sources': sources_data}, ensure_ascii=False)}\n\n"
            
            # 완료 메시지
            yield f"data: {json.dumps({'type': 'complete', 'confidence': response.confidence, 'timestamp': response.timestamp.isoformat()}, ensure_ascii=False)}\n\n"
            
        except Exception as e:
            logger.error(f"스트리밍 처리 실패: {e}")
            yield f"data: {json.dumps({'type': 'error', 'message': f'처리 중 오류가 발생했습니다: {str(e)}'}, ensure_ascii=False)}\n\n"
        
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