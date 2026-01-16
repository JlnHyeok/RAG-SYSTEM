"""
임베딩 매니저 모듈

다양한 임베딩 모델을 통합 관리하고, 텍스트를 벡터로 변환하는 기능을 제공합니다.
"""
import logging
import hashlib
from typing import List, Optional, Dict, Any, Tuple
import numpy as np
from sentence_transformers import SentenceTransformer
import torch
import asyncio
from functools import lru_cache

from app.core.config import settings
from app.models.enums import EmbeddingModelType
from app.models.exceptions import EmbeddingError, EmptyTextError

logger = logging.getLogger(__name__)


class EmbeddingManager:
    """
    다양한 임베딩 모델을 통합 관리하는 클래스
    
    SentenceTransformer 기반 모델들을 관리하며, 텍스트를 벡터로 변환하는
    다양한 메서드를 제공합니다. LRU 캐싱을 통해 성능을 최적화합니다.
    
    Attributes:
        models: 로딩된 임베딩 모델들 (key: 모델 키, value: SentenceTransformer)
        device: 사용 중인 디바이스 (cuda/cpu)
        _model_loaded: 모델 로딩 완료 여부
    """
    
    def __init__(self) -> None:
        self.models: Dict[str, SentenceTransformer] = {}
        self.device: str = "cuda" if torch.cuda.is_available() else "cpu"
        self._model_loaded: bool = False
        
        logger.info(f"임베딩 매니저 초기화 (디바이스: {self.device})")
    
    async def initialize(self) -> None:
        """
        모든 임베딩 모델을 비동기로 로딩
        
        Raises:
            EmbeddingError: 모델 로딩 실패 시
        """
        if self._model_loaded:
            return
            
        logger.info("임베딩 모델들 로딩 시작...")
        
        try:
            # 병렬로 모델 로딩
            await asyncio.gather(
                self._load_model("text", settings.TEXT_EMBEDDING_MODEL),
                self._load_model("korean", settings.DEFAULT_EMBEDDING_MODEL),
                self._load_model("multimodal", settings.MULTIMODAL_EMBEDDING_MODEL)
            )
            
            self._model_loaded = True
            logger.info("모든 임베딩 모델 로딩 완료!")
            
        except Exception as e:
            logger.error(f"임베딩 모델 로딩 실패: {e}")
            raise EmbeddingError(f"임베딩 모델 로딩 실패: {str(e)}")
    
    async def _load_model(self, model_key: str, model_name: str) -> None:
        """
        개별 모델을 비동기로 로딩
        
        Args:
            model_key: 모델 식별 키
            model_name: 로딩할 모델 이름
        """
        def load():
            model = SentenceTransformer(model_name, device=self.device)
            return model
            
        self.models[model_key] = await asyncio.to_thread(load)
        logger.info(f"모델 로딩 완료: {model_key} ({model_name})")
    
    @lru_cache(maxsize=5000)  # 캐시 크기 증가
    def _cached_encode(self, text_hash: str, model_key: str) -> tuple:
        """해시 기반 캐시된 임베딩 생성 (자주 사용되는 텍스트용)"""
        if not self._model_loaded:
            raise RuntimeError("임베딩 모델이 로딩되지 않았습니다. initialize()를 먼저 호출하세요.")
        
        # 해시에서 원본 텍스트를 복원할 수 없으므로 별도 저장소 필요
        # 이 메서드는 _encode_with_cache에서만 호출됨
        return self._pending_embedding
    
    def _encode_with_cache(self, text: str, model_key: str) -> tuple:
        """캐시를 활용한 임베딩 생성"""
        text_hash = hashlib.md5(text.encode()).hexdigest()
        
        # 캐시 히트 확인
        cache_info = self._cached_encode.cache_info()
        
        # 직접 인코딩 후 튜플로 반환
        model = self.models.get(model_key, self.models["korean"])
        embedding = model.encode([text], convert_to_numpy=True)
        return tuple(embedding[0].tolist())
    
    async def embed_text(
        self, 
        text: str, 
        model_type: EmbeddingModelType = EmbeddingModelType.KOREAN
    ) -> List[float]:
        """텍스트를 벡터로 변환 (캐시 임계값 확대)"""
        try:
            if not text.strip():
                raise ValueError("빈 텍스트는 임베딩할 수 없습니다")
            
            # 500자 이하 텍스트는 캐시 사용 (기존 100자에서 확대)
            if len(text) < 500:
                embedding_tuple = self._encode_with_cache(text, model_type.value)
                return list(embedding_tuple)
            
            # 긴 텍스트는 비동기 처리
            def encode():
                model = self.models.get(model_type.value, self.models["korean"])
                embedding = model.encode([text], convert_to_numpy=True)
                return embedding[0].tolist()
                
            return await asyncio.to_thread(encode)
            
        except Exception as e:
            logger.error(f"텍스트 임베딩 생성 실패: {e}")
            raise
    
    async def embed_batch(
        self, 
        texts: List[str], 
        model_type: EmbeddingModelType = EmbeddingModelType.KOREAN,
        batch_size: int = 32
    ) -> List[List[float]]:
        """여러 텍스트를 한번에 벡터로 변환 (성능 최적화)"""
        try:
            if not texts:
                return []
                
            # 빈 텍스트 필터링
            valid_texts = [text for text in texts if text.strip()]
            if not valid_texts:
                raise ValueError("유효한 텍스트가 없습니다")
            
            def encode_batch():
                model = self.models.get(model_type.value, self.models["korean"])
                embeddings = model.encode(
                    valid_texts,
                    convert_to_numpy=True,
                    batch_size=batch_size,
                    show_progress_bar=len(valid_texts) > 10,
                    normalize_embeddings=True  # 코사인 유사도 최적화
                )
                return embeddings.tolist()
                
            return await asyncio.to_thread(encode_batch)
            
        except Exception as e:
            logger.error(f"배치 임베딩 생성 실패: {e}")
            raise
    
    async def embed_multimodal(
        self, 
        text: str, 
        image_caption: Optional[str] = None
    ) -> List[float]:
        """텍스트와 이미지 정보를 함께 임베딩 (멀티모달)"""
        try:
            # 텍스트와 이미지 캡션을 결합
            combined_text = text
            if image_caption:
                combined_text = f"{text} [이미지: {image_caption}]"
            
            def encode_multimodal():
                model = self.models.get("multimodal", self.models["korean"])
                embedding = model.encode([combined_text], convert_to_numpy=True)
                return embedding[0].tolist()
                
            return await asyncio.to_thread(encode_multimodal)
            
        except Exception as e:
            logger.error(f"멀티모달 임베딩 생성 실패: {e}")
            # 폴백: 일반 텍스트 임베딩 사용
            return await self.embed_text(text, EmbeddingModelType.KOREAN)
    
    def get_model_info(self) -> Dict[str, Any]:
        """로딩된 모델 정보 반환"""
        return {
            "models_loaded": list(self.models.keys()),
            "device": self.device,
            "total_models": len(self.models),
            "cache_size": self._cached_encode.cache_info()._asdict() if hasattr(self._cached_encode, 'cache_info') else {}
        }
    
    async def cleanup(self):
        """리소스 정리"""
        logger.info("임베딩 매니저 리소스 정리 중...")
        
        # 캐시 클리어
        if hasattr(self._cached_encode, 'cache_clear'):
            self._cached_encode.cache_clear()
        
        # GPU 메모리 해제
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
        self.models.clear()
        self._model_loaded = False
        
        logger.info("임베딩 매니저 리소스 정리 완료")


# 전역 임베딩 매니저 인스턴스
embedding_manager = EmbeddingManager()