"""
임베딩 생성기 모듈

sentence-transformers를 활용한 텍스트 임베딩 생성, 배치 처리 및 캐싱 기능을 제공합니다.
"""

import hashlib
import logging
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Union

from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)


class EmbeddingGenerator:
    """
    텍스트 임베딩 생성을 담당하는 클래스
    
    sentence-transformers를 사용하여 텍스트를 벡터로 변환하고,
    성능 향상을 위한 배치 처리 및 캐싱 기능을 제공합니다.
    """
    
    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        cache_dir: Optional[str] = None,
        enable_cache: bool = True,
        batch_size: int = 32
    ):
        """
        EmbeddingGenerator 초기화
        
        Args:
            model_name: 사용할 sentence-transformers 모델명
            cache_dir: 캐시 파일을 저장할 디렉토리 경로
            enable_cache: 캐싱 기능 활성화 여부
            batch_size: 배치 처리 시 한 번에 처리할 텍스트 수
        """
        self.model_name = model_name
        self.batch_size = batch_size
        self.enable_cache = enable_cache
        
        # 캐시 디렉토리 설정
        if cache_dir:
            self.cache_dir = Path(cache_dir)
        else:
            self.cache_dir = Path.home() / ".cache" / "document_rag_english_study" / "embeddings"
        
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # 모델 로드
        try:
            logger.info(f"임베딩 모델 로드 중: {model_name}")
            self.model = SentenceTransformer(model_name)
            logger.info("임베딩 모델 로드 완료")
        except Exception as e:
            logger.error(f"임베딩 모델 로드 실패: {e}")
            raise
        
        # 메모리 캐시
        self._memory_cache: Dict[str, List[float]] = {}
    
    def _get_cache_key(self, text: str) -> str:
        """
        텍스트에 대한 캐시 키 생성
        
        Args:
            text: 입력 텍스트
            
        Returns:
            SHA256 해시 기반 캐시 키
        """
        # 모델명과 텍스트를 조합하여 고유한 키 생성
        content = f"{self.model_name}:{text}"
        return hashlib.sha256(content.encode('utf-8')).hexdigest()
    
    def _get_cache_file_path(self, cache_key: str) -> Path:
        """
        캐시 파일 경로 생성
        
        Args:
            cache_key: 캐시 키
            
        Returns:
            캐시 파일 경로
        """
        return self.cache_dir / f"{cache_key}.pkl"
    
    def _load_from_cache(self, cache_key: str) -> Optional[List[float]]:
        """
        캐시에서 임베딩 로드
        
        Args:
            cache_key: 캐시 키
            
        Returns:
            캐시된 임베딩 벡터 또는 None
        """
        if not self.enable_cache:
            return None
        
        # 메모리 캐시 확인
        if cache_key in self._memory_cache:
            return self._memory_cache[cache_key]
        
        # 파일 캐시 확인
        cache_file = self._get_cache_file_path(cache_key)
        if cache_file.exists():
            try:
                with open(cache_file, 'rb') as f:
                    embedding = pickle.load(f)
                # 메모리 캐시에도 저장
                self._memory_cache[cache_key] = embedding
                return embedding
            except Exception as e:
                logger.warning(f"캐시 파일 로드 실패: {cache_file}, 오류: {e}")
                # 손상된 캐시 파일 삭제
                try:
                    cache_file.unlink()
                except:
                    pass
        
        return None
    
    def _save_to_cache(self, cache_key: str, embedding: List[float]) -> None:
        """
        임베딩을 캐시에 저장
        
        Args:
            cache_key: 캐시 키
            embedding: 저장할 임베딩 벡터
        """
        if not self.enable_cache:
            return
        
        # 메모리 캐시에 저장
        self._memory_cache[cache_key] = embedding
        
        # 파일 캐시에 저장
        cache_file = self._get_cache_file_path(cache_key)
        try:
            with open(cache_file, 'wb') as f:
                pickle.dump(embedding, f)
        except Exception as e:
            logger.warning(f"캐시 파일 저장 실패: {cache_file}, 오류: {e}")
    
    def generate_embedding(self, text: str) -> List[float]:
        """
        단일 텍스트에 대한 임베딩 생성
        
        Args:
            text: 임베딩을 생성할 텍스트
            
        Returns:
            임베딩 벡터 (리스트 형태)
            
        Raises:
            ValueError: 빈 텍스트가 입력된 경우
            RuntimeError: 임베딩 생성 실패 시
        """
        if not text or not text.strip():
            raise ValueError("빈 텍스트는 임베딩을 생성할 수 없습니다")
        
        text = text.strip()
        cache_key = self._get_cache_key(text)
        
        # 캐시에서 확인
        cached_embedding = self._load_from_cache(cache_key)
        if cached_embedding is not None:
            logger.debug(f"캐시에서 임베딩 로드: {text[:50]}...")
            return cached_embedding
        
        try:
            # 임베딩 생성
            logger.debug(f"임베딩 생성 중: {text[:50]}...")
            embedding = self.model.encode(text, convert_to_tensor=False)
            
            # numpy 배열인 경우 리스트로 변환, 이미 리스트인 경우 그대로 사용
            if hasattr(embedding, 'tolist'):
                embedding_list = embedding.tolist()
            else:
                embedding_list = list(embedding)
            
            # 캐시에 저장
            self._save_to_cache(cache_key, embedding_list)
            
            return embedding_list
            
        except Exception as e:
            logger.error(f"임베딩 생성 실패: {text[:50]}..., 오류: {e}")
            raise RuntimeError(f"임베딩 생성 실패: {e}")
    
    def generate_batch_embeddings(self, texts: List[str]) -> List[List[float]]:
        """
        여러 텍스트에 대한 임베딩 배치 생성
        
        Args:
            texts: 임베딩을 생성할 텍스트 리스트
            
        Returns:
            임베딩 벡터 리스트
            
        Raises:
            ValueError: 빈 텍스트 리스트가 입력된 경우
            RuntimeError: 임베딩 생성 실패 시
        """
        if not texts:
            raise ValueError("빈 텍스트 리스트는 처리할 수 없습니다")
        
        logger.info(f"배치 임베딩 생성 시작: {len(texts)}개 텍스트")
        
        results = []
        
        # 각 텍스트에 대해 개별적으로 처리 (캐싱 활용)
        for text in texts:
            if not text or not text.strip():
                # 빈 텍스트의 경우 0 벡터 생성
                logger.warning(f"빈 텍스트가 발견되어 0 벡터를 생성합니다")
                embedding_dim = self.get_embedding_dimension()
                results.append([0.0] * embedding_dim)
            else:
                # 유효한 텍스트는 개별 임베딩 생성 (캐싱 활용)
                embedding = self.generate_embedding(text.strip())
                results.append(embedding)
        
        logger.info(f"배치 임베딩 생성 완료: {len(results)}개 벡터")
        return results
    
    def get_embedding_dimension(self) -> int:
        """
        임베딩 벡터의 차원 수 반환
        
        Returns:
            임베딩 벡터 차원 수
        """
        return self.model.get_sentence_embedding_dimension()
    
    def clear_cache(self) -> None:
        """
        모든 캐시 삭제 (메모리 및 파일)
        """
        # 메모리 캐시 삭제
        self._memory_cache.clear()
        
        # 파일 캐시 삭제
        if self.cache_dir.exists():
            try:
                for cache_file in self.cache_dir.glob("*.pkl"):
                    cache_file.unlink()
                logger.info("캐시 파일 삭제 완료")
            except Exception as e:
                logger.warning(f"캐시 파일 삭제 중 오류 발생: {e}")
    
    def get_cache_info(self) -> Dict[str, Union[int, str]]:
        """
        캐시 정보 반환
        
        Returns:
            캐시 통계 정보
        """
        memory_cache_size = len(self._memory_cache)
        
        file_cache_size = 0
        if self.cache_dir.exists():
            file_cache_size = len(list(self.cache_dir.glob("*.pkl")))
        
        return {
            "memory_cache_size": memory_cache_size,
            "file_cache_size": file_cache_size,
            "cache_dir": str(self.cache_dir),
            "model_name": self.model_name,
            "embedding_dimension": self.get_embedding_dimension()
        }