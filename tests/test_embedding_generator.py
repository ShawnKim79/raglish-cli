"""
EmbeddingGenerator 클래스에 대한 단위 테스트
"""

import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from document_rag_english_study.rag.embeddings import EmbeddingGenerator


class TestEmbeddingGenerator:
    """EmbeddingGenerator 클래스 테스트"""
    
    @pytest.fixture
    def temp_cache_dir(self):
        """임시 캐시 디렉토리 생성"""
        with tempfile.TemporaryDirectory() as temp_dir:
            yield temp_dir
    
    @pytest.fixture
    def mock_sentence_transformer(self):
        """SentenceTransformer 모킹"""
        with patch('document_rag_english_study.rag.embeddings.SentenceTransformer') as mock:
            mock_model = MagicMock()
            mock_model.encode.return_value = [0.1, 0.2, 0.3, 0.4, 0.5]
            mock_model.get_sentence_embedding_dimension.return_value = 5
            mock.return_value = mock_model
            yield mock_model
    
    def test_init_default_parameters(self, mock_sentence_transformer):
        """기본 매개변수로 초기화 테스트"""
        generator = EmbeddingGenerator()
        
        assert generator.model_name == "all-MiniLM-L6-v2"
        assert generator.batch_size == 32
        assert generator.enable_cache is True
        assert generator.cache_dir.name == "embeddings"
        assert generator.model is not None
    
    def test_init_custom_parameters(self, mock_sentence_transformer, temp_cache_dir):
        """사용자 정의 매개변수로 초기화 테스트"""
        generator = EmbeddingGenerator(
            model_name="custom-model",
            cache_dir=temp_cache_dir,
            enable_cache=False,
            batch_size=16
        )
        
        assert generator.model_name == "custom-model"
        assert generator.batch_size == 16
        assert generator.enable_cache is False
        assert str(generator.cache_dir) == temp_cache_dir
    
    def test_get_cache_key(self, mock_sentence_transformer):
        """캐시 키 생성 테스트"""
        generator = EmbeddingGenerator()
        
        key1 = generator._get_cache_key("test text")
        key2 = generator._get_cache_key("test text")
        key3 = generator._get_cache_key("different text")
        
        # 같은 텍스트는 같은 키 생성
        assert key1 == key2
        # 다른 텍스트는 다른 키 생성
        assert key1 != key3
        # SHA256 해시 길이 확인
        assert len(key1) == 64
    
    def test_generate_embedding_success(self, mock_sentence_transformer):
        """임베딩 생성 성공 테스트"""
        generator = EmbeddingGenerator(enable_cache=False)
        
        result = generator.generate_embedding("test text")
        
        assert result == [0.1, 0.2, 0.3, 0.4, 0.5]
        mock_sentence_transformer.encode.assert_called_once_with("test text", convert_to_tensor=False)
    
    def test_generate_embedding_empty_text(self, mock_sentence_transformer):
        """빈 텍스트 입력 시 오류 테스트"""
        generator = EmbeddingGenerator()
        
        with pytest.raises(ValueError, match="빈 텍스트는 임베딩을 생성할 수 없습니다"):
            generator.generate_embedding("")
        
        with pytest.raises(ValueError, match="빈 텍스트는 임베딩을 생성할 수 없습니다"):
            generator.generate_embedding("   ")
    
    def test_generate_embedding_with_cache(self, mock_sentence_transformer, temp_cache_dir):
        """캐싱 기능 테스트"""
        generator = EmbeddingGenerator(cache_dir=temp_cache_dir)
        
        # 첫 번째 호출 - 모델에서 생성
        result1 = generator.generate_embedding("test text")
        assert result1 == [0.1, 0.2, 0.3, 0.4, 0.5]
        assert mock_sentence_transformer.encode.call_count == 1
        
        # 두 번째 호출 - 캐시에서 로드
        result2 = generator.generate_embedding("test text")
        assert result2 == [0.1, 0.2, 0.3, 0.4, 0.5]
        # 모델 호출 횟수는 증가하지 않음
        assert mock_sentence_transformer.encode.call_count == 1
    
    def test_generate_batch_embeddings_success(self, mock_sentence_transformer):
        """배치 임베딩 생성 성공 테스트"""
        # 개별 호출을 위한 모킹 설정
        mock_sentence_transformer.encode.side_effect = [
            [0.1, 0.2, 0.3],  # text1
            [0.4, 0.5, 0.6],  # text2
            [0.7, 0.8, 0.9]   # text3
        ]
        
        generator = EmbeddingGenerator(enable_cache=False, batch_size=2)
        texts = ["text1", "text2", "text3"]
        
        results = generator.generate_batch_embeddings(texts)
        
        assert len(results) == 3
        assert results[0] == [0.1, 0.2, 0.3]
        assert results[1] == [0.4, 0.5, 0.6]
        assert results[2] == [0.7, 0.8, 0.9]
    
    def test_generate_batch_embeddings_empty_list(self, mock_sentence_transformer):
        """빈 리스트 입력 시 오류 테스트"""
        generator = EmbeddingGenerator()
        
        with pytest.raises(ValueError, match="빈 텍스트 리스트는 처리할 수 없습니다"):
            generator.generate_batch_embeddings([])
    
    def test_generate_batch_embeddings_with_empty_texts(self, mock_sentence_transformer):
        """빈 텍스트가 포함된 리스트 처리 테스트"""
        mock_sentence_transformer.encode.side_effect = [
            [0.1, 0.2, 0.3],  # "valid text"에 대한 임베딩
            [0.1, 0.2, 0.3]   # get_sentence_embedding_dimension 호출용
        ]
        mock_sentence_transformer.get_sentence_embedding_dimension.return_value = 3
        
        generator = EmbeddingGenerator(enable_cache=False)
        texts = ["", "valid text", "   "]
        
        results = generator.generate_batch_embeddings(texts)
        
        assert len(results) == 3
        assert results[0] == [0.0, 0.0, 0.0]  # 빈 텍스트는 0 벡터
        assert results[1] == [0.1, 0.2, 0.3]  # 유효한 텍스트
        assert results[2] == [0.0, 0.0, 0.0]  # 공백만 있는 텍스트는 0 벡터
    
    def test_generate_batch_embeddings_with_cache(self, mock_sentence_transformer, temp_cache_dir):
        """배치 처리에서 캐싱 기능 테스트"""
        mock_sentence_transformer.encode.side_effect = [
            [0.1, 0.2, 0.3],  # text1
            [0.4, 0.5, 0.6],  # text2
            [0.7, 0.8, 0.9]   # text3 (캐시되지 않은 텍스트)
        ]
        
        generator = EmbeddingGenerator(cache_dir=temp_cache_dir, batch_size=2)
        
        # 첫 번째 배치 처리
        texts1 = ["text1", "text2"]
        results1 = generator.generate_batch_embeddings(texts1)
        assert len(results1) == 2
        
        # 두 번째 배치 처리 (일부는 캐시에서, 일부는 새로 생성)
        texts2 = ["text1", "text3"]  # text1은 캐시에서, text3은 새로 생성
        results2 = generator.generate_batch_embeddings(texts2)
        assert len(results2) == 2
        assert results2[0] == results1[0]  # text1은 캐시에서 가져온 것
        assert results2[1] == [0.7, 0.8, 0.9]  # text3은 새로 생성
    
    def test_get_embedding_dimension(self, mock_sentence_transformer):
        """임베딩 차원 수 반환 테스트"""
        generator = EmbeddingGenerator()
        
        dimension = generator.get_embedding_dimension()
        
        assert dimension == 5
        mock_sentence_transformer.get_sentence_embedding_dimension.assert_called_once()
    
    def test_clear_cache(self, mock_sentence_transformer, temp_cache_dir):
        """캐시 삭제 테스트"""
        generator = EmbeddingGenerator(cache_dir=temp_cache_dir)
        
        # 캐시에 데이터 추가
        generator.generate_embedding("test text")
        assert len(generator._memory_cache) > 0
        
        # 캐시 삭제
        generator.clear_cache()
        assert len(generator._memory_cache) == 0
    
    def test_get_cache_info(self, mock_sentence_transformer, temp_cache_dir):
        """캐시 정보 반환 테스트"""
        generator = EmbeddingGenerator(cache_dir=temp_cache_dir)
        
        # 캐시에 데이터 추가
        generator.generate_embedding("test text")
        
        cache_info = generator.get_cache_info()
        
        assert "memory_cache_size" in cache_info
        assert "file_cache_size" in cache_info
        assert "cache_dir" in cache_info
        assert "model_name" in cache_info
        assert "embedding_dimension" in cache_info
        
        assert cache_info["memory_cache_size"] >= 1
        assert cache_info["model_name"] == "all-MiniLM-L6-v2"
        assert cache_info["embedding_dimension"] == 5
    
    def test_model_loading_failure(self):
        """모델 로딩 실패 테스트"""
        with patch('document_rag_english_study.rag.embeddings.SentenceTransformer') as mock:
            mock.side_effect = Exception("Model loading failed")
            
            with pytest.raises(Exception):
                EmbeddingGenerator()
    
    def test_embedding_generation_failure(self, mock_sentence_transformer):
        """임베딩 생성 실패 테스트"""
        mock_sentence_transformer.encode.side_effect = Exception("Encoding failed")
        
        generator = EmbeddingGenerator(enable_cache=False)
        
        with pytest.raises(RuntimeError, match="임베딩 생성 실패"):
            generator.generate_embedding("test text")
    
    def test_batch_embedding_generation_failure(self, mock_sentence_transformer):
        """배치 임베딩 생성 실패 테스트"""
        mock_sentence_transformer.encode.side_effect = Exception("Batch encoding failed")
        
        generator = EmbeddingGenerator(enable_cache=False)
        
        with pytest.raises(RuntimeError, match="임베딩 생성 실패"):
            generator.generate_batch_embeddings(["text1", "text2"])