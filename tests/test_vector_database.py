"""
VectorDatabase 클래스에 대한 단위 테스트
"""

import pytest
import tempfile
import shutil
from pathlib import Path
from typing import List, Dict, Any
from unittest.mock import Mock, patch

from src.document_rag_english_study.rag.vector_database import VectorDatabase
from src.document_rag_english_study.models.response import SearchResult


class TestVectorDatabase:
    """VectorDatabase 클래스 테스트"""
    
    @pytest.fixture
    def temp_dir(self):
        """임시 디렉토리 생성"""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir, ignore_errors=True)
    
    @pytest.fixture
    def vector_db(self, temp_dir):
        """테스트용 VectorDatabase 인스턴스"""
        return VectorDatabase(
            collection_name="test_collection",
            persist_directory=temp_dir
        )
    
    def test_initialization(self, temp_dir):
        """VectorDatabase 초기화 테스트"""
        db = VectorDatabase(
            collection_name="test_init",
            persist_directory=temp_dir
        )
        
        assert db.collection_name == "test_init"
        assert db.persist_directory == Path(temp_dir)
        assert db.client is not None
        assert db.collection is not None
    
    def test_initialization_default_directory(self):
        """기본 디렉토리로 초기화 테스트"""
        db = VectorDatabase(collection_name="test_default")
        
        expected_dir = Path.home() / ".cache" / "document_rag_english_study" / "chroma_db"
        assert db.persist_directory == expected_dir
        assert db.persist_directory.exists()
    
    def test_add_documents_basic(self, vector_db):
        """기본 문서 추가 테스트"""
        documents = [
            "This is a test document about machine learning.",
            "Another document discussing natural language processing."
        ]
        metadatas = [
            {"source_file": "ml_doc.txt", "page": 1},
            {"source_file": "nlp_doc.txt", "page": 1}
        ]
        
        ids = vector_db.add_documents(documents, metadatas)
        
        assert len(ids) == 2
        assert all(isinstance(doc_id, str) for doc_id in ids)
        assert vector_db.get_document_count() == 2
    
    def test_add_documents_with_custom_ids(self, vector_db):
        """사용자 정의 ID로 문서 추가 테스트"""
        documents = ["Test document with custom ID"]
        metadatas = [{"source_file": "custom.txt"}]
        custom_ids = ["custom_id_123"]
        
        returned_ids = vector_db.add_documents(documents, metadatas, ids=custom_ids)
        
        assert returned_ids == custom_ids
        assert vector_db.get_document_count() == 1
    
    def test_add_documents_with_embeddings(self, vector_db):
        """미리 계산된 임베딩으로 문서 추가 테스트"""
        documents = ["Document with pre-computed embedding"]
        metadatas = [{"source_file": "embedding_doc.txt"}]
        embeddings = [[0.1, 0.2, 0.3, 0.4, 0.5]]  # 간단한 5차원 벡터
        
        # ChromaDB의 기본 임베딩 함수를 모킹
        with patch.object(vector_db.collection, 'add') as mock_add:
            ids = vector_db.add_documents(documents, metadatas, embeddings=embeddings)
            
            mock_add.assert_called_once()
            call_args = mock_add.call_args[1]
            assert 'embeddings' in call_args
            assert call_args['embeddings'] == embeddings
    
    def test_add_documents_validation_errors(self, vector_db):
        """문서 추가 시 유효성 검사 오류 테스트"""
        # 빈 문서 리스트
        with pytest.raises(ValueError, match="추가할 문서가 없습니다"):
            vector_db.add_documents([], [])
        
        # 문서와 메타데이터 수 불일치
        with pytest.raises(ValueError, match="문서 수와 메타데이터 수가 일치하지 않습니다"):
            vector_db.add_documents(["doc1"], [{"meta1": 1}, {"meta2": 2}])
        
        # 문서와 ID 수 불일치
        with pytest.raises(ValueError, match="문서 수와 ID 수가 일치하지 않습니다"):
            vector_db.add_documents(["doc1"], [{"meta1": 1}], ids=["id1", "id2"])
        
        # 문서와 임베딩 수 불일치
        with pytest.raises(ValueError, match="문서 수와 임베딩 수가 일치하지 않습니다"):
            vector_db.add_documents(["doc1"], [{"meta1": 1}], embeddings=[[1, 2], [3, 4]])
    
    def test_search_similar_documents(self, vector_db):
        """유사 문서 검색 테스트"""
        # 테스트 문서 추가
        documents = [
            "Machine learning is a subset of artificial intelligence.",
            "Deep learning uses neural networks with multiple layers.",
            "Natural language processing deals with text analysis."
        ]
        metadatas = [
            {"source_file": "ml.txt", "topic": "machine_learning"},
            {"source_file": "dl.txt", "topic": "deep_learning"},
            {"source_file": "nlp.txt", "topic": "nlp"}
        ]
        
        vector_db.add_documents(documents, metadatas)
        
        # 검색 수행
        results = vector_db.search_similar_documents("artificial intelligence", n_results=2)
        
        assert len(results) <= 2
        assert all(isinstance(result, SearchResult) for result in results)
        
        if results:
            # 첫 번째 결과가 가장 관련성이 높아야 함
            assert results[0].relevance_score >= 0.0
            assert results[0].source_file in ["ml.txt", "dl.txt", "nlp.txt"]
            assert "machine learning" in results[0].content.lower() or "artificial intelligence" in results[0].content.lower()
    
    def test_search_with_metadata_filter(self, vector_db):
        """메타데이터 필터링을 사용한 검색 테스트"""
        documents = [
            "Python programming language",
            "Java programming language",
            "JavaScript for web development"
        ]
        metadatas = [
            {"source_file": "python.txt", "language": "python"},
            {"source_file": "java.txt", "language": "java"},
            {"source_file": "js.txt", "language": "javascript"}
        ]
        
        vector_db.add_documents(documents, metadatas)
        
        # 특정 언어로 필터링하여 검색
        results = vector_db.search_similar_documents(
            "programming",
            n_results=5,
            where={"language": "python"}
        )
        
        # 결과가 있다면 모두 Python 관련이어야 함
        for result in results:
            assert result.metadata.get("language") == "python"
    
    def test_search_validation_errors(self, vector_db):
        """검색 시 유효성 검사 오류 테스트"""
        # 빈 쿼리
        with pytest.raises(ValueError, match="검색 쿼리 또는 임베딩이 필요합니다"):
            vector_db.search_similar_documents("", n_results=5)
        
        # 잘못된 결과 수
        with pytest.raises(ValueError, match="결과 수는 1 이상이어야 합니다"):
            vector_db.search_similar_documents("test", n_results=0)
    
    def test_update_document(self, vector_db):
        """문서 업데이트 테스트"""
        # 문서 추가
        documents = ["Original document content"]
        metadatas = [{"source_file": "original.txt", "version": 1}]
        ids = vector_db.add_documents(documents, metadatas)
        
        doc_id = ids[0]
        
        # 문서 업데이트
        new_content = "Updated document content"
        new_metadata = {"source_file": "updated.txt", "version": 2}
        
        vector_db.update_document(
            document_id=doc_id,
            document=new_content,
            metadata=new_metadata
        )
        
        # 업데이트 확인을 위한 검색
        results = vector_db.search_similar_documents("updated", n_results=1)
        
        if results:
            assert "updated" in results[0].content.lower()
            assert results[0].metadata.get("version") == 2
    
    def test_update_document_validation_error(self, vector_db):
        """문서 업데이트 시 유효성 검사 오류 테스트"""
        with pytest.raises(ValueError, match="문서 ID가 필요합니다"):
            vector_db.update_document("", document="test")
    
    def test_delete_documents(self, vector_db):
        """문서 삭제 테스트"""
        # 문서 추가
        documents = ["Document to be deleted", "Document to keep"]
        metadatas = [{"source_file": "delete.txt"}, {"source_file": "keep.txt"}]
        ids = vector_db.add_documents(documents, metadatas)
        
        initial_count = vector_db.get_document_count()
        assert initial_count == 2
        
        # 첫 번째 문서 삭제
        vector_db.delete_documents([ids[0]])
        
        final_count = vector_db.get_document_count()
        assert final_count == 1
    
    def test_delete_documents_validation_error(self, vector_db):
        """문서 삭제 시 유효성 검사 오류 테스트"""
        with pytest.raises(ValueError, match="삭제할 문서 ID가 없습니다"):
            vector_db.delete_documents([])
    
    def test_get_document_count(self, vector_db):
        """문서 수 조회 테스트"""
        assert vector_db.get_document_count() == 0
        
        # 문서 추가
        documents = ["Doc 1", "Doc 2", "Doc 3"]
        metadatas = [{"id": i} for i in range(3)]
        vector_db.add_documents(documents, metadatas)
        
        assert vector_db.get_document_count() == 3
    
    def test_clear_collection(self, vector_db):
        """컬렉션 전체 삭제 테스트"""
        # 문서 추가
        documents = ["Doc 1", "Doc 2"]
        metadatas = [{"id": 1}, {"id": 2}]
        vector_db.add_documents(documents, metadatas)
        
        assert vector_db.get_document_count() == 2
        
        # 컬렉션 삭제
        vector_db.clear_collection()
        
        assert vector_db.get_document_count() == 0
    
    def test_get_collection_info(self, vector_db):
        """컬렉션 정보 조회 테스트"""
        info = vector_db.get_collection_info()
        
        assert isinstance(info, dict)
        assert "collection_name" in info
        assert "document_count" in info
        assert "persist_directory" in info
        assert "embedding_function" in info
        
        assert info["collection_name"] == "test_collection"
        assert info["document_count"] == 0
    
    def test_search_result_conversion(self, vector_db):
        """SearchResult 객체 변환 테스트"""
        # 문서 추가
        documents = ["Test document for search result conversion"]
        metadatas = [{"source_file": "test.txt", "extra_info": "test_value"}]
        vector_db.add_documents(documents, metadatas)
        
        # 검색 수행
        results = vector_db.search_similar_documents("test document", n_results=1)
        
        assert len(results) == 1
        result = results[0]
        
        # SearchResult 객체 검증
        assert isinstance(result, SearchResult)
        assert result.content == documents[0]
        assert result.source_file == "test.txt"
        assert 0.0 <= result.relevance_score <= 1.0
        assert result.metadata["extra_info"] == "test_value"
    
    def test_relevance_score_calculation(self, vector_db):
        """유사도 점수 계산 테스트"""
        # 동일한 문서 추가
        documents = ["machine learning artificial intelligence"]
        metadatas = [{"source_file": "test.txt"}]
        vector_db.add_documents(documents, metadatas)
        
        # 동일한 쿼리로 검색 (높은 유사도 기대)
        results = vector_db.search_similar_documents("machine learning artificial intelligence", n_results=1)
        
        assert len(results) == 1
        # 동일한 텍스트이므로 높은 유사도 점수를 가져야 함
        assert results[0].relevance_score > 0.5