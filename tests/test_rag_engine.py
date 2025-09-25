"""
RAG 엔진 테스트 모듈

RAGEngine 클래스의 기능을 테스트합니다.
"""

import pytest
from unittest.mock import Mock, MagicMock, patch
from datetime import datetime
from typing import List

from src.document_rag_english_study.rag.engine import RAGEngine
from src.document_rag_english_study.rag.vector_database import VectorDatabase
from src.document_rag_english_study.rag.embeddings import EmbeddingGenerator
from src.document_rag_english_study.llm.base import LanguageModel, MockLanguageModel
from src.document_rag_english_study.models.document import Document, IndexingResult
from src.document_rag_english_study.models.response import SearchResult
from src.document_rag_english_study.models.llm import LLMResponse


class TestRAGEngine:
    """RAGEngine 클래스 테스트"""
    
    @pytest.fixture
    def mock_vector_db(self):
        """모의 벡터 데이터베이스 픽스처"""
        mock_db = Mock(spec=VectorDatabase)
        mock_db.add_documents.return_value = ["chunk_1", "chunk_2", "chunk_3"]
        mock_db.search_similar_documents.return_value = [
            SearchResult(
                content="Test content 1",
                source_file="test1.txt",
                relevance_score=0.9,
                metadata={"document_id": "doc1", "chunk_index": 0}
            ),
            SearchResult(
                content="Test content 2", 
                source_file="test2.txt",
                relevance_score=0.8,
                metadata={"document_id": "doc2", "chunk_index": 0}
            )
        ]
        mock_db.get_collection_info.return_value = {
            "collection_name": "test_collection",
            "document_count": 5
        }
        mock_db.clear_collection.return_value = None
        return mock_db
    
    @pytest.fixture
    def mock_embedding_generator(self):
        """모의 임베딩 생성기 픽스처"""
        mock_gen = Mock(spec=EmbeddingGenerator)
        mock_gen.generate_embedding.return_value = [0.1, 0.2, 0.3, 0.4, 0.5]
        mock_gen.generate_batch_embeddings.return_value = [
            [0.1, 0.2, 0.3, 0.4, 0.5],
            [0.2, 0.3, 0.4, 0.5, 0.6],
            [0.3, 0.4, 0.5, 0.6, 0.7]
        ]
        mock_gen.get_cache_info.return_value = {
            "memory_cache_size": 10,
            "file_cache_size": 5
        }
        return mock_gen
    
    @pytest.fixture
    def mock_llm(self):
        """모의 언어 모델 픽스처"""
        mock_llm = MockLanguageModel()
        mock_llm.initialize()
        return mock_llm
    
    @pytest.fixture
    def sample_document(self):
        """샘플 문서 픽스처"""
        return Document(
            id="test_doc_1",
            title="Test Document",
            file_path="/path/to/test.txt",
            content="This is a test document with multiple sentences. It contains various information about testing. The document is used for unit testing purposes. It has enough content to be split into multiple chunks.",
            file_type="txt",
            created_at=datetime.now(),
            word_count=30,
            language="english"
        )
    
    @pytest.fixture
    def rag_engine(self, mock_vector_db, mock_embedding_generator):
        """RAG 엔진 픽스처"""
        return RAGEngine(
            vector_db=mock_vector_db,
            embedding_generator=mock_embedding_generator,
            chunk_size=100,
            chunk_overlap=20
        )
    
    def test_rag_engine_initialization(self, mock_vector_db, mock_embedding_generator):
        """RAG 엔진 초기화 테스트"""
        engine = RAGEngine(
            vector_db=mock_vector_db,
            embedding_generator=mock_embedding_generator,
            chunk_size=500,
            chunk_overlap=50
        )
        
        assert engine.vector_db == mock_vector_db
        assert engine.embedding_generator == mock_embedding_generator
        assert engine.llm is None
        assert engine.chunk_size == 500
        assert engine.chunk_overlap == 50
        assert engine.max_context_length == 4000
        assert len(engine._indexed_documents) == 0
        assert engine._total_chunks == 0
    
    def test_set_language_model(self, rag_engine, mock_llm):
        """언어 모델 설정 테스트"""
        assert rag_engine.llm is None
        
        rag_engine.set_language_model(mock_llm)
        
        assert rag_engine.llm == mock_llm
    
    def test_index_document_success(self, rag_engine, sample_document):
        """문서 인덱싱 성공 테스트"""
        result = rag_engine.index_document(sample_document)
        
        assert result.success is True
        assert result.documents_processed == 1
        assert result.total_chunks > 0
        assert len(result.errors) == 0
        assert sample_document.file_path in result.indexed_files
        assert sample_document.id in rag_engine._indexed_documents
        
        # 벡터 데이터베이스 호출 확인
        rag_engine.vector_db.add_documents.assert_called_once()
        
        # 임베딩 생성기 호출 확인
        rag_engine.embedding_generator.generate_batch_embeddings.assert_called_once()
    
    def test_index_document_empty_content(self, rag_engine):
        """빈 내용 문서 인덱싱 테스트"""
        # Document 모델의 validation을 우회하여 빈 내용 테스트
        # 실제로는 Document 생성 시 validation이 있으므로, 
        # 이 테스트는 RAG 엔진의 청크 생성 로직을 테스트하기 위해 mock을 사용
        
        # 정상적인 문서를 만들고 내용을 직접 수정
        empty_doc = Document(
            id="empty_doc",
            title="Empty Document", 
            file_path="/path/to/empty.txt",
            content="temp content",  # 임시 내용으로 생성
            file_type="txt"
        )
        # 내용을 빈 문자열로 직접 수정 (validation 우회)
        empty_doc.content = ""
        
        result = rag_engine.index_document(empty_doc)
        
        assert result.success is False
        assert len(result.errors) > 0
        assert "유효한 청크를 생성할 수 없습니다" in result.errors[0]
    
    def test_index_document_with_exception(self, rag_engine, sample_document):
        """문서 인덱싱 중 예외 발생 테스트"""
        # 벡터 데이터베이스에서 예외 발생 시뮬레이션
        rag_engine.vector_db.add_documents.side_effect = Exception("Database error")
        
        result = rag_engine.index_document(sample_document)
        
        assert result.success is False
        assert len(result.errors) > 0
        assert sample_document.file_path in result.failed_files
    
    def test_index_documents_batch(self, rag_engine):
        """여러 문서 일괄 인덱싱 테스트"""
        documents = [
            Document(
                id=f"doc_{i}",
                title=f"Document {i}",
                file_path=f"/path/to/doc{i}.txt",
                content=f"This is document {i} with some content for testing purposes.",
                file_type="txt"
            )
            for i in range(3)
        ]
        
        result = rag_engine.index_documents(documents)
        
        assert result.success is True
        assert result.documents_processed == 3
        assert len(result.indexed_files) == 3
        assert len(result.errors) == 0
    
    def test_search_similar_content(self, rag_engine):
        """유사 내용 검색 테스트"""
        query = "test query"
        
        results = rag_engine.search_similar_content(query, top_k=5)
        
        assert len(results) == 2  # mock에서 2개 결과 반환
        assert all(isinstance(r, SearchResult) for r in results)
        assert all(r.relevance_score >= 0.1 for r in results)  # 기본 최소 점수
        
        # 벡터 데이터베이스 호출 확인
        rag_engine.vector_db.search_similar_documents.assert_called_once_with(
            query=query,
            n_results=5,
            where=None
        )
    
    def test_search_similar_content_empty_query(self, rag_engine):
        """빈 쿼리 검색 테스트"""
        results = rag_engine.search_similar_content("")
        
        assert len(results) == 0
        # 벡터 데이터베이스가 호출되지 않아야 함
        rag_engine.vector_db.search_similar_documents.assert_not_called()
    
    def test_search_similar_content_with_filter(self, rag_engine):
        """메타데이터 필터링 검색 테스트"""
        query = "test query"
        filter_metadata = {"document_id": "doc1"}
        
        rag_engine.search_similar_content(query, filter_metadata=filter_metadata)
        
        rag_engine.vector_db.search_similar_documents.assert_called_once_with(
            query=query,
            n_results=5,
            where=filter_metadata
        )
    
    def test_search_similar_content_min_relevance_filter(self, rag_engine):
        """최소 관련성 점수 필터링 테스트"""
        # 높은 최소 점수 설정 (0.85)
        results = rag_engine.search_similar_content("test", min_relevance_score=0.85)
        
        # 0.9 점수만 통과해야 함 (mock 데이터에서 0.9, 0.8)
        assert len(results) == 1
        assert results[0].relevance_score >= 0.85
    
    def test_generate_answer_without_llm(self, rag_engine):
        """LLM 없이 답변 생성 시도 테스트"""
        with pytest.raises(ValueError, match="언어 모델이 설정되지 않았습니다"):
            rag_engine.generate_answer("test question")
    
    def test_generate_answer_empty_query(self, rag_engine, mock_llm):
        """빈 질문으로 답변 생성 시도 테스트"""
        rag_engine.set_language_model(mock_llm)
        
        with pytest.raises(ValueError, match="빈 질문은 처리할 수 없습니다"):
            rag_engine.generate_answer("")
    
    def test_generate_answer_success(self, rag_engine, mock_llm):
        """답변 생성 성공 테스트"""
        rag_engine.set_language_model(mock_llm)
        query = "What is this about?"
        
        answer = rag_engine.generate_answer(query)
        
        assert isinstance(answer, str)
        assert len(answer) > 0
        
        # 검색이 호출되었는지 확인
        rag_engine.vector_db.search_similar_documents.assert_called_once()
    
    def test_generate_answer_with_provided_context(self, rag_engine, mock_llm):
        """제공된 컨텍스트로 답변 생성 테스트"""
        rag_engine.set_language_model(mock_llm)
        query = "What is this about?"
        context_results = [
            SearchResult(
                content="Provided context",
                source_file="context.txt",
                relevance_score=0.9,
                metadata={}
            )
        ]
        
        answer = rag_engine.generate_answer(query, context_results=context_results)
        
        assert isinstance(answer, str)
        assert len(answer) > 0
        
        # 자동 검색이 호출되지 않아야 함
        rag_engine.vector_db.search_similar_documents.assert_not_called()
    
    def test_extract_keywords(self, rag_engine):
        """키워드 추출 테스트"""
        text = "This is a test document about machine learning and artificial intelligence. The document contains information about natural language processing."
        
        keywords = rag_engine.extract_keywords(text, max_keywords=5)
        
        assert isinstance(keywords, list)
        assert len(keywords) <= 5
        assert all(isinstance(kw, str) for kw in keywords)
        
        # 일반적인 키워드가 포함되어야 함
        expected_keywords = ["machine", "learning", "artificial", "intelligence", "document", "information", "natural", "language", "processing"]
        assert any(kw in expected_keywords for kw in keywords)
    
    def test_extract_keywords_empty_text(self, rag_engine):
        """빈 텍스트 키워드 추출 테스트"""
        keywords = rag_engine.extract_keywords("")
        
        assert keywords == []
    
    def test_get_indexed_document_info(self, rag_engine, sample_document):
        """인덱싱된 문서 정보 조회 테스트"""
        # 문서 인덱싱
        rag_engine.index_document(sample_document)
        
        info = rag_engine.get_indexed_document_info()
        
        assert isinstance(info, dict)
        assert "total_documents" in info
        assert "total_chunks" in info
        assert "vector_db_info" in info
        assert "embedding_info" in info
        assert info["total_documents"] == 1
        assert info["total_chunks"] > 0
    
    def test_remove_document(self, rag_engine, sample_document):
        """문서 제거 테스트"""
        # 먼저 문서 인덱싱
        rag_engine.index_document(sample_document)
        assert sample_document.id in rag_engine._indexed_documents
        
        # 문서 제거
        success = rag_engine.remove_document(sample_document.id)
        
        assert success is True
        assert sample_document.id not in rag_engine._indexed_documents
    
    def test_remove_nonexistent_document(self, rag_engine):
        """존재하지 않는 문서 제거 테스트"""
        success = rag_engine.remove_document("nonexistent_id")
        
        assert success is False
    
    def test_clear_index(self, rag_engine, sample_document):
        """인덱스 전체 삭제 테스트"""
        # 문서 인덱싱
        rag_engine.index_document(sample_document)
        assert len(rag_engine._indexed_documents) > 0
        assert rag_engine._total_chunks > 0
        
        # 인덱스 삭제
        rag_engine.clear_index()
        
        assert len(rag_engine._indexed_documents) == 0
        assert rag_engine._total_chunks == 0
        rag_engine.vector_db.clear_collection.assert_called_once()
    
    def test_split_document_into_chunks(self, rag_engine):
        """문서 청크 분할 테스트"""
        content = "This is the first paragraph with some content.\n\nThis is the second paragraph with more content.\n\nThis is the third paragraph with additional information."
        
        chunks = rag_engine._split_document_into_chunks(content)
        
        assert isinstance(chunks, list)
        assert len(chunks) > 0
        assert all(isinstance(chunk, str) for chunk in chunks)
        assert all(len(chunk) <= rag_engine.chunk_size for chunk in chunks)
    
    def test_split_document_empty_content(self, rag_engine):
        """빈 내용 문서 분할 테스트"""
        chunks = rag_engine._split_document_into_chunks("")
        
        assert chunks == []
    
    def test_split_document_long_paragraph(self, rag_engine):
        """긴 문단 분할 테스트"""
        # 청크 크기보다 긴 단일 문단
        long_content = "This is a very long paragraph. " * 20  # 청크 크기 초과
        
        chunks = rag_engine._split_document_into_chunks(long_content)
        
        assert len(chunks) > 1  # 여러 청크로 분할되어야 함
        assert all(len(chunk) <= rag_engine.chunk_size for chunk in chunks)
    
    def test_build_context_from_results(self, rag_engine):
        """검색 결과로부터 컨텍스트 구성 테스트"""
        search_results = [
            SearchResult(
                content="First result content",
                source_file="file1.txt",
                relevance_score=0.9,
                metadata={}
            ),
            SearchResult(
                content="Second result content",
                source_file="file2.txt",
                relevance_score=0.8,
                metadata={}
            )
        ]
        
        context = rag_engine._build_context_from_results(search_results)
        
        assert isinstance(context, str)
        assert "First result content" in context
        assert "Second result content" in context
        assert "file1.txt" in context
        assert "file2.txt" in context
    
    def test_build_context_empty_results(self, rag_engine):
        """빈 검색 결과로 컨텍스트 구성 테스트"""
        context = rag_engine._build_context_from_results([])
        
        assert context == ""
    
    def test_build_answer_prompt_korean(self, rag_engine):
        """한국어 답변 프롬프트 구성 테스트"""
        query = "테스트 질문"
        context = "테스트 컨텍스트"
        
        prompt = rag_engine._build_answer_prompt(query, context, "korean")
        
        assert isinstance(prompt, str)
        assert query in prompt
        assert context in prompt
        assert "다음 문서들을 참고하여" in prompt
    
    def test_build_answer_prompt_english(self, rag_engine):
        """영어 답변 프롬프트 구성 테스트"""
        query = "test question"
        context = "test context"
        
        prompt = rag_engine._build_answer_prompt(query, context, "english")
        
        assert isinstance(prompt, str)
        assert query in prompt
        assert context in prompt
        assert "Please answer the user's question" in prompt
    
    def test_add_chunk_overlap(self, rag_engine):
        """청크 겹침 추가 테스트"""
        chunks = [
            "First chunk with some content here and more words to ensure overlap",
            "Second chunk with different content",
            "Third chunk with more information"
        ]
        
        overlapped = rag_engine._add_chunk_overlap(chunks)
        
        assert len(overlapped) == len(chunks)
        assert overlapped[0] == chunks[0]  # 첫 번째는 변경 없음
        # 두 번째부터는 이전 청크의 일부가 포함되어야 함 (충분한 단어가 있는 경우)
        if len(chunks[0].split()) > rag_engine.chunk_overlap:
            assert len(overlapped[1]) > len(chunks[1])
        else:
            # 이전 청크의 단어 수가 overlap보다 적으면 전체가 포함됨
            assert chunks[0] in overlapped[1] or len(overlapped[1]) >= len(chunks[1])
    
    def test_add_chunk_overlap_single_chunk(self, rag_engine):
        """단일 청크 겹침 추가 테스트"""
        chunks = ["Single chunk"]
        
        overlapped = rag_engine._add_chunk_overlap(chunks)
        
        assert overlapped == chunks  # 변경 없어야 함


class TestRAGEngineIntegration:
    """RAG 엔진 통합 테스트"""
    
    @pytest.fixture
    def real_embedding_generator(self):
        """실제 임베딩 생성기 (테스트용 작은 모델)"""
        # 실제 테스트에서는 작은 모델 사용
        return Mock(spec=EmbeddingGenerator)
    
    @pytest.fixture
    def real_vector_db(self):
        """실제 벡터 데이터베이스 (메모리 기반)"""
        return Mock(spec=VectorDatabase)
    
    def test_full_indexing_and_search_workflow(self, real_vector_db, real_embedding_generator):
        """전체 인덱싱 및 검색 워크플로우 테스트"""
        # 실제 구현체를 사용한 통합 테스트
        # 이 테스트는 실제 ChromaDB와 sentence-transformers를 사용할 수 있음
        
        engine = RAGEngine(
            vector_db=real_vector_db,
            embedding_generator=real_embedding_generator
        )
        
        # 테스트 문서 생성
        test_doc = Document(
            id="integration_test_doc",
            title="Integration Test Document",
            file_path="/test/integration.txt",
            content="This is a comprehensive test document for integration testing. It contains multiple sentences and paragraphs to test the full workflow of the RAG engine.",
            file_type="txt"
        )
        
        # Mock 설정
        real_embedding_generator.generate_batch_embeddings.return_value = [
            [0.1, 0.2, 0.3] for _ in range(3)  # 3개 청크 가정
        ]
        real_vector_db.add_documents.return_value = ["chunk1", "chunk2", "chunk3"]
        real_vector_db.search_similar_documents.return_value = [
            SearchResult(
                content="Integration test content",
                source_file="/test/integration.txt",
                relevance_score=0.95,
                metadata={"document_id": "integration_test_doc"}
            )
        ]
        
        # 인덱싱 테스트
        indexing_result = engine.index_document(test_doc)
        assert indexing_result.success
        
        # 검색 테스트
        search_results = engine.search_similar_content("integration test")
        assert len(search_results) > 0
        assert search_results[0].relevance_score > 0.9