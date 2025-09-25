"""
Unit tests for the DocumentManager class.

This module contains comprehensive tests for document management functionality
including directory scanning, document indexing, and metadata management.
"""

import pytest
import tempfile
import os
import json
import time
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime

from src.document_rag_english_study.document_manager.manager import (
    DocumentManager,
    DocumentManagerError
)
from src.document_rag_english_study.document_manager.parser import DocumentParser
from src.document_rag_english_study.models.document import (
    Document,
    IndexingResult,
    IndexingStatus,
    DocumentSummary
)


class TestDocumentManager:
    """DocumentManager 클래스에 대한 테스트 케이스."""
    
    def setup_method(self):
        """각 테스트 메서드 실행 전 설정."""
        # 임시 디렉토리 생성
        self.temp_dir = tempfile.mkdtemp()
        self.manager = DocumentManager(storage_path=self.temp_dir)
    
    def teardown_method(self):
        """각 테스트 메서드 실행 후 정리."""
        # 임시 디렉토리 정리
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_init(self):
        """DocumentManager 초기화 테스트."""
        manager = DocumentManager()
        assert manager is not None
        assert isinstance(manager.parser, DocumentParser)
        assert manager.storage_path.exists()
        assert manager._documents_cache == {}
        assert not manager._indexing_status.is_indexing
    
    def test_init_with_storage_path(self):
        """저장 경로를 지정한 DocumentManager 초기화 테스트."""
        storage_path = self.temp_dir + "/custom_storage"
        manager = DocumentManager(storage_path=storage_path)
        assert str(manager.storage_path) == storage_path
        assert manager.storage_path.exists()
    
    def test_set_progress_callback(self):
        """진행률 콜백 설정 테스트."""
        callback = Mock()
        self.manager.set_progress_callback(callback)
        assert self.manager._progress_callback == callback
    
    def test_set_document_directory_invalid_path(self):
        """존재하지 않는 디렉토리 설정 테스트."""
        with pytest.raises(DocumentManagerError, match="Directory does not exist"):
            self.manager.set_document_directory("/nonexistent/directory")
    
    def test_set_document_directory_not_directory(self):
        """파일을 디렉토리로 설정하는 테스트."""
        # 임시 파일 생성
        temp_file = os.path.join(self.temp_dir, "test_file.txt")
        with open(temp_file, 'w') as f:
            f.write("test content")
        
        with pytest.raises(DocumentManagerError, match="Path is not a directory"):
            self.manager.set_document_directory(temp_file)
    
    @patch.object(DocumentManager, 'index_documents')
    def test_set_document_directory_success(self, mock_index):
        """유효한 디렉토리 설정 성공 테스트."""
        # Mock 설정
        mock_result = IndexingResult(success=True, documents_processed=2)
        mock_index.return_value = mock_result
        
        # 테스트 디렉토리 생성
        test_dir = os.path.join(self.temp_dir, "test_docs")
        os.makedirs(test_dir)
        
        result = self.manager.set_document_directory(test_dir)
        
        assert result.success is True
        assert result.documents_processed == 2
        mock_index.assert_called_once_with(test_dir)
    
    def test_scan_directory_empty(self):
        """빈 디렉토리 스캔 테스트."""
        empty_dir = os.path.join(self.temp_dir, "empty")
        os.makedirs(empty_dir)
        
        files = self.manager._scan_directory(empty_dir)
        assert files == []
    
    def test_scan_directory_with_supported_files(self):
        """지원되는 파일들이 있는 디렉토리 스캔 테스트."""
        test_dir = os.path.join(self.temp_dir, "test_docs")
        os.makedirs(test_dir)
        
        # 지원되는 파일들 생성
        supported_files = ["test1.txt", "test2.pdf", "test3.docx", "test4.md"]
        for filename in supported_files:
            filepath = os.path.join(test_dir, filename)
            with open(filepath, 'w') as f:
                f.write("test content")
        
        # 지원되지 않는 파일도 생성
        unsupported_file = os.path.join(test_dir, "test.xlsx")
        with open(unsupported_file, 'w') as f:
            f.write("unsupported content")
        
        files = self.manager._scan_directory(test_dir)
        
        # 지원되는 파일들만 반환되어야 함
        assert len(files) == 4
        for file_path in files:
            assert any(file_path.endswith(ext) for ext in ['.txt', '.pdf', '.docx', '.md'])
    
    def test_scan_directory_recursive(self):
        """재귀적 디렉토리 스캔 테스트."""
        test_dir = os.path.join(self.temp_dir, "test_docs")
        sub_dir = os.path.join(test_dir, "subdirectory")
        os.makedirs(sub_dir)
        
        # 루트 디렉토리에 파일 생성
        root_file = os.path.join(test_dir, "root.txt")
        with open(root_file, 'w') as f:
            f.write("root content")
        
        # 서브 디렉토리에 파일 생성
        sub_file = os.path.join(sub_dir, "sub.md")
        with open(sub_file, 'w') as f:
            f.write("sub content")
        
        files = self.manager._scan_directory(test_dir)
        
        assert len(files) == 2
        assert any("root.txt" in f for f in files)
        assert any("sub.md" in f for f in files)
    
    @patch.object(DocumentParser, 'parse_file')
    def test_index_single_document_success(self, mock_parse):
        """단일 문서 인덱싱 성공 테스트."""
        # Mock 설정
        test_doc = Document(
            id="test_doc_1",
            title="Test Document",
            file_path="/test/path.txt",
            content="Test content",
            file_type="txt"
        )
        mock_parse.return_value = test_doc
        
        result = self.manager._index_single_document("/test/path.txt")
        
        assert result == test_doc
        mock_parse.assert_called_once_with("/test/path.txt")
    
    @patch.object(DocumentParser, 'parse_file')
    def test_index_single_document_failure(self, mock_parse):
        """단일 문서 인덱싱 실패 테스트."""
        # Mock이 예외를 발생시키도록 설정
        mock_parse.side_effect = Exception("Parsing failed")
        
        result = self.manager._index_single_document("/test/path.txt")
        
        assert result is None
    
    def test_get_existing_document_not_found(self):
        """기존 문서 없음 테스트."""
        result = self.manager._get_existing_document("/nonexistent/path.txt")
        assert result is None
    
    def test_add_document_to_cache(self):
        """문서 캐시 추가 테스트."""
        test_doc = Document(
            id="test_doc_1",
            title="Test Document",
            file_path="/test/path.txt",
            content="Test content",
            file_type="txt"
        )
        
        self.manager._add_document_to_cache(test_doc)
        
        assert "test_doc_1" in self.manager._documents_cache
        assert self.manager._documents_cache["test_doc_1"] == test_doc
    
    def test_get_indexing_status(self):
        """인덱싱 상태 조회 테스트."""
        status = self.manager.get_indexing_status()
        
        assert isinstance(status, IndexingStatus)
        assert status.is_indexing is False
        assert status.total_documents == 0
        assert status.processed_documents == 0
    
    def test_get_document_summary_empty(self):
        """빈 문서 요약 테스트."""
        summary = self.manager.get_document_summary()
        
        assert isinstance(summary, DocumentSummary)
        assert summary.total_documents == 0
        assert summary.total_words == 0
        assert summary.file_types == {}
    
    def test_get_document_summary_with_documents(self):
        """문서가 있는 경우 요약 테스트."""
        # 테스트 문서들 추가
        doc1 = Document(
            id="doc1",
            title="Document 1",
            file_path="/test/doc1.txt",
            content="This is test content",
            file_type="txt",
            word_count=4
        )
        doc2 = Document(
            id="doc2",
            title="Document 2",
            file_path="/test/doc2.pdf",
            content="Another test document",
            file_type="pdf",
            word_count=3
        )
        
        self.manager._add_document_to_cache(doc1)
        self.manager._add_document_to_cache(doc2)
        
        summary = self.manager.get_document_summary()
        
        assert summary.total_documents == 2
        assert summary.total_words == 7
        assert summary.file_types == {"txt": 1, "pdf": 1}
        assert summary.get_average_words_per_document() == 3.5
    
    def test_get_document_by_id(self):
        """ID로 문서 조회 테스트."""
        test_doc = Document(
            id="test_doc_1",
            title="Test Document",
            file_path="/test/path.txt",
            content="Test content",
            file_type="txt"
        )
        
        self.manager._add_document_to_cache(test_doc)
        
        # 존재하는 문서 조회
        result = self.manager.get_document_by_id("test_doc_1")
        assert result == test_doc
        
        # 존재하지 않는 문서 조회
        result = self.manager.get_document_by_id("nonexistent")
        assert result is None
    
    def test_get_all_documents(self):
        """모든 문서 조회 테스트."""
        # 테스트 문서들 추가
        doc1 = Document(
            id="doc1",
            title="Document 1",
            file_path="/test/doc1.txt",
            content="Content 1",
            file_type="txt"
        )
        doc2 = Document(
            id="doc2",
            title="Document 2",
            file_path="/test/doc2.pdf",
            content="Content 2",
            file_type="pdf"
        )
        
        self.manager._add_document_to_cache(doc1)
        self.manager._add_document_to_cache(doc2)
        
        all_docs = self.manager.get_all_documents()
        
        assert len(all_docs) == 2
        assert doc1 in all_docs
        assert doc2 in all_docs
    
    def test_search_documents(self):
        """문서 검색 테스트."""
        # 테스트 문서들 추가
        doc1 = Document(
            id="doc1",
            title="Python Programming",
            file_path="/test/python.txt",
            content="Learn Python programming language",
            file_type="txt"
        )
        doc2 = Document(
            id="doc2",
            title="Java Tutorial",
            file_path="/test/java.pdf",
            content="Java programming tutorial",
            file_type="pdf"
        )
        doc3 = Document(
            id="doc3",
            title="Web Development",
            file_path="/test/web.md",
            content="HTML, CSS, JavaScript for web development",
            file_type="md"
        )
        
        self.manager._add_document_to_cache(doc1)
        self.manager._add_document_to_cache(doc2)
        self.manager._add_document_to_cache(doc3)
        
        # 제목에서 검색
        results = self.manager.search_documents("Python")
        assert len(results) == 1
        assert results[0] == doc1
        
        # 내용에서 검색
        results = self.manager.search_documents("programming")
        assert len(results) == 2
        assert doc1 in results
        assert doc2 in results
        
        # 대소문자 구분 없는 검색
        results = self.manager.search_documents("JAVA")
        assert len(results) == 1
        assert results[0] == doc2
        
        # 검색 결과 제한
        results = self.manager.search_documents("development", limit=1)
        assert len(results) == 1
    
    def test_remove_document(self):
        """문서 제거 테스트."""
        test_doc = Document(
            id="test_doc_1",
            title="Test Document",
            file_path="/test/path.txt",
            content="Test content",
            file_type="txt"
        )
        
        self.manager._add_document_to_cache(test_doc)
        assert "test_doc_1" in self.manager._documents_cache
        
        # 문서 제거
        result = self.manager.remove_document("test_doc_1")
        assert result is True
        assert "test_doc_1" not in self.manager._documents_cache
        
        # 존재하지 않는 문서 제거
        result = self.manager.remove_document("nonexistent")
        assert result is False
    
    def test_clear_all_documents(self):
        """모든 문서 제거 테스트."""
        # 테스트 문서들 추가
        doc1 = Document(
            id="doc1",
            title="Document 1",
            file_path="/test/doc1.txt",
            content="Content 1",
            file_type="txt"
        )
        doc2 = Document(
            id="doc2",
            title="Document 2",
            file_path="/test/doc2.pdf",
            content="Content 2",
            file_type="pdf"
        )
        
        self.manager._add_document_to_cache(doc1)
        self.manager._add_document_to_cache(doc2)
        
        assert len(self.manager._documents_cache) == 2
        
        # 모든 문서 제거
        result = self.manager.clear_all_documents()
        assert result is True
        assert len(self.manager._documents_cache) == 0
    
    def test_save_and_load_metadata(self):
        """메타데이터 저장 및 로드 테스트."""
        # 테스트 문서 추가
        test_doc = Document(
            id="test_doc_1",
            title="Test Document",
            file_path="/test/path.txt",
            content="Test content",
            file_type="txt",
            word_count=2
        )
        
        self.manager._add_document_to_cache(test_doc)
        
        # 메타데이터 저장
        self.manager._save_metadata()
        
        # 메타데이터 파일 존재 확인
        assert self.manager.metadata_file.exists()
        
        # 새로운 매니저 인스턴스로 메타데이터 로드 테스트
        new_manager = DocumentManager(storage_path=self.temp_dir)
        
        # 문서가 로드되었는지 확인
        assert len(new_manager._documents_cache) == 1
        loaded_doc = new_manager.get_document_by_id("test_doc_1")
        assert loaded_doc is not None
        assert loaded_doc.title == "Test Document"
        assert loaded_doc.content == "Test content"
        assert loaded_doc.file_type == "txt"
        assert loaded_doc.word_count == 2
    
    def test_get_manager_info(self):
        """관리자 정보 조회 테스트."""
        info = self.manager.get_manager_info()
        
        assert isinstance(info, dict)
        assert 'storage_path' in info
        assert 'metadata_file' in info
        assert 'document_count' in info
        assert 'is_indexing' in info
        assert 'parser_info' in info
        
        assert info['document_count'] == 0
        assert info['is_indexing'] is False
        assert isinstance(info['parser_info'], dict)
    
    def test_notify_progress_with_callback(self):
        """진행률 콜백 호출 테스트."""
        callback = Mock()
        self.manager.set_progress_callback(callback)
        
        # 진행률 알림 호출
        self.manager._notify_progress()
        
        callback.assert_called_once()
        # 콜백에 전달된 인자가 IndexingStatus 인스턴스인지 확인
        args, kwargs = callback.call_args
        assert isinstance(args[0], IndexingStatus)
    
    def test_notify_progress_without_callback(self):
        """콜백 없이 진행률 알림 테스트."""
        # 콜백이 설정되지 않은 상태에서 호출해도 오류가 발생하지 않아야 함
        self.manager._notify_progress()
        # 예외가 발생하지 않으면 성공


class TestDocumentManagerIntegration:
    """DocumentManager 통합 테스트."""
    
    def setup_method(self):
        """각 테스트 메서드 실행 전 설정."""
        self.temp_dir = tempfile.mkdtemp()
        self.manager = DocumentManager(storage_path=self.temp_dir)
    
    def teardown_method(self):
        """각 테스트 메서드 실행 후 정리."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_full_indexing_workflow(self):
        """전체 인덱싱 워크플로우 통합 테스트."""
        # 테스트 문서 디렉토리 생성
        docs_dir = os.path.join(self.temp_dir, "test_documents")
        os.makedirs(docs_dir)
        
        # 테스트 문서들 생성
        test_files = {
            "document1.txt": "This is the first test document with some content.",
            "document2.md": "# Second Document\nThis is a markdown document with **bold** text.",
            "document3.txt": "Third document contains different content for testing."
        }
        
        for filename, content in test_files.items():
            filepath = os.path.join(docs_dir, filename)
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(content)
        
        # 진행률 콜백 설정
        progress_updates = []
        def progress_callback(status):
            progress_updates.append(status.get_progress_percentage())
        
        self.manager.set_progress_callback(progress_callback)
        
        # 문서 디렉토리 설정 및 인덱싱
        result = self.manager.set_document_directory(docs_dir)
        
        # 결과 검증
        assert result.success is True
        assert result.documents_processed == 3
        assert len(result.indexed_files) == 3
        assert len(result.failed_files) == 0
        assert result.processing_time > 0
        
        # 진행률 업데이트 확인
        assert len(progress_updates) > 0
        assert progress_updates[-1] == 100.0  # 마지막 업데이트는 100%
        
        # 인덱싱된 문서들 확인
        all_docs = self.manager.get_all_documents()
        assert len(all_docs) == 3
        
        # 문서 요약 확인
        summary = self.manager.get_document_summary()
        assert summary.total_documents == 3
        assert summary.file_types['txt'] == 2
        assert summary.file_types['md'] == 1
        assert summary.total_words > 0
        
        # 검색 기능 확인
        search_results = self.manager.search_documents("test")
        assert len(search_results) >= 2  # "test"가 포함된 문서들
        
        # 메타데이터 저장 확인
        assert self.manager.metadata_file.exists()
        
        # 새로운 매니저로 메타데이터 로드 확인
        new_manager = DocumentManager(storage_path=self.temp_dir)
        loaded_docs = new_manager.get_all_documents()
        assert len(loaded_docs) == 3
    
    @patch('src.document_rag_english_study.document_manager.parser.DocumentParser.parse_file')
    def test_indexing_with_failures(self, mock_parse):
        """일부 파일 인덱싱 실패 시나리오 테스트."""
        # 테스트 문서 디렉토리 생성
        docs_dir = os.path.join(self.temp_dir, "test_documents")
        os.makedirs(docs_dir)
        
        # 테스트 파일들 생성
        for i in range(3):
            filepath = os.path.join(docs_dir, f"document{i}.txt")
            with open(filepath, 'w') as f:
                f.write(f"Content of document {i}")
        
        # Mock 설정: 첫 번째 파일은 성공, 두 번째는 실패, 세 번째는 성공
        def mock_parse_side_effect(file_path):
            if "document1.txt" in file_path:
                raise Exception("Parsing failed for document1")
            else:
                return Document(
                    id=f"doc_{Path(file_path).stem}",
                    title=f"Document {Path(file_path).stem}",
                    file_path=file_path,
                    content=f"Content of {Path(file_path).stem}",
                    file_type="txt"
                )
        
        mock_parse.side_effect = mock_parse_side_effect
        
        # 인덱싱 수행
        result = self.manager.index_documents(docs_dir)
        
        # 결과 검증
        assert result.success is False  # 일부 실패로 인해 전체적으로는 실패
        assert result.documents_processed == 2  # 2개 성공
        assert len(result.failed_files) == 1  # 1개 실패
        assert len(result.errors) >= 1
        
        # 성공한 문서들은 캐시에 있어야 함
        all_docs = self.manager.get_all_documents()
        assert len(all_docs) == 2


class TestDocumentManagerError:
    """DocumentManagerError 예외 클래스 테스트."""
    
    def test_document_manager_error(self):
        """DocumentManagerError 예외 테스트."""
        error_msg = "Test manager error"
        
        with pytest.raises(DocumentManagerError) as exc_info:
            raise DocumentManagerError(error_msg)
        
        assert str(exc_info.value) == error_msg
        assert isinstance(exc_info.value, Exception)