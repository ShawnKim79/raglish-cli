"""
Unit tests for the DocumentParser class.

This module contains comprehensive tests for document parsing functionality
including PDF, DOCX, TXT, and Markdown file parsing.
"""

import pytest
import tempfile
import os
from pathlib import Path
from unittest.mock import Mock, patch, mock_open
from datetime import datetime

from src.document_rag_english_study.document_manager.parser import (
    DocumentParser,
    DocumentParsingError
)
from src.document_rag_english_study.models.document import Document


class TestDocumentParser:
    """DocumentParser 클래스에 대한 테스트 케이스."""
    
    def setup_method(self):
        """각 테스트 메서드 실행 전 설정."""
        self.parser = DocumentParser()
    
    def test_init(self):
        """DocumentParser 초기화 테스트."""
        parser = DocumentParser()
        assert parser is not None
        assert hasattr(parser, 'SUPPORTED_EXTENSIONS')
        assert len(parser.SUPPORTED_EXTENSIONS) == 4
    
    def test_is_supported_file(self):
        """지원되는 파일 형식 확인 테스트."""
        # 지원되는 파일들
        assert self.parser.is_supported_file("test.pdf") is True
        assert self.parser.is_supported_file("test.docx") is True
        assert self.parser.is_supported_file("test.txt") is True
        assert self.parser.is_supported_file("test.md") is True
        
        # 대소문자 구분 없음
        assert self.parser.is_supported_file("test.PDF") is True
        assert self.parser.is_supported_file("test.DOCX") is True
        
        # 지원되지 않는 파일들
        assert self.parser.is_supported_file("test.doc") is False
        assert self.parser.is_supported_file("test.xlsx") is False
        assert self.parser.is_supported_file("test.pptx") is False
        assert self.parser.is_supported_file("test.jpg") is False
        
        # 잘못된 경로
        assert self.parser.is_supported_file("") is False
        assert self.parser.is_supported_file("test") is False
    
    def test_get_file_type(self):
        """파일 타입 확인 테스트."""
        assert self.parser.get_file_type("test.pdf") == "pdf"
        assert self.parser.get_file_type("test.docx") == "docx"
        assert self.parser.get_file_type("test.txt") == "txt"
        assert self.parser.get_file_type("test.md") == "md"
        
        # 대소문자 구분 없음
        assert self.parser.get_file_type("test.PDF") == "pdf"
        assert self.parser.get_file_type("test.DOCX") == "docx"
        
        # 지원되지 않는 파일
        assert self.parser.get_file_type("test.doc") is None
        assert self.parser.get_file_type("test.xlsx") is None
        assert self.parser.get_file_type("test") is None
    
    @patch('pathlib.Path.exists')
    @patch('pathlib.Path.is_file')
    @patch('pathlib.Path.stat')
    def test_validate_file(self, mock_stat, mock_is_file, mock_exists):
        """파일 검증 테스트."""
        # Mock 설정
        mock_stat.return_value.st_size = 1000
        mock_exists.return_value = True
        mock_is_file.return_value = True
        
        # 유효한 파일
        assert self.parser.validate_file("test.pdf") is True
        
        # 존재하지 않는 파일
        mock_exists.return_value = False
        assert self.parser.validate_file("nonexistent.pdf") is False
        
        # 디렉토리인 경우
        mock_exists.return_value = True
        mock_is_file.return_value = False
        assert self.parser.validate_file("directory.pdf") is False
        
        # 빈 파일
        mock_is_file.return_value = True
        mock_stat.return_value.st_size = 0
        assert self.parser.validate_file("empty.pdf") is False
        
        # 지원되지 않는 파일 형식
        mock_stat.return_value.st_size = 1000
        assert self.parser.validate_file("test.doc") is False
    
    def test_extract_text_from_txt(self):
        """텍스트 파일 파싱 테스트."""
        # 임시 텍스트 파일 생성
        test_content = "This is a test document.\nIt has multiple lines.\nAnd some content."
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False, encoding='utf-8') as f:
            f.write(test_content)
            temp_path = f.name
        
        try:
            result = self.parser.extract_text_from_txt(temp_path)
            assert result == test_content
        finally:
            os.unlink(temp_path)
    
    def test_extract_text_from_txt_encoding_error(self):
        """텍스트 파일 인코딩 오류 처리 테스트."""
        # 존재하지 않는 파일
        with pytest.raises(DocumentParsingError):
            self.parser.extract_text_from_txt("nonexistent.txt")
    
    @patch('src.document_rag_english_study.document_manager.parser.MARKDOWN_AVAILABLE', True)
    @patch('markdown.Markdown')
    def test_extract_text_from_md_with_markdown(self, mock_markdown_class):
        """Markdown 파일 파싱 테스트 (markdown 패키지 사용)."""
        # Mock 설정
        mock_md = Mock()
        mock_md.convert.return_value = "<h1>Test Title</h1><p>Test content</p>"
        mock_markdown_class.return_value = mock_md
        
        test_content = "# Test Title\nTest content"
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.md', delete=False, encoding='utf-8') as f:
            f.write(test_content)
            temp_path = f.name
        
        try:
            result = self.parser.extract_text_from_md(temp_path)
            assert "Test Title" in result
            assert "Test content" in result
            # HTML 태그가 제거되었는지 확인
            assert "<h1>" not in result
            assert "<p>" not in result
        finally:
            os.unlink(temp_path)
    
    @patch('src.document_rag_english_study.document_manager.parser.MARKDOWN_AVAILABLE', False)
    def test_extract_text_from_md_without_markdown(self):
        """Markdown 파일 파싱 테스트 (markdown 패키지 없음)."""
        test_content = "# Test Title\nTest content"
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.md', delete=False, encoding='utf-8') as f:
            f.write(test_content)
            temp_path = f.name
        
        try:
            result = self.parser.extract_text_from_md(temp_path)
            assert result == test_content  # 원본 내용 반환
        finally:
            os.unlink(temp_path)
    
    @patch('src.document_rag_english_study.document_manager.parser.PDF_AVAILABLE', True)
    @patch('pypdf.PdfReader')
    @patch('builtins.open', new_callable=mock_open, read_data=b'fake pdf content')
    def test_extract_text_from_pdf(self, mock_file, mock_pdf_reader):
        """PDF 파일 파싱 테스트."""
        # Mock 설정
        mock_page = Mock()
        mock_page.extract_text.return_value = "Test PDF content"
        mock_reader = Mock()
        mock_reader.pages = [mock_page]
        mock_pdf_reader.return_value = mock_reader
        
        result = self.parser.extract_text_from_pdf("test.pdf")
        assert result == "Test PDF content"
        mock_pdf_reader.assert_called_once()
    
    @patch('src.document_rag_english_study.document_manager.parser.PDF_AVAILABLE', False)
    def test_extract_text_from_pdf_no_dependency(self):
        """PDF 파싱 의존성 없음 테스트."""
        with pytest.raises(DocumentParsingError, match="pypdf package is required"):
            self.parser.extract_text_from_pdf("test.pdf")
    
    @patch('src.document_rag_english_study.document_manager.parser.DOCX_AVAILABLE', True)
    @patch('docx.Document')
    def test_extract_text_from_docx(self, mock_docx_document):
        """DOCX 파일 파싱 테스트."""
        # Mock 설정
        mock_paragraph = Mock()
        mock_paragraph.text = "Test paragraph content"
        
        mock_doc = Mock()
        mock_doc.paragraphs = [mock_paragraph]
        mock_doc.tables = []
        mock_docx_document.return_value = mock_doc
        
        result = self.parser.extract_text_from_docx("test.docx")
        assert result == "Test paragraph content"
        mock_docx_document.assert_called_once_with("test.docx")
    
    @patch('src.document_rag_english_study.document_manager.parser.DOCX_AVAILABLE', False)
    def test_extract_text_from_docx_no_dependency(self):
        """DOCX 파싱 의존성 없음 테스트."""
        with pytest.raises(DocumentParsingError, match="python-docx package is required"):
            self.parser.extract_text_from_docx("test.docx")
    
    @patch.object(DocumentParser, 'validate_file')
    @patch.object(DocumentParser, 'extract_text_from_txt')
    def test_parse_file_success(self, mock_extract, mock_validate):
        """파일 파싱 성공 테스트."""
        # Mock 설정
        mock_validate.return_value = True
        mock_extract.return_value = "Test document content for parsing"
        
        result = self.parser.parse_file("test.txt")
        
        assert isinstance(result, Document)
        assert result.content == "Test document content for parsing"
        assert result.file_type == "txt"
        assert result.word_count == 5  # "Test document content for parsing"
        assert result.title == "Test"  # 파일명에서 추출
        assert result.language == "english"
    
    @patch.object(DocumentParser, 'validate_file')
    def test_parse_file_validation_failure(self, mock_validate):
        """파일 검증 실패 테스트."""
        mock_validate.return_value = False
        
        with pytest.raises(DocumentParsingError, match="File validation failed"):
            self.parser.parse_file("invalid.txt")
    
    @patch.object(DocumentParser, 'validate_file')
    @patch.object(DocumentParser, 'extract_text_from_txt')
    def test_parse_file_empty_content(self, mock_extract, mock_validate):
        """빈 내용 파싱 테스트."""
        mock_validate.return_value = True
        mock_extract.return_value = ""
        
        with pytest.raises(DocumentParsingError, match="No content extracted"):
            self.parser.parse_file("empty.txt")
    
    def test_generate_document_id(self):
        """문서 ID 생성 테스트."""
        doc_id1 = self.parser._generate_document_id("test.txt")
        doc_id2 = self.parser._generate_document_id("test.txt")
        doc_id3 = self.parser._generate_document_id("other.txt")
        
        # 같은 파일은 같은 ID
        assert doc_id1 == doc_id2
        # 다른 파일은 다른 ID
        assert doc_id1 != doc_id3
        # ID 형식 확인
        assert doc_id1.startswith("doc_")
        assert len(doc_id1) == 16  # "doc_" + 12자리 해시
    
    @patch('builtins.open', new_callable=mock_open, read_data=b'test content')
    def test_generate_file_hash(self, mock_file):
        """파일 해시 생성 테스트."""
        file_hash = self.parser._generate_file_hash("test.txt")
        assert isinstance(file_hash, str)
        assert len(file_hash) == 32  # MD5 해시 길이
    
    def test_extract_title(self):
        """제목 추출 테스트."""
        # 첫 줄이 제목인 경우
        content1 = "Document Title\nThis is the content of the document."
        title1 = self.parser._extract_title("filename", content1)
        assert title1 == "Document Title"
        
        # Markdown 헤더인 경우
        content2 = "# Markdown Title\nThis is markdown content."
        title2 = self.parser._extract_title("filename", content2)
        assert title2 == "Markdown Title"
        
        # 첫 줄이 너무 긴 경우 (파일명 사용)
        content3 = "This is a very long first line that should not be used as a title because it exceeds the reasonable length limit for titles"
        title3 = self.parser._extract_title("my_document_file", content3)
        assert title3 == "My Document File"
        
        # 빈 내용인 경우
        content4 = ""
        title4 = self.parser._extract_title("empty_file", content4)
        assert title4 == "Empty File"
    
    def test_get_supported_extensions(self):
        """지원되는 확장자 목록 테스트."""
        extensions = self.parser.get_supported_extensions()
        assert isinstance(extensions, list)
        assert '.pdf' in extensions
        assert '.docx' in extensions
        assert '.txt' in extensions
        assert '.md' in extensions
        assert len(extensions) == 4
    
    def test_get_parser_info(self):
        """파서 정보 테스트."""
        info = self.parser.get_parser_info()
        assert isinstance(info, dict)
        assert 'supported_extensions' in info
        assert 'dependencies' in info
        assert 'parser_version' in info
        
        # 의존성 정보 확인
        deps = info['dependencies']
        assert 'pypdf' in deps
        assert 'python-docx' in deps
        assert 'markdown' in deps
        assert isinstance(deps['pypdf'], bool)
        assert isinstance(deps['python-docx'], bool)
        assert isinstance(deps['markdown'], bool)


class TestDocumentParsingError:
    """DocumentParsingError 예외 클래스 테스트."""
    
    def test_document_parsing_error(self):
        """DocumentParsingError 예외 테스트."""
        error_msg = "Test parsing error"
        
        with pytest.raises(DocumentParsingError) as exc_info:
            raise DocumentParsingError(error_msg)
        
        assert str(exc_info.value) == error_msg
        assert isinstance(exc_info.value, Exception)


# 통합 테스트
class TestDocumentParserIntegration:
    """DocumentParser 통합 테스트."""
    
    def setup_method(self):
        """각 테스트 메서드 실행 전 설정."""
        self.parser = DocumentParser()
    
    def test_parse_real_txt_file(self):
        """실제 텍스트 파일 파싱 통합 테스트."""
        test_content = """English Learning Document
        
This is a sample document for testing the English learning system.
It contains multiple paragraphs and various content types.

The system should be able to parse this content and use it for
conversational English learning with RAG capabilities."""
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False, encoding='utf-8') as f:
            f.write(test_content)
            temp_path = f.name
        
        try:
            document = self.parser.parse_file(temp_path)
            
            assert document is not None
            assert isinstance(document, Document)
            assert document.file_type == "txt"
            assert document.content.strip() == test_content.strip()
            assert document.word_count > 0
            assert document.title is not None
            assert document.file_hash is not None
            assert len(document.file_hash) == 32
            
        finally:
            os.unlink(temp_path)
    
    def test_parse_real_md_file(self):
        """실제 Markdown 파일 파싱 통합 테스트."""
        test_content = """# English Learning Guide

## Introduction

This is a **markdown document** for testing purposes.

### Features

- Support for *italic* and **bold** text
- Lists and other markdown features
- Code blocks and links

## Conclusion

The parser should handle this content properly."""
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.md', delete=False, encoding='utf-8') as f:
            f.write(test_content)
            temp_path = f.name
        
        try:
            document = self.parser.parse_file(temp_path)
            
            assert document is not None
            assert isinstance(document, Document)
            assert document.file_type == "md"
            assert len(document.content) > 0
            assert document.word_count > 0
            assert "English Learning Guide" in document.title or document.title == "English Learning Guide"
            
        finally:
            os.unlink(temp_path)