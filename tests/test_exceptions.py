"""
Exceptions 모듈 단위 테스트.

이 모듈은 사용자 정의 예외 클래스들의 기능을 테스트합니다.
"""

import pytest
from src.document_rag_english_study.utils.exceptions import (
    DocumentRAGError,
    DocumentError,
    RAGError,
    LearningError,
    ConfigurationError,
    ValidationError,
    LLMError,
    VectorDatabaseError,
    EmbeddingError
)


class TestDocumentRAGError:
    """DocumentRAGError 기본 예외 클래스 테스트."""

    def test_basic_error(self):
        """기본 오류 생성 테스트."""
        error = DocumentRAGError("Test error message")
        
        assert str(error) == "Test error message"
        assert error.message == "Test error message"
        assert error.error_code is None
        assert error.context == {}

    def test_error_with_code(self):
        """오류 코드가 있는 오류 테스트."""
        error = DocumentRAGError(
            "Test error",
            error_code="ERR001"
        )
        
        assert error.error_code == "ERR001"
        assert error.message == "Test error"

    def test_error_with_context(self):
        """컨텍스트가 있는 오류 테스트."""
        context = {"user_id": "123", "operation": "test"}
        error = DocumentRAGError(
            "Test error",
            context=context
        )
        
        assert error.context == context

    def test_error_inheritance(self):
        """예외 상속 확인 테스트."""
        error = DocumentRAGError("Test")
        
        assert isinstance(error, Exception)
        assert isinstance(error, DocumentRAGError)


class TestDocumentError:
    """DocumentError 예외 클래스 테스트."""

    def test_basic_document_error(self):
        """기본 문서 오류 테스트."""
        error = DocumentError("Document processing failed")
        
        assert str(error) == "Document processing failed"
        assert error.file_path is None
        assert isinstance(error, DocumentRAGError)

    def test_document_error_with_file_path(self):
        """파일 경로가 있는 문서 오류 테스트."""
        error = DocumentError(
            "Failed to parse document",
            file_path="/test/document.pdf"
        )
        
        assert error.file_path == "/test/document.pdf"
        assert error.context["file_path"] == "/test/document.pdf"

    def test_document_error_with_all_params(self):
        """모든 매개변수가 있는 문서 오류 테스트."""
        error = DocumentError(
            "Parse error",
            file_path="/test/doc.txt",
            error_code="DOC001",
            context={"line": 42}
        )
        
        assert error.message == "Parse error"
        assert error.file_path == "/test/doc.txt"
        assert error.error_code == "DOC001"
        assert error.context["file_path"] == "/test/doc.txt"
        assert error.context["line"] == 42


class TestRAGError:
    """RAGError 예외 클래스 테스트."""

    def test_basic_rag_error(self):
        """기본 RAG 오류 테스트."""
        error = RAGError("RAG operation failed")
        
        assert str(error) == "RAG operation failed"
        assert error.operation is None
        assert isinstance(error, DocumentRAGError)

    def test_rag_error_with_operation(self):
        """작업명이 있는 RAG 오류 테스트."""
        error = RAGError(
            "Search failed",
            operation="vector_search"
        )
        
        assert error.operation == "vector_search"
        assert error.context["operation"] == "vector_search"

    def test_rag_error_inheritance(self):
        """RAG 오류 상속 확인 테스트."""
        error = RAGError("Test")
        
        assert isinstance(error, DocumentRAGError)
        assert isinstance(error, RAGError)


class TestLearningError:
    """LearningError 예외 클래스 테스트."""

    def test_basic_learning_error(self):
        """기본 학습 오류 테스트."""
        error = LearningError("Learning component failed")
        
        assert str(error) == "Learning component failed"
        assert error.learning_component is None

    def test_learning_error_with_component(self):
        """컴포넌트명이 있는 학습 오류 테스트."""
        error = LearningError(
            "Grammar analysis failed",
            learning_component="grammar_analyzer"
        )
        
        assert error.learning_component == "grammar_analyzer"
        assert error.context["learning_component"] == "grammar_analyzer"

    def test_learning_error_inheritance(self):
        """학습 오류 상속 확인 테스트."""
        error = LearningError("Test")
        
        assert isinstance(error, DocumentRAGError)
        assert isinstance(error, LearningError)


class TestConfigurationError:
    """ConfigurationError 예외 클래스 테스트."""

    def test_basic_configuration_error(self):
        """기본 설정 오류 테스트."""
        error = ConfigurationError("Configuration is invalid")
        
        assert str(error) == "Configuration is invalid"
        assert error.config_key is None

    def test_configuration_error_with_key(self):
        """설정 키가 있는 설정 오류 테스트."""
        error = ConfigurationError(
            "Invalid API key",
            config_key="openai_api_key"
        )
        
        assert error.config_key == "openai_api_key"
        assert error.context["config_key"] == "openai_api_key"

    def test_configuration_error_inheritance(self):
        """설정 오류 상속 확인 테스트."""
        error = ConfigurationError("Test")
        
        assert isinstance(error, DocumentRAGError)
        assert isinstance(error, ConfigurationError)


class TestValidationError:
    """ValidationError 예외 클래스 테스트."""

    def test_basic_validation_error(self):
        """기본 검증 오류 테스트."""
        error = ValidationError("Validation failed")
        
        assert str(error) == "Validation failed"
        assert error.field_name is None
        assert error.field_value is None

    def test_validation_error_with_field(self):
        """필드명이 있는 검증 오류 테스트."""
        error = ValidationError(
            "Invalid email format",
            field_name="email"
        )
        
        assert error.field_name == "email"
        assert error.context["field_name"] == "email"

    def test_validation_error_with_value(self):
        """필드값이 있는 검증 오류 테스트."""
        error = ValidationError(
            "Value out of range",
            field_name="temperature",
            field_value=5.0
        )
        
        assert error.field_name == "temperature"
        assert error.field_value == 5.0
        assert error.context["field_name"] == "temperature"
        assert error.context["field_value"] == "5.0"

    def test_validation_error_with_none_value(self):
        """None 값이 있는 검증 오류 테스트."""
        error = ValidationError(
            "Required field is None",
            field_name="api_key",
            field_value=None
        )
        
        assert error.field_value is None
        assert "field_value" not in error.context  # None은 컨텍스트에 추가되지 않음

    def test_validation_error_inheritance(self):
        """검증 오류 상속 확인 테스트."""
        error = ValidationError("Test")
        
        assert isinstance(error, DocumentRAGError)
        assert isinstance(error, ValidationError)


class TestLLMError:
    """LLMError 예외 클래스 테스트."""

    def test_basic_llm_error(self):
        """기본 LLM 오류 테스트."""
        error = LLMError("LLM request failed")
        
        assert str(error) == "LLM request failed"
        assert error.provider is None
        assert error.model is None

    def test_llm_error_with_provider(self):
        """제공업체가 있는 LLM 오류 테스트."""
        error = LLMError(
            "API rate limit exceeded",
            provider="openai"
        )
        
        assert error.provider == "openai"
        assert error.context["provider"] == "openai"

    def test_llm_error_with_model(self):
        """모델명이 있는 LLM 오류 테스트."""
        error = LLMError(
            "Model not found",
            provider="openai",
            model="gpt-4"
        )
        
        assert error.provider == "openai"
        assert error.model == "gpt-4"
        assert error.context["provider"] == "openai"
        assert error.context["model"] == "gpt-4"

    def test_llm_error_inheritance(self):
        """LLM 오류 상속 확인 테스트."""
        error = LLMError("Test")
        
        assert isinstance(error, DocumentRAGError)
        assert isinstance(error, LLMError)


class TestVectorDatabaseError:
    """VectorDatabaseError 예외 클래스 테스트."""

    def test_basic_vector_db_error(self):
        """기본 벡터 DB 오류 테스트."""
        error = VectorDatabaseError("Vector database connection failed")
        
        assert str(error) == "Vector database connection failed"
        assert error.collection_name is None

    def test_vector_db_error_with_collection(self):
        """컬렉션명이 있는 벡터 DB 오류 테스트."""
        error = VectorDatabaseError(
            "Collection not found",
            collection_name="documents"
        )
        
        assert error.collection_name == "documents"
        assert error.context["collection_name"] == "documents"

    def test_vector_db_error_inheritance(self):
        """벡터 DB 오류 상속 확인 테스트."""
        error = VectorDatabaseError("Test")
        
        assert isinstance(error, DocumentRAGError)
        assert isinstance(error, RAGError)
        assert isinstance(error, VectorDatabaseError)


class TestEmbeddingError:
    """EmbeddingError 예외 클래스 테스트."""

    def test_basic_embedding_error(self):
        """기본 임베딩 오류 테스트."""
        error = EmbeddingError("Embedding generation failed")
        
        assert str(error) == "Embedding generation failed"
        assert error.text_length is None

    def test_embedding_error_with_length(self):
        """텍스트 길이가 있는 임베딩 오류 테스트."""
        error = EmbeddingError(
            "Text too long for embedding",
            text_length=10000
        )
        
        assert error.text_length == 10000
        assert error.context["text_length"] == 10000

    def test_embedding_error_with_zero_length(self):
        """길이가 0인 임베딩 오류 테스트."""
        error = EmbeddingError(
            "Empty text",
            text_length=0
        )
        
        assert error.text_length == 0
        assert error.context["text_length"] == 0

    def test_embedding_error_inheritance(self):
        """임베딩 오류 상속 확인 테스트."""
        error = EmbeddingError("Test")
        
        assert isinstance(error, DocumentRAGError)
        assert isinstance(error, RAGError)
        assert isinstance(error, EmbeddingError)


class TestExceptionHierarchy:
    """예외 계층 구조 테스트."""

    def test_all_exceptions_inherit_from_base(self):
        """모든 예외가 기본 클래스를 상속하는지 테스트."""
        exceptions = [
            DocumentError("test"),
            RAGError("test"),
            LearningError("test"),
            ConfigurationError("test"),
            ValidationError("test"),
            LLMError("test"),
            VectorDatabaseError("test"),
            EmbeddingError("test")
        ]
        
        for exc in exceptions:
            assert isinstance(exc, DocumentRAGError)
            assert isinstance(exc, Exception)

    def test_rag_specific_exceptions(self):
        """RAG 관련 예외들이 RAGError를 상속하는지 테스트."""
        rag_exceptions = [
            VectorDatabaseError("test"),
            EmbeddingError("test")
        ]
        
        for exc in rag_exceptions:
            assert isinstance(exc, RAGError)
            assert isinstance(exc, DocumentRAGError)

    def test_exception_messages(self):
        """예외 메시지가 올바르게 설정되는지 테스트."""
        test_message = "This is a test error message"
        exceptions = [
            DocumentRAGError(test_message),
            DocumentError(test_message),
            RAGError(test_message),
            LearningError(test_message),
            ConfigurationError(test_message),
            ValidationError(test_message),
            LLMError(test_message),
            VectorDatabaseError(test_message),
            EmbeddingError(test_message)
        ]
        
        for exc in exceptions:
            assert str(exc) == test_message
            assert exc.message == test_message


class TestExceptionContextHandling:
    """예외 컨텍스트 처리 테스트."""

    def test_context_merging(self):
        """컨텍스트 병합 테스트."""
        initial_context = {"user_id": "123", "session_id": "abc"}
        error = DocumentError(
            "Test error",
            file_path="/test/file.txt",
            context=initial_context
        )
        
        # 초기 컨텍스트와 file_path가 모두 포함되어야 함
        assert error.context["user_id"] == "123"
        assert error.context["session_id"] == "abc"
        assert error.context["file_path"] == "/test/file.txt"

    def test_context_override(self):
        """컨텍스트 덮어쓰기 테스트."""
        initial_context = {"file_path": "/old/path.txt"}
        error = DocumentError(
            "Test error",
            file_path="/new/path.txt",
            context=initial_context
        )
        
        # 새로운 file_path가 초기 컨텍스트를 덮어써야 함
        assert error.context["file_path"] == "/new/path.txt"

    def test_empty_context_handling(self):
        """빈 컨텍스트 처리 테스트."""
        error = DocumentRAGError("Test error")
        
        assert error.context == {}
        assert isinstance(error.context, dict)

    def test_none_context_handling(self):
        """None 컨텍스트 처리 테스트."""
        error = DocumentRAGError("Test error", context=None)
        
        assert error.context == {}
        assert isinstance(error.context, dict)