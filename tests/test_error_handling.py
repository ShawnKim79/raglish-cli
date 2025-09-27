"""
오류 처리 및 로깅 시스템 테스트 모듈
"""

import pytest
import logging
import tempfile
import os
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

from src.document_rag_english_study.utils import (
    DocumentRAGError, DocumentError, RAGError, LearningError,
    ConfigurationError, ValidationError, LLMError, VectorDatabaseError,
    EmbeddingError, ErrorHandler, get_error_handler, handle_error,
    error_handler_decorator, retry_on_error, LoggingConfig,
    get_logging_config, setup_logging, get_logger
)


class TestExceptions:
    """예외 클래스들의 테스트"""
    
    def test_document_rag_error_basic(self):
        """기본 DocumentRAGError 테스트"""
        error = DocumentRAGError("테스트 오류")
        assert str(error) == "테스트 오류"
        assert error.message == "테스트 오류"
        assert error.error_code is None
        assert error.context == {}
    
    def test_document_rag_error_with_context(self):
        """컨텍스트가 있는 DocumentRAGError 테스트"""
        context = {"key": "value", "number": 42}
        error = DocumentRAGError("테스트 오류", error_code="TEST001", context=context)
        
        assert error.error_code == "TEST001"
        assert error.context == context
    
    def test_document_error(self):
        """DocumentError 테스트"""
        error = DocumentError("파일 오류", file_path="/test/file.txt")
        assert error.file_path == "/test/file.txt"
        assert error.context["file_path"] == "/test/file.txt"
    
    def test_rag_error(self):
        """RAGError 테스트"""
        error = RAGError("검색 오류", operation="search")
        assert error.operation == "search"
        assert error.context["operation"] == "search"
    
    def test_learning_error(self):
        """LearningError 테스트"""
        error = LearningError("학습 오류", learning_component="grammar_checker")
        assert error.learning_component == "grammar_checker"
        assert error.context["learning_component"] == "grammar_checker"
    
    def test_configuration_error(self):
        """ConfigurationError 테스트"""
        error = ConfigurationError("설정 오류", config_key="llm_provider")
        assert error.config_key == "llm_provider"
        assert error.context["config_key"] == "llm_provider"
    
    def test_validation_error(self):
        """ValidationError 테스트"""
        error = ValidationError("검증 오류", field_name="email", field_value="invalid-email")
        assert error.field_name == "email"
        assert error.field_value == "invalid-email"
        assert error.context["field_name"] == "email"
        assert error.context["field_value"] == "invalid-email"
    
    def test_llm_error(self):
        """LLMError 테스트"""
        error = LLMError("LLM 오류", provider="openai", model="gpt-3.5-turbo")
        assert error.provider == "openai"
        assert error.model == "gpt-3.5-turbo"
        assert error.context["provider"] == "openai"
        assert error.context["model"] == "gpt-3.5-turbo"
    
    def test_vector_database_error(self):
        """VectorDatabaseError 테스트"""
        error = VectorDatabaseError("벡터 DB 오류", collection_name="documents")
        assert error.collection_name == "documents"
        assert error.context["collection_name"] == "documents"
    
    def test_embedding_error(self):
        """EmbeddingError 테스트"""
        error = EmbeddingError("임베딩 오류", text_length=1000)
        assert error.text_length == 1000
        assert error.context["text_length"] == 1000


class TestErrorHandler:
    """ErrorHandler 클래스 테스트"""
    
    def setup_method(self):
        """각 테스트 메서드 실행 전 설정"""
        self.handler = ErrorHandler()
    
    def test_get_user_friendly_message_document_error(self):
        """DocumentError에 대한 사용자 친화적 메시지 테스트"""
        error = DocumentError("파일을 읽을 수 없습니다", file_path="/test/file.txt")
        message = self.handler.get_user_friendly_message(error)
        
        assert "문서 처리 중 오류가 발생했습니다" in message
        assert "/test/file.txt" in message
    
    def test_get_user_friendly_message_rag_error(self):
        """RAGError에 대한 사용자 친화적 메시지 테스트"""
        error = RAGError("검색 실패", operation="similarity_search")
        message = self.handler.get_user_friendly_message(error)
        
        assert "검색 시스템에서 오류가 발생했습니다" in message
        assert "similarity_search" in message
    
    def test_get_user_friendly_message_configuration_error(self):
        """ConfigurationError에 대한 사용자 친화적 메시지 테스트"""
        error = ConfigurationError("설정 누락", config_key="api_key")
        message = self.handler.get_user_friendly_message(error)
        
        assert "설정에 문제가 있습니다" in message
        assert "api_key" in message
    
    def test_get_user_friendly_message_unknown_error(self):
        """알 수 없는 오류에 대한 기본 메시지 테스트"""
        error = RuntimeError("알 수 없는 오류")
        message = self.handler.get_user_friendly_message(error)
        
        assert "예상치 못한 오류가 발생했습니다" in message
    
    def test_should_retry(self):
        """재시도 가능 여부 테스트"""
        # 재시도 가능한 오류들
        assert self.handler.should_retry(ConnectionError("연결 오류"))
        assert self.handler.should_retry(TimeoutError("시간 초과"))
        assert self.handler.should_retry(LLMError("LLM 오류"))
        
        # 재시도 불가능한 오류들
        assert not self.handler.should_retry(ValueError("값 오류"))
        assert not self.handler.should_retry(DocumentError("문서 오류"))
    
    def test_get_retry_count(self):
        """재시도 횟수 테스트"""
        assert self.handler.get_retry_count(ConnectionError("연결 오류")) == 3
        assert self.handler.get_retry_count(TimeoutError("시간 초과")) == 2
        assert self.handler.get_retry_count(LLMError("LLM 오류")) == 2
        assert self.handler.get_retry_count(ValueError("값 오류")) == 0
    
    def test_is_critical_error(self):
        """치명적 오류 판단 테스트"""
        # 치명적 오류들
        assert self.handler.is_critical_error(SystemExit())
        assert self.handler.is_critical_error(KeyboardInterrupt())
        assert self.handler.is_critical_error(MemoryError())
        assert self.handler.is_critical_error(PermissionError("권한 없음"))
        
        # 일반 오류들
        assert not self.handler.is_critical_error(ValueError("값 오류"))
        assert not self.handler.is_critical_error(DocumentError("문서 오류"))
    
    def test_handle_error(self):
        """오류 처리 테스트"""
        with patch.object(self.handler, 'logger') as mock_logger:
            error = DocumentError("테스트 오류", file_path="/test/file.txt")
            context = {"operation": "test"}
            
            self.handler.handle_error(error, context)
            
            # 로거가 호출되었는지 확인
            mock_logger.log.assert_called()
    
    def test_log_error(self):
        """오류 로깅 테스트"""
        with patch.object(self.handler, 'logger') as mock_logger:
            error = ValueError("테스트 오류")
            context = {"key": "value"}
            
            self.handler.log_error(error, context, logging.WARNING)
            
            # 로거가 호출되었는지 확인 (여러 번 호출되므로 call_args_list 확인)
            calls = mock_logger.log.call_args_list
            assert len(calls) >= 1
            # 첫 번째 호출이 오류 메시지인지 확인
            assert calls[0][0] == (logging.WARNING, "오류 발생: ValueError: 테스트 오류")


class TestLoggingConfig:
    """LoggingConfig 클래스 테스트"""
    
    def setup_method(self):
        """각 테스트 메서드 실행 전 설정"""
        self.temp_dir = tempfile.mkdtemp()
        self.config = LoggingConfig(log_dir=self.temp_dir, app_name="test_app")
    
    def teardown_method(self):
        """각 테스트 메서드 실행 후 정리"""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_initialization(self):
        """LoggingConfig 초기화 테스트"""
        assert self.config.app_name == "test_app"
        assert self.config.log_dir == Path(self.temp_dir)
        assert self.config.log_dir.exists()
    
    def test_setup_logging(self):
        """로깅 설정 테스트"""
        self.config.setup_logging(level="DEBUG", console_output=True, file_output=True)
        
        # 로그 파일들이 생성되는지 확인 (실제 로그가 기록될 때 생성됨)
        logger = self.config.get_logger("test")
        logger.info("테스트 로그")
        
        assert self.config.main_log_file.exists()
    
    def test_get_logger(self):
        """로거 생성 테스트"""
        logger = self.config.get_logger("test_logger")
        assert isinstance(logger, logging.Logger)
        assert logger.name == "test_logger"
    
    def test_log_system_info(self):
        """시스템 정보 로깅 테스트"""
        # 예외가 발생하지 않는지 확인
        self.config.log_system_info()
    
    def test_cleanup_old_logs(self):
        """오래된 로그 파일 정리 테스트"""
        # 테스트 로그 파일 생성
        test_log = self.config.log_dir / "old_test.log"
        test_log.write_text("test log content")
        
        # 파일 수정 시간을 과거로 설정
        old_time = os.path.getmtime(test_log) - (31 * 24 * 60 * 60)  # 31일 전
        os.utime(test_log, (old_time, old_time))
        
        self.config.cleanup_old_logs(days_to_keep=30)
        
        # 파일이 삭제되었는지 확인
        assert not test_log.exists()


class TestDecorators:
    """데코레이터 함수들 테스트"""
    
    def test_error_handler_decorator_success(self):
        """오류 처리 데코레이터 - 성공 케이스"""
        @error_handler_decorator(context={"test": "context"})
        def test_function():
            return "success"
        
        result = test_function()
        assert result == "success"
    
    def test_error_handler_decorator_error(self):
        """오류 처리 데코레이터 - 오류 케이스"""
        @error_handler_decorator(context={"test": "context"})
        def test_function():
            raise ValueError("테스트 오류")
        
        with pytest.raises(DocumentRAGError):
            test_function()
    
    def test_error_handler_decorator_critical_error(self):
        """오류 처리 데코레이터 - 치명적 오류 케이스"""
        @error_handler_decorator()
        def test_function():
            raise KeyboardInterrupt()
        
        with pytest.raises(KeyboardInterrupt):
            test_function()
    
    def test_retry_on_error_success(self):
        """재시도 데코레이터 - 성공 케이스"""
        @retry_on_error(max_retries=2, delay=0.1)
        def test_function():
            return "success"
        
        result = test_function()
        assert result == "success"
    
    def test_retry_on_error_with_retries(self):
        """재시도 데코레이터 - 재시도 후 성공"""
        call_count = 0
        
        @retry_on_error(max_retries=2, delay=0.1)
        def test_function():
            nonlocal call_count
            call_count += 1
            if call_count < 2:
                raise ConnectionError("연결 실패")
            return "success"
        
        result = test_function()
        assert result == "success"
        assert call_count == 2
    
    def test_retry_on_error_max_retries_exceeded(self):
        """재시도 데코레이터 - 최대 재시도 초과"""
        @retry_on_error(max_retries=2, delay=0.1)
        def test_function():
            raise ConnectionError("연결 실패")
        
        with pytest.raises(ConnectionError):
            test_function()
    
    def test_retry_on_error_non_retryable(self):
        """재시도 데코레이터 - 재시도 불가능한 오류"""
        @retry_on_error(max_retries=2, delay=0.1)
        def test_function():
            raise ValueError("값 오류")
        
        with pytest.raises(ValueError):
            test_function()


class TestGlobalFunctions:
    """전역 함수들 테스트"""
    
    def test_get_error_handler(self):
        """전역 오류 처리기 인스턴스 테스트"""
        handler1 = get_error_handler()
        handler2 = get_error_handler()
        
        assert isinstance(handler1, ErrorHandler)
        assert handler1 is handler2  # 싱글톤 패턴
    
    def test_handle_error_function(self):
        """전역 handle_error 함수 테스트"""
        error = DocumentError("테스트 오류")
        message = handle_error(error, {"context": "test"})
        
        assert isinstance(message, str)
        assert "문서 처리 중 오류가 발생했습니다" in message
    
    def test_get_logging_config(self):
        """전역 로깅 설정 인스턴스 테스트"""
        config1 = get_logging_config()
        config2 = get_logging_config()
        
        assert isinstance(config1, LoggingConfig)
        assert config1 is config2  # 싱글톤 패턴
    
    def test_setup_logging_function(self):
        """전역 setup_logging 함수 테스트"""
        # 예외가 발생하지 않는지 확인
        setup_logging(level="INFO", console_output=False, file_output=True)
    
    def test_get_logger_function(self):
        """전역 get_logger 함수 테스트"""
        logger = get_logger("test_logger")
        assert isinstance(logger, logging.Logger)
        assert logger.name == "test_logger"


if __name__ == "__main__":
    pytest.main([__file__])