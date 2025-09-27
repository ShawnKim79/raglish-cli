"""
유틸리티 모듈들을 제공하는 패키지

이 패키지는 오류 처리, 로깅, 예외 클래스 등
애플리케이션 전반에서 사용되는 유틸리티 기능들을 제공합니다.
"""

from .exceptions import (
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

from .error_handler import (
    ErrorHandler,
    get_error_handler,
    handle_error,
    error_handler_decorator,
    retry_on_error
)

from .logging_config import (
    LoggingConfig,
    get_logging_config,
    setup_logging,
    get_logger,
    log_context
)

__all__ = [
    # 예외 클래스들
    'DocumentRAGError',
    'DocumentError',
    'RAGError',
    'LearningError',
    'ConfigurationError',
    'ValidationError',
    'LLMError',
    'VectorDatabaseError',
    'EmbeddingError',
    
    # 오류 처리
    'ErrorHandler',
    'get_error_handler',
    'handle_error',
    'error_handler_decorator',
    'retry_on_error',
    
    # 로깅
    'LoggingConfig',
    'get_logging_config',
    'setup_logging',
    'get_logger',
    'log_context'
]