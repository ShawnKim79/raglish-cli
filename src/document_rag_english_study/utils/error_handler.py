"""
전역 오류 처리 및 사용자 친화적 메시지 생성 모듈

이 모듈은 시스템에서 발생하는 모든 오류를 처리하고,
사용자에게 이해하기 쉬운 메시지를 제공합니다.
"""

import logging
import traceback
from typing import Dict, Any, Optional, Callable, Type
from functools import wraps

from .exceptions import (
    DocumentRAGError, DocumentError, RAGError, LearningError,
    ConfigurationError, ValidationError, LLMError, VectorDatabaseError,
    EmbeddingError
)
from .logging_config import get_logger


class ErrorHandler:
    """전역 오류 처리를 담당하는 클래스"""
    
    def __init__(self):
        """ErrorHandler 초기화"""
        self.logger = get_logger(__name__)
        self._error_messages = self._initialize_error_messages()
        self._retry_strategies = self._initialize_retry_strategies()
    
    def _initialize_error_messages(self) -> Dict[Type[Exception], str]:
        """오류 타입별 사용자 친화적 메시지를 초기화합니다"""
        return {
            DocumentError: "문서 처리 중 오류가 발생했습니다. 파일 형식이나 내용을 확인해주세요.",
            RAGError: "검색 시스템에서 오류가 발생했습니다. 잠시 후 다시 시도해주세요.",
            LearningError: "학습 기능에서 오류가 발생했습니다. 입력 내용을 확인해주세요.",
            ConfigurationError: "설정에 문제가 있습니다. 설정을 다시 확인해주세요.",
            ValidationError: "입력 값이 올바르지 않습니다. 형식을 확인해주세요.",
            LLMError: "AI 모델 연결에 문제가 있습니다. API 키나 네트워크 연결을 확인해주세요.",
            VectorDatabaseError: "데이터베이스 연결에 문제가 있습니다. 잠시 후 다시 시도해주세요.",
            EmbeddingError: "텍스트 처리 중 오류가 발생했습니다. 입력 텍스트를 확인해주세요.",
            FileNotFoundError: "파일을 찾을 수 없습니다. 파일 경로를 확인해주세요.",
            PermissionError: "파일 접근 권한이 없습니다. 권한을 확인해주세요.",
            ConnectionError: "네트워크 연결에 문제가 있습니다. 인터넷 연결을 확인해주세요.",
            TimeoutError: "요청 시간이 초과되었습니다. 잠시 후 다시 시도해주세요.",
            KeyError: "필요한 정보가 누락되었습니다. 설정을 확인해주세요.",
            ValueError: "입력 값이 올바르지 않습니다. 값을 확인해주세요.",
            TypeError: "데이터 타입이 올바르지 않습니다. 입력을 확인해주세요.",
        }
    
    def _initialize_retry_strategies(self) -> Dict[Type[Exception], int]:
        """오류 타입별 재시도 횟수를 초기화합니다"""
        return {
            ConnectionError: 3,
            TimeoutError: 2,
            LLMError: 2,
            VectorDatabaseError: 2,
            EmbeddingError: 1,
        }
    
    def handle_error(self, error: Exception, context: Optional[Dict[str, Any]] = None) -> None:
        """
        오류를 처리하고 로그에 기록합니다
        
        Args:
            error: 발생한 오류
            context: 추가 컨텍스트 정보
        """
        context = context or {}
        
        # 오류 정보 수집
        error_info = {
            'error_type': type(error).__name__,
            'error_message': str(error),
            'traceback': traceback.format_exc(),
            **context
        }
        
        # DocumentRAGError의 경우 추가 정보 포함
        if isinstance(error, DocumentRAGError):
            error_info.update({
                'error_code': getattr(error, 'error_code', None),
                'error_context': getattr(error, 'context', {})
            })
        
        # 로그 레벨 결정
        if isinstance(error, (DocumentRAGError, ConnectionError, TimeoutError)):
            log_level = logging.WARNING
        elif isinstance(error, (KeyError, ValueError, TypeError)):
            log_level = logging.ERROR
        else:
            log_level = logging.ERROR
        
        # 로그 기록
        self.log_error(error, error_info, log_level)
    
    def log_error(self, error: Exception, context: Dict[str, Any], level: int = logging.ERROR) -> None:
        """
        오류를 로그에 기록합니다
        
        Args:
            error: 발생한 오류
            context: 컨텍스트 정보
            level: 로그 레벨
        """
        self.logger.log(level, f"오류 발생: {type(error).__name__}: {str(error)}")
        
        # 컨텍스트 정보 로그
        if context:
            self.logger.log(level, "컨텍스트 정보:")
            for key, value in context.items():
                if key != 'traceback':  # traceback은 별도로 처리
                    self.logger.log(level, f"  {key}: {value}")
        
        # 디버그 레벨에서만 전체 traceback 출력
        if self.logger.isEnabledFor(logging.DEBUG):
            self.logger.debug(f"전체 traceback:\n{context.get('traceback', '')}")
    
    def get_user_friendly_message(self, error: Exception) -> str:
        """
        사용자 친화적인 오류 메시지를 생성합니다
        
        Args:
            error: 발생한 오류
            
        Returns:
            str: 사용자 친화적인 오류 메시지
        """
        error_type = type(error)
        
        # 직접 매칭되는 메시지 찾기
        if error_type in self._error_messages:
            base_message = self._error_messages[error_type]
        else:
            # 상속 관계를 고려한 메시지 찾기
            for exc_type, message in self._error_messages.items():
                if isinstance(error, exc_type):
                    base_message = message
                    break
            else:
                base_message = "예상치 못한 오류가 발생했습니다. 잠시 후 다시 시도해주세요."
        
        # DocumentRAGError의 경우 추가 정보 포함
        if isinstance(error, DocumentRAGError):
            if hasattr(error, 'context') and error.context:
                if 'file_path' in error.context:
                    base_message += f" (파일: {error.context['file_path']})"
                elif 'operation' in error.context:
                    base_message += f" (작업: {error.context['operation']})"
                elif 'config_key' in error.context:
                    base_message += f" (설정: {error.context['config_key']})"
        
        return base_message
    
    def should_retry(self, error: Exception) -> bool:
        """
        오류가 재시도 가능한지 확인합니다
        
        Args:
            error: 발생한 오류
            
        Returns:
            bool: 재시도 가능 여부
        """
        error_type = type(error)
        return error_type in self._retry_strategies
    
    def get_retry_count(self, error: Exception) -> int:
        """
        오류에 대한 재시도 횟수를 반환합니다
        
        Args:
            error: 발생한 오류
            
        Returns:
            int: 재시도 횟수
        """
        error_type = type(error)
        return self._retry_strategies.get(error_type, 0)
    
    def is_critical_error(self, error: Exception) -> bool:
        """
        치명적 오류인지 확인합니다
        
        Args:
            error: 발생한 오류
            
        Returns:
            bool: 치명적 오류 여부
        """
        critical_errors = (
            SystemExit, KeyboardInterrupt, MemoryError,
            PermissionError, OSError
        )
        return isinstance(error, critical_errors)


# 전역 오류 처리기 인스턴스
_error_handler: Optional[ErrorHandler] = None


def get_error_handler() -> ErrorHandler:
    """전역 오류 처리기 인스턴스를 반환합니다"""
    global _error_handler
    if _error_handler is None:
        _error_handler = ErrorHandler()
    return _error_handler


def handle_error(error: Exception, context: Optional[Dict[str, Any]] = None) -> str:
    """
    오류를 처리하고 사용자 친화적 메시지를 반환합니다
    
    Args:
        error: 발생한 오류
        context: 추가 컨텍스트 정보
        
    Returns:
        str: 사용자 친화적인 오류 메시지
    """
    handler = get_error_handler()
    handler.handle_error(error, context)
    return handler.get_user_friendly_message(error)


def error_handler_decorator(context: Optional[Dict[str, Any]] = None):
    """
    함수에 오류 처리를 적용하는 데코레이터
    
    Args:
        context: 추가 컨텍스트 정보
    """
    def decorator(func: Callable):
        @wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                error_context = {
                    'function': func.__name__,
                    'module': func.__module__,
                    **(context or {})
                }
                
                handler = get_error_handler()
                handler.handle_error(e, error_context)
                
                # 치명적 오류는 다시 발생시킴
                if handler.is_critical_error(e):
                    raise
                
                # 사용자 친화적 메시지와 함께 새로운 예외 발생
                user_message = handler.get_user_friendly_message(e)
                if isinstance(e, DocumentRAGError):
                    raise e  # 이미 우리의 예외이므로 그대로 전파
                else:
                    # 일반 예외를 우리의 예외로 래핑
                    raise DocumentRAGError(user_message) from e
        
        return wrapper
    return decorator


def retry_on_error(max_retries: Optional[int] = None, delay: float = 1.0):
    """
    오류 발생 시 재시도하는 데코레이터
    
    Args:
        max_retries: 최대 재시도 횟수 (None이면 오류 타입별 기본값 사용)
        delay: 재시도 간격 (초)
    """
    def decorator(func: Callable):
        @wraps(func)
        def wrapper(*args, **kwargs):
            handler = get_error_handler()
            last_error = None
            
            # 첫 번째 시도
            try:
                return func(*args, **kwargs)
            except Exception as e:
                last_error = e
                
                # 재시도 가능한 오류인지 확인
                if not handler.should_retry(e):
                    raise
                
                # 재시도 횟수 결정
                retry_count = max_retries if max_retries is not None else handler.get_retry_count(e)
                
                # 재시도 수행
                import time
                for attempt in range(retry_count):
                    try:
                        handler.logger.info(f"재시도 {attempt + 1}/{retry_count}: {func.__name__}")
                        time.sleep(delay)
                        return func(*args, **kwargs)
                    except Exception as retry_error:
                        last_error = retry_error
                        if attempt == retry_count - 1:  # 마지막 시도
                            break
                
                # 모든 재시도 실패
                handler.handle_error(last_error, {
                    'function': func.__name__,
                    'max_retries': retry_count,
                    'final_attempt': True
                })
                raise last_error
        
        return wrapper
    return decorator