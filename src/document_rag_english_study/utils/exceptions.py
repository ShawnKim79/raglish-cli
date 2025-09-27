"""
사용자 정의 예외 클래스들을 정의하는 모듈

이 모듈은 시스템의 다양한 컴포넌트에서 발생할 수 있는 
특정 오류 상황을 나타내는 예외 클래스들을 제공합니다.
"""

from typing import Optional, Dict, Any


class DocumentRAGError(Exception):
    """모든 Document RAG 관련 오류의 기본 클래스"""
    
    def __init__(self, message: str, error_code: Optional[str] = None, context: Optional[Dict[str, Any]] = None):
        """
        DocumentRAGError 초기화
        
        Args:
            message: 오류 메시지
            error_code: 오류 코드 (선택사항)
            context: 추가 컨텍스트 정보 (선택사항)
        """
        super().__init__(message)
        self.message = message
        self.error_code = error_code
        self.context = context or {}


class DocumentError(DocumentRAGError):
    """문서 처리 관련 오류"""
    
    def __init__(self, message: str, file_path: Optional[str] = None, **kwargs):
        """
        DocumentError 초기화
        
        Args:
            message: 오류 메시지
            file_path: 문제가 발생한 파일 경로 (선택사항)
            **kwargs: 추가 인자들
        """
        super().__init__(message, **kwargs)
        self.file_path = file_path
        if file_path:
            self.context['file_path'] = file_path


class RAGError(DocumentRAGError):
    """RAG 엔진 관련 오류"""
    
    def __init__(self, message: str, operation: Optional[str] = None, **kwargs):
        """
        RAGError 초기화
        
        Args:
            message: 오류 메시지
            operation: 실패한 작업명 (선택사항)
            **kwargs: 추가 인자들
        """
        super().__init__(message, **kwargs)
        self.operation = operation
        if operation:
            self.context['operation'] = operation


class LearningError(DocumentRAGError):
    """학습 모듈 관련 오류"""
    
    def __init__(self, message: str, learning_component: Optional[str] = None, **kwargs):
        """
        LearningError 초기화
        
        Args:
            message: 오류 메시지
            learning_component: 오류가 발생한 학습 컴포넌트 (선택사항)
            **kwargs: 추가 인자들
        """
        super().__init__(message, **kwargs)
        self.learning_component = learning_component
        if learning_component:
            self.context['learning_component'] = learning_component


class ConfigurationError(DocumentRAGError):
    """설정 관련 오류"""
    
    def __init__(self, message: str, config_key: Optional[str] = None, **kwargs):
        """
        ConfigurationError 초기화
        
        Args:
            message: 오류 메시지
            config_key: 문제가 발생한 설정 키 (선택사항)
            **kwargs: 추가 인자들
        """
        super().__init__(message, **kwargs)
        self.config_key = config_key
        if config_key:
            self.context['config_key'] = config_key


class ValidationError(DocumentRAGError):
    """입력 검증 오류"""
    
    def __init__(self, message: str, field_name: Optional[str] = None, field_value: Optional[Any] = None, **kwargs):
        """
        ValidationError 초기화
        
        Args:
            message: 오류 메시지
            field_name: 검증에 실패한 필드명 (선택사항)
            field_value: 검증에 실패한 값 (선택사항)
            **kwargs: 추가 인자들
        """
        super().__init__(message, **kwargs)
        self.field_name = field_name
        self.field_value = field_value
        if field_name:
            self.context['field_name'] = field_name
        if field_value is not None:
            self.context['field_value'] = str(field_value)


class LLMError(DocumentRAGError):
    """LLM 관련 오류"""
    
    def __init__(self, message: str, provider: Optional[str] = None, model: Optional[str] = None, **kwargs):
        """
        LLMError 초기화
        
        Args:
            message: 오류 메시지
            provider: LLM 제공업체 (선택사항)
            model: 모델명 (선택사항)
            **kwargs: 추가 인자들
        """
        super().__init__(message, **kwargs)
        self.provider = provider
        self.model = model
        if provider:
            self.context['provider'] = provider
        if model:
            self.context['model'] = model


class VectorDatabaseError(RAGError):
    """벡터 데이터베이스 관련 오류"""
    
    def __init__(self, message: str, collection_name: Optional[str] = None, **kwargs):
        """
        VectorDatabaseError 초기화
        
        Args:
            message: 오류 메시지
            collection_name: 컬렉션명 (선택사항)
            **kwargs: 추가 인자들
        """
        super().__init__(message, **kwargs)
        self.collection_name = collection_name
        if collection_name:
            self.context['collection_name'] = collection_name


class EmbeddingError(RAGError):
    """임베딩 생성 관련 오류"""
    
    def __init__(self, message: str, text_length: Optional[int] = None, **kwargs):
        """
        EmbeddingError 초기화
        
        Args:
            message: 오류 메시지
            text_length: 처리하려던 텍스트 길이 (선택사항)
            **kwargs: 추가 인자들
        """
        super().__init__(message, **kwargs)
        self.text_length = text_length
        if text_length is not None:
            self.context['text_length'] = text_length