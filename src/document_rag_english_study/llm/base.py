"""
언어 모델 추상 베이스 클래스.

이 모듈은 다양한 LLM 제공업체들을 위한 공통 인터페이스를 정의합니다.
"""

from abc import ABC, abstractmethod
from typing import Optional, Dict, Any, List
import logging

from ..models.llm import EnglishAnalysis, LLMResponse


logger = logging.getLogger(__name__)


class LanguageModelError(Exception):
    """언어 모델 관련 오류를 나타내는 예외 클래스."""
    pass


class APIConnectionError(LanguageModelError):
    """API 연결 오류를 나타내는 예외 클래스."""
    pass


class AuthenticationError(LanguageModelError):
    """인증 오류를 나타내는 예외 클래스."""
    pass


class RateLimitError(LanguageModelError):
    """API 요청 한도 초과 오류를 나타내는 예외 클래스."""
    pass


class LanguageModel(ABC):
    """언어 모델을 위한 추상 베이스 클래스.
    
    이 클래스는 모든 언어 모델 구현체가 따라야 하는 공통 인터페이스를 정의합니다.
    OpenAI GPT, Google Gemini, Ollama 등 다양한 제공업체의 모델들이 
    이 인터페이스를 구현해야 합니다.
    """
    
    def __init__(self, model_name: str, **kwargs):
        """언어 모델 초기화.
        
        Args:
            model_name: 사용할 모델의 이름
            **kwargs: 모델별 추가 설정 매개변수
        """
        self.model_name = model_name
        self.config = kwargs
        self._is_initialized = False
        
    @abstractmethod
    def initialize(self) -> None:
        """모델을 초기화합니다.
        
        API 키 검증, 연결 테스트 등을 수행합니다.
        
        Raises:
            AuthenticationError: 인증 실패 시
            APIConnectionError: 연결 실패 시
            LanguageModelError: 기타 초기화 오류 시
        """
        pass
    
    @abstractmethod
    def generate_response(self, prompt: str, context: str = "", **kwargs) -> LLMResponse:
        """주어진 프롬프트와 컨텍스트를 바탕으로 응답을 생성합니다.
        
        Args:
            prompt: 사용자 프롬프트
            context: 추가 컨텍스트 정보 (RAG에서 검색된 문서 내용 등)
            **kwargs: 모델별 추가 매개변수 (temperature, max_tokens 등)
            
        Returns:
            LLMResponse: 생성된 응답
            
        Raises:
            LanguageModelError: 응답 생성 실패 시
            RateLimitError: API 요청 한도 초과 시
            APIConnectionError: 연결 오류 시
        """
        pass
    
    @abstractmethod
    def translate_text(self, text: str, target_language: str, source_language: str = "auto") -> str:
        """텍스트를 지정된 언어로 번역합니다.
        
        Args:
            text: 번역할 텍스트
            target_language: 목표 언어 (예: "korean", "english")
            source_language: 원본 언어 (기본값: "auto" - 자동 감지)
            
        Returns:
            str: 번역된 텍스트
            
        Raises:
            LanguageModelError: 번역 실패 시
        """
        pass
    
    @abstractmethod
    def analyze_grammar(self, text: str, user_language: str = "korean") -> EnglishAnalysis:
        """영어 텍스트의 문법을 분석하고 피드백을 제공합니다.
        
        Args:
            text: 분석할 영어 텍스트
            user_language: 사용자의 모국어 (설명에 사용)
            
        Returns:
            EnglishAnalysis: 문법 분석 결과
            
        Raises:
            LanguageModelError: 분석 실패 시
        """
        pass
    
    def is_available(self) -> bool:
        """모델이 사용 가능한 상태인지 확인합니다.
        
        Returns:
            bool: 사용 가능하면 True, 그렇지 않으면 False
        """
        return self._is_initialized
    
    def get_model_info(self) -> Dict[str, Any]:
        """모델 정보를 반환합니다.
        
        Returns:
            Dict[str, Any]: 모델 이름, 제공업체, 설정 등의 정보
        """
        return {
            'model_name': self.model_name,
            'provider': self.__class__.__name__,
            'initialized': self._is_initialized,
            'config': self.config
        }
    
    def validate_input(self, text: str, max_length: int = 10000) -> None:
        """입력 텍스트를 검증합니다.
        
        Args:
            text: 검증할 텍스트
            max_length: 최대 허용 길이
            
        Raises:
            ValueError: 입력이 유효하지 않은 경우
        """
        if not isinstance(text, str):
            raise ValueError("Input must be a string")
        
        if not text.strip():
            raise ValueError("Input text cannot be empty")
        
        if len(text) > max_length:
            raise ValueError(f"Input text too long. Maximum {max_length} characters allowed")
    
    def _log_api_call(self, method: str, **kwargs) -> None:
        """API 호출을 로깅합니다.
        
        Args:
            method: 호출된 메서드명
            **kwargs: 로깅할 추가 정보
        """
        logger.info(f"API call: {self.__class__.__name__}.{method}", extra=kwargs)
    
    def _handle_api_error(self, error: Exception, method: str) -> None:
        """API 오류를 처리하고 적절한 예외를 발생시킵니다.
        
        Args:
            error: 발생한 원본 오류
            method: 오류가 발생한 메서드명
            
        Raises:
            LanguageModelError: 처리된 오류
        """
        logger.error(f"API error in {self.__class__.__name__}.{method}: {error}")
        
        # 일반적인 오류 유형별 처리
        error_message = str(error).lower()
        
        if "authentication" in error_message or "unauthorized" in error_message:
            raise AuthenticationError(f"Authentication failed: {error}")
        elif "rate limit" in error_message or "quota" in error_message:
            raise RateLimitError(f"Rate limit exceeded: {error}")
        elif "connection" in error_message or "network" in error_message:
            raise APIConnectionError(f"Connection error: {error}")
        else:
            raise LanguageModelError(f"API error in {method}: {error}")


class MockLanguageModel(LanguageModel):
    """테스트용 모의 언어 모델 구현체.
    
    실제 API 호출 없이 테스트를 위한 더미 응답을 제공합니다.
    """
    
    def __init__(self, model_name: str = "mock-model", **kwargs):
        """모의 모델 초기화."""
        super().__init__(model_name, **kwargs)
        self.call_count = 0
    
    def initialize(self) -> None:
        """모의 초기화."""
        self._is_initialized = True
        logger.info("Mock language model initialized")
    
    def generate_response(self, prompt: str, context: str = "", **kwargs) -> LLMResponse:
        """모의 응답 생성."""
        self.validate_input(prompt)
        self.call_count += 1
        
        response_content = f"Mock response to: {prompt[:50]}..."
        if context:
            response_content += f" (with context: {context[:30]}...)"
        
        return LLMResponse(
            content=response_content,
            model=self.model_name,
            usage={'prompt_tokens': len(prompt.split()), 'completion_tokens': 10},
            metadata={'call_count': self.call_count}
        )
    
    def translate_text(self, text: str, target_language: str, source_language: str = "auto") -> str:
        """모의 번역."""
        self.validate_input(text)
        return f"[Mock translation to {target_language}]: {text}"
    
    def analyze_grammar(self, text: str, user_language: str = "korean") -> EnglishAnalysis:
        """모의 문법 분석."""
        self.validate_input(text)
        
        from ..models.llm import GrammarError, ErrorType, ImprovementSuggestion
        
        # 간단한 모의 분석 결과 생성
        analysis = EnglishAnalysis(
            vocabulary_level="intermediate",
            fluency_score=0.8,
            complexity_score=0.6
        )
        
        # 모의 문법 오류 추가 (단어 수가 많으면)
        if len(text.split()) > 10:
            analysis.grammar_errors.append(
                GrammarError(
                    text="example error",
                    error_type=ErrorType.GRAMMAR,
                    position=(0, 7),
                    suggestion="corrected example",
                    explanation="Mock grammar explanation"
                )
            )
        
        # 모의 개선 제안 추가
        analysis.suggestions.append(
            ImprovementSuggestion(
                category="vocabulary",
                original="good",
                improved="excellent",
                reason="More sophisticated vocabulary choice",
                confidence=0.9
            )
        )
        
        return analysis