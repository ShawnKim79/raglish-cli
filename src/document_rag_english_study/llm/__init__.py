"""
LLM (Large Language Model) 모듈.

이 모듈은 다양한 언어 모델 제공업체들과의 통합을 위한 
추상화 레이어와 구현체들을 포함합니다.
"""

from typing import Optional
from ..models.config import LLMConfig

from .base import (
    LanguageModel,
    LanguageModelError,
    APIConnectionError,
    AuthenticationError,
    RateLimitError,
    MockLanguageModel
)
from .openai_model import OpenAILanguageModel
from .gemini_model import GeminiLanguageModel
from .ollama_model import OllamaLanguageModel


def create_language_model(llm_config: LLMConfig) -> LanguageModel:
    """LLM 설정을 기반으로 언어 모델 인스턴스를 생성합니다.
    
    Args:
        llm_config: LLM 설정 객체
        
    Returns:
        LanguageModel: 생성된 언어 모델 인스턴스
        
    Raises:
        ValueError: 지원되지 않는 제공업체인 경우
        LanguageModelError: 모델 초기화 실패 시
    """
    if not llm_config:
        raise ValueError("LLM 설정이 제공되지 않았습니다.")
    
    provider = llm_config.provider.lower()
    
    try:
        if provider == 'openai':
            if not llm_config.api_key:
                raise ValueError("OpenAI API 키가 필요합니다.")
            
            return OpenAILanguageModel(
                api_key=llm_config.api_key,
                model=llm_config.model_name,
                temperature=llm_config.temperature,
                max_tokens=llm_config.max_tokens
            )
        
        elif provider == 'gemini':
            if not llm_config.api_key:
                raise ValueError("Gemini API 키가 필요합니다.")
            
            return GeminiLanguageModel(
                model_name=llm_config.model_name,
                api_key=llm_config.api_key,
                temperature=llm_config.temperature,
                max_output_tokens=llm_config.max_tokens
            )
        
        elif provider == 'ollama':
            host = getattr(llm_config, 'host', 'localhost:11434')
            
            return OllamaLanguageModel(
                model=llm_config.model_name,
                host=host,
                temperature=llm_config.temperature,
                max_tokens=llm_config.max_tokens
            )
        
        else:
            raise ValueError(f"지원되지 않는 LLM 제공업체: {provider}")
    
    except Exception as e:
        raise LanguageModelError(f"언어 모델 생성 실패 ({provider}): {e}")


__all__ = [
    'LanguageModel',
    'LanguageModelError',
    'APIConnectionError', 
    'AuthenticationError',
    'RateLimitError',
    'MockLanguageModel',
    'OpenAILanguageModel',
    'GeminiLanguageModel',
    'OllamaLanguageModel',
    'create_language_model'
]