"""
OpenAI GPT 언어 모델 구현체.

이 모듈은 OpenAI GPT API를 사용하는 언어 모델 구현을 제공합니다.
"""

import os
import json
import logging
from typing import Optional, Dict, Any, List
import time

from .base import (
    LanguageModel, 
    LanguageModelError, 
    APIConnectionError, 
    AuthenticationError, 
    RateLimitError
)
from ..models.llm import EnglishAnalysis, LLMResponse, GrammarError, ErrorType, ImprovementSuggestion


logger = logging.getLogger(__name__)


class OpenAILanguageModel(LanguageModel):
    """OpenAI GPT API를 사용하는 언어 모델 구현체.
    
    GPT-3.5-turbo, GPT-4 등의 OpenAI 모델들을 지원합니다.
    """
    
    def __init__(self, model_name: str = "gpt-3.5-turbo", api_key: Optional[str] = None, **kwargs):
        """OpenAI 언어 모델 초기화.
        
        Args:
            model_name: 사용할 OpenAI 모델명 (기본값: "gpt-3.5-turbo")
            api_key: OpenAI API 키 (None이면 환경변수에서 가져옴)
            **kwargs: 추가 설정 (temperature, max_tokens 등)
        """
        super().__init__(model_name, **kwargs)
        self.api_key = api_key or os.getenv('OPENAI_API_KEY')
        self.base_url = kwargs.get('base_url', 'https://api.openai.com/v1')
        self.temperature = kwargs.get('temperature', 0.7)
        self.max_tokens = kwargs.get('max_tokens', 1000)
        self.client = None
        
        # 지원되는 모델 목록
        self.supported_models = [
            'gpt-3.5-turbo',
            'gpt-3.5-turbo-16k',
            'gpt-4',
            'gpt-4-turbo-preview',
            'gpt-4o',
            'gpt-4o-mini'
        ]
    
    def initialize(self) -> None:
        """OpenAI 클라이언트를 초기화하고 연결을 테스트합니다.
        
        Raises:
            AuthenticationError: API 키가 없거나 유효하지 않은 경우
            APIConnectionError: API 연결 실패 시
            LanguageModelError: 기타 초기화 오류 시
        """
        if not self.api_key:
            raise AuthenticationError("OpenAI API key is required")
        
        try:
            # OpenAI 클라이언트 임포트 및 초기화
            try:
                from openai import OpenAI
            except ImportError:
                raise LanguageModelError(
                    "OpenAI package not installed. Please install with: pip install openai"
                )
            
            self.client = OpenAI(
                api_key=self.api_key,
                base_url=self.base_url
            )
            
            # 연결 테스트 - 간단한 요청으로 API 키 유효성 확인
            test_response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": "Hello"}],
                max_tokens=5
            )
            
            if not test_response:
                raise APIConnectionError("Failed to get response from OpenAI API")
            
            self._is_initialized = True
            logger.info(f"OpenAI model {self.model_name} initialized successfully")
            
        except Exception as e:
            self._handle_api_error(e, "initialize")
    
    def generate_response(self, prompt: str, context: str = "", **kwargs) -> LLMResponse:
        """OpenAI GPT를 사용하여 응답을 생성합니다.
        
        Args:
            prompt: 사용자 프롬프트
            context: 추가 컨텍스트 정보
            **kwargs: 추가 매개변수 (temperature, max_tokens 등)
            
        Returns:
            LLMResponse: 생성된 응답
        """
        if not self._is_initialized:
            raise LanguageModelError("Model not initialized. Call initialize() first.")
        
        self.validate_input(prompt)
        self._log_api_call("generate_response", prompt_length=len(prompt))
        
        try:
            # 메시지 구성
            messages = []
            
            if context:
                messages.append({
                    "role": "system",
                    "content": f"다음 컨텍스트를 참고하여 답변해주세요:\n\n{context}"
                })
            
            messages.append({
                "role": "user", 
                "content": prompt
            })
            
            # API 호출 매개변수 설정
            api_params = {
                "model": self.model_name,
                "messages": messages,
                "temperature": kwargs.get('temperature', self.temperature),
                "max_tokens": kwargs.get('max_tokens', self.max_tokens)
            }
            
            # API 호출
            response = self.client.chat.completions.create(**api_params)
            
            # 응답 처리
            content = response.choices[0].message.content
            usage = {
                'prompt_tokens': response.usage.prompt_tokens,
                'completion_tokens': response.usage.completion_tokens,
                'total_tokens': response.usage.total_tokens
            }
            
            return LLMResponse(
                content=content,
                model=self.model_name,
                usage=usage,
                metadata={
                    'finish_reason': response.choices[0].finish_reason,
                    'response_id': response.id
                }
            )
            
        except Exception as e:
            self._handle_api_error(e, "generate_response")
    
    def translate_text(self, text: str, target_language: str, source_language: str = "auto") -> str:
        """텍스트를 지정된 언어로 번역합니다.
        
        Args:
            text: 번역할 텍스트
            target_language: 목표 언어
            source_language: 원본 언어 (기본값: "auto")
            
        Returns:
            str: 번역된 텍스트
        """
        if not self._is_initialized:
            raise LanguageModelError("Model not initialized. Call initialize() first.")
        
        self.validate_input(text)
        self._log_api_call("translate_text", target_language=target_language)
        
        # 번역 프롬프트 구성
        if source_language == "auto":
            prompt = f"""다음 텍스트를 {target_language}로 번역해주세요. 자연스럽고 정확한 번역을 제공해주세요.

텍스트: {text}

번역:"""
        else:
            prompt = f"""다음 {source_language} 텍스트를 {target_language}로 번역해주세요. 자연스럽고 정확한 번역을 제공해주세요.

텍스트: {text}

번역:"""
        
        try:
            response = self.generate_response(prompt)
            return response.content.strip()
        except Exception as e:
            self._handle_api_error(e, "translate_text")
    
    def analyze_grammar(self, text: str, user_language: str = "korean") -> EnglishAnalysis:
        """영어 텍스트의 문법을 분석하고 피드백을 제공합니다.
        
        Args:
            text: 분석할 영어 텍스트
            user_language: 사용자의 모국어
            
        Returns:
            EnglishAnalysis: 문법 분석 결과
        """
        if not self._is_initialized:
            raise LanguageModelError("Model not initialized. Call initialize() first.")
        
        self.validate_input(text)
        self._log_api_call("analyze_grammar", text_length=len(text))
        
        # 문법 분석 프롬프트 구성
        prompt = f"""다음 영어 텍스트를 분석하고 JSON 형식으로 결과를 제공해주세요. 
사용자의 모국어는 {user_language}입니다.

분석할 텍스트: "{text}"

다음 JSON 형식으로 응답해주세요:
{{
    "grammar_errors": [
        {{
            "text": "오류가 있는 부분",
            "error_type": "grammar|vocabulary|spelling|punctuation|syntax",
            "position": [시작위치, 끝위치],
            "suggestion": "수정 제안",
            "explanation": "오류 설명 ({user_language}로)"
        }}
    ],
    "vocabulary_level": "beginner|intermediate|advanced",
    "fluency_score": 0.0-1.0,
    "complexity_score": 0.0-1.0,
    "suggestions": [
        {{
            "category": "vocabulary|grammar|style",
            "original": "원본 표현",
            "improved": "개선된 표현", 
            "reason": "개선 이유 ({user_language}로)",
            "confidence": 0.0-1.0
        }}
    ]
}}

JSON만 응답하고 다른 텍스트는 포함하지 마세요."""
        
        try:
            response = self.generate_response(prompt, temperature=0.3)
            
            # JSON 응답 파싱
            try:
                analysis_data = json.loads(response.content.strip())
            except json.JSONDecodeError:
                # JSON 파싱 실패 시 기본 분석 결과 반환
                logger.warning("Failed to parse grammar analysis JSON response")
                return EnglishAnalysis(
                    vocabulary_level="intermediate",
                    fluency_score=0.7,
                    complexity_score=0.5
                )
            
            # EnglishAnalysis 객체 생성
            analysis = EnglishAnalysis(
                vocabulary_level=analysis_data.get('vocabulary_level', 'intermediate'),
                fluency_score=analysis_data.get('fluency_score', 0.7),
                complexity_score=analysis_data.get('complexity_score', 0.5)
            )
            
            # 문법 오류 추가
            for error_data in analysis_data.get('grammar_errors', []):
                try:
                    error = GrammarError(
                        text=error_data['text'],
                        error_type=ErrorType(error_data['error_type']),
                        position=tuple(error_data['position']),
                        suggestion=error_data['suggestion'],
                        explanation=error_data['explanation']
                    )
                    analysis.grammar_errors.append(error)
                except (KeyError, ValueError) as e:
                    logger.warning(f"Invalid grammar error data: {e}")
                    continue
            
            # 개선 제안 추가
            for suggestion_data in analysis_data.get('suggestions', []):
                try:
                    suggestion = ImprovementSuggestion(
                        category=suggestion_data['category'],
                        original=suggestion_data['original'],
                        improved=suggestion_data['improved'],
                        reason=suggestion_data['reason'],
                        confidence=suggestion_data.get('confidence', 0.8)
                    )
                    analysis.suggestions.append(suggestion)
                except (KeyError, ValueError) as e:
                    logger.warning(f"Invalid suggestion data: {e}")
                    continue
            
            return analysis
            
        except Exception as e:
            self._handle_api_error(e, "analyze_grammar")
    
    def _handle_api_error(self, error: Exception, method: str) -> None:
        """OpenAI API 특화 오류 처리.
        
        Args:
            error: 발생한 원본 오류
            method: 오류가 발생한 메서드명
        """
        logger.error(f"OpenAI API error in {method}: {error}")
        
        # OpenAI 특화 오류 처리
        error_message = str(error).lower()
        
        if "invalid api key" in error_message or "unauthorized" in error_message:
            raise AuthenticationError(f"Invalid OpenAI API key: {error}")
        elif "rate limit" in error_message or "quota exceeded" in error_message:
            raise RateLimitError(f"OpenAI API rate limit exceeded: {error}")
        elif "connection" in error_message or "timeout" in error_message:
            raise APIConnectionError(f"OpenAI API connection error: {error}")
        elif "model" in error_message and "not found" in error_message:
            raise LanguageModelError(f"OpenAI model not found: {self.model_name}")
        else:
            raise LanguageModelError(f"OpenAI API error in {method}: {error}")
    
    def get_model_info(self) -> Dict[str, Any]:
        """OpenAI 모델 정보를 반환합니다."""
        info = super().get_model_info()
        info.update({
            'provider': 'OpenAI',
            'api_base': self.base_url,
            'supported_models': self.supported_models,
            'has_api_key': bool(self.api_key),
            'temperature': self.temperature,
            'max_tokens': self.max_tokens
        })
        return info