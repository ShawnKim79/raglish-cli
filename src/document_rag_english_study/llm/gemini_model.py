"""
Google Gemini 언어 모델 구현체.

이 모듈은 Google Gemini API를 사용하는 언어 모델 구현을 제공합니다.
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


class GeminiLanguageModel(LanguageModel):
    """Google Gemini API를 사용하는 언어 모델 구현체.
    
    Gemini Pro, Gemini Pro Vision 등의 Google 모델들을 지원합니다.
    """
    
    def __init__(self, model_name: str = "gemini-pro", api_key: Optional[str] = None, **kwargs):
        """Gemini 언어 모델 초기화.
        
        Args:
            model_name: 사용할 Gemini 모델명 (기본값: "gemini-pro")
            api_key: Google API 키 (None이면 환경변수에서 가져옴)
            **kwargs: 추가 설정 (temperature, max_output_tokens 등)
        """
        super().__init__(model_name, **kwargs)
        self.api_key = api_key or os.getenv('GOOGLE_API_KEY')
        self.temperature = kwargs.get('temperature', 0.7)
        self.max_output_tokens = kwargs.get('max_output_tokens', 1000)
        self.client = None
        
        # 지원되는 모델 목록 (기본값 - API에서 동적으로 가져올 예정)
        self.supported_models = [
            'gemini-2.5-flash',
            'gemini-2.5-pro', 
            'gemini-2.0-flash',
            'gemini-pro-latest',
            'gemini-pro',
            'gemini-pro-vision'
        ]
    
    def initialize(self) -> None:
        """Gemini 클라이언트를 초기화하고 연결을 테스트합니다."""
        if not self.api_key:
            raise AuthenticationError("Google API key is required")
        
        try:
            # Google Generative AI 클라이언트 임포트 및 초기화
            try:
                import google.generativeai as genai
                self.genai = genai
            except ImportError:
                raise LanguageModelError(
                    "Google Generative AI package not installed. "
                    "Please install with: uv add google-generativeai"
                )
            
            # API 키 설정
            genai.configure(api_key=self.api_key)
            
            # 사용 가능한 모델 목록 가져오기
            try:
                available_models = self._get_available_models()
                logger.info(f"사용 가능한 Gemini 모델: {available_models}")
                
                # 요청한 모델이 사용 가능한지 확인
                if self.model_name not in available_models:
                    logger.warning(f"요청한 모델 '{self.model_name}'이 사용 가능한 모델 목록에 없습니다.")
                    logger.warning(f"사용 가능한 모델: {available_models}")
                    
                    # 대체 모델 제안
                    if 'gemini-1.5-pro' in available_models:
                        suggested_model = 'gemini-1.5-pro'
                    elif 'gemini-pro' in available_models:
                        suggested_model = 'gemini-pro'
                    else:
                        suggested_model = available_models[0] if available_models else None
                    
                    if suggested_model:
                        logger.info(f"대체 모델로 '{suggested_model}' 사용을 권장합니다.")
                        # 자동으로 대체하지 않고 사용자에게 알림만 제공
                
            except Exception as e:
                logger.warning(f"사용 가능한 모델 목록을 가져올 수 없습니다: {e}")
                logger.info("기본 모델 목록을 사용합니다.")
            
            # 모델 초기화
            self.client = genai.GenerativeModel(self.model_name)
            
            # 연결 테스트 (간단한 프롬프트로)
            try:
                test_response = self.client.generate_content(
                    "Test connection",
                    generation_config={'max_output_tokens': 10}
                )
                
                # 응답 검증
                if not test_response:
                    raise APIConnectionError("No response from Gemini API during initialization")
                
                # 안전성 필터로 인한 차단 확인
                if hasattr(test_response, 'prompt_feedback') and test_response.prompt_feedback:
                    if hasattr(test_response.prompt_feedback, 'block_reason'):
                        logger.warning(f"Test prompt blocked: {test_response.prompt_feedback.block_reason}")
                
            except Exception as test_error:
                raise APIConnectionError(f"Failed to connect to Gemini API: {test_error}")
            
            self._is_initialized = True
            logger.info(f"Gemini model {self.model_name} initialized successfully")
            
        except (AuthenticationError, LanguageModelError, APIConnectionError):
            # 이미 처리된 예외는 다시 발생
            raise
        except Exception as e:
            self._handle_api_error(e, "initialize")    

    def generate_response(self, prompt: str, context: str = "", **kwargs) -> LLMResponse:
        """Gemini를 사용하여 응답을 생성합니다.
        
        Args:
            prompt: 사용자 프롬프트
            context: 추가 컨텍스트 정보
            **kwargs: 추가 매개변수 (temperature, max_output_tokens 등)
            
        Returns:
            LLMResponse: 생성된 응답
        """
        if not self._is_initialized:
            raise LanguageModelError("Model not initialized. Call initialize() first.")
        
        self.validate_input(prompt)
        self._log_api_call("generate_response", prompt_length=len(prompt))
        
        try:
            # 프롬프트 구성
            full_prompt = prompt
            if context:
                full_prompt = f"다음 컨텍스트를 참고하여 답변해주세요:\n\n{context}\n\n질문: {prompt}"
            
            # 디버그: 최종 프롬프트 로깅
            logger.debug(f"Final prompt to Gemini (length: {len(full_prompt)}): {full_prompt[:300]}...")
            
            # 생성 설정
            generation_config = {
                'temperature': kwargs.get('temperature', self.temperature),
                'max_output_tokens': kwargs.get('max_output_tokens', self.max_output_tokens),
            }
            
            # 안전성 설정 (최대한 관대하게)
            safety_settings = [
                {
                    "category": "HARM_CATEGORY_HARASSMENT",
                    "threshold": "BLOCK_NONE"
                },
                {
                    "category": "HARM_CATEGORY_HATE_SPEECH", 
                    "threshold": "BLOCK_NONE"
                },
                {
                    "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
                    "threshold": "BLOCK_NONE"
                },
                {
                    "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
                    "threshold": "BLOCK_NONE"
                }
            ]
            
            # API 호출
            response = self.client.generate_content(
                full_prompt,
                generation_config=generation_config,
                safety_settings=safety_settings
            )
            
            # 응답 검증
            if not response:
                raise LanguageModelError("No response from Gemini API")
            
            # 프롬프트 피드백 확인 (입력 차단)
            if hasattr(response, 'prompt_feedback') and response.prompt_feedback:
                if hasattr(response.prompt_feedback, 'block_reason') and response.prompt_feedback.block_reason:
                    raise LanguageModelError(f"Input blocked by safety filter: {response.prompt_feedback.block_reason}")
            
            # 응답 후보 확인
            if not hasattr(response, 'candidates') or not response.candidates:
                raise LanguageModelError("No response candidates from Gemini API")
            
            candidate = response.candidates[0]
            
            # 후보의 완료 이유 확인
            if hasattr(candidate, 'finish_reason'):
                finish_reason = candidate.finish_reason
                if finish_reason == 2:  # SAFETY
                    raise LanguageModelError("Response blocked by safety filter. Try rephrasing your input with more neutral language.")
                elif finish_reason == 3:  # RECITATION
                    raise LanguageModelError("Response blocked due to recitation concerns.")
                elif finish_reason == 4:  # OTHER
                    raise LanguageModelError("Response generation failed for unknown reasons.")
                elif finish_reason != 1:  # 1 = STOP (정상 완료)
                    raise LanguageModelError(f"Response generation incomplete (finish_reason: {finish_reason})")
            
            # 응답 텍스트 추출
            response_text = None
            
            # 방법 1: response.text 직접 접근 (안전하게)
            try:
                if hasattr(response, 'text') and response.text:
                    response_text = response.text
            except Exception:
                pass  # text 속성 접근 실패 시 다른 방법 시도
            
            # 방법 2: candidates를 통한 접근
            if not response_text and hasattr(candidate, 'content') and candidate.content:
                if hasattr(candidate.content, 'parts') and candidate.content.parts:
                    for part in candidate.content.parts:
                        if hasattr(part, 'text') and part.text:
                            response_text = part.text
                            break
            
            # 응답 텍스트가 없는 경우
            if not response_text:
                raise LanguageModelError("No valid text content in Gemini API response")
            
            # 사용량 정보 계산
            usage = self._calculate_usage(full_prompt, response_text, response)
            
            # 메타데이터 수집
            metadata = self._extract_metadata(response)
            
            return LLMResponse(
                content=response_text,
                model=self.model_name,
                usage=usage,
                metadata=metadata
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
        
        # 문법 분석 프롬프트 구성 (단순화된 버전)
        prompt = f"""Please analyze this English text and provide feedback in JSON format:

Text: "{text}"

Respond with this JSON structure:
{{
    "grammar_errors": [
        {{
            "text": "error text",
            "error_type": "grammar",
            "suggestion": "correction",
            "explanation": "explanation in {user_language}"
        }}
    ],
    "vocabulary_level": "beginner",
    "fluency_score": 0.8,
    "suggestions": [
        {{
            "original": "original text",
            "improved": "improved text",
            "reason": "reason in {user_language}"
        }}
    ]
}}

Only respond with valid JSON."""
        
        try:
            # 디버그: 실제 사용되는 프롬프트 로깅
            logger.debug(f"Grammar analysis prompt (length: {len(prompt)}): {prompt[:200]}...")
            response = self.generate_response(prompt, temperature=0.3)
            
            # JSON 응답 파싱
            try:
                # 응답에서 JSON 부분만 추출 (마크다운 코드 블록 제거)
                content = response.content.strip()
                if content.startswith('```json'):
                    content = content[7:]
                if content.endswith('```'):
                    content = content[:-3]
                content = content.strip()
                
                analysis_data = json.loads(content)
            except json.JSONDecodeError as e:
                # JSON 파싱 실패 시 재시도 또는 기본 분석 결과 반환
                logger.warning(f"Failed to parse grammar analysis JSON response: {e}")
                logger.debug(f"Raw response: {response.content}")
                
                # 간단한 재시도 (더 명확한 프롬프트로)
                retry_prompt = f"""다음 영어 텍스트를 분석하고 반드시 유효한 JSON만 응답해주세요.

텍스트: "{text}"

JSON 형식:
{{"grammar_errors":[],"vocabulary_level":"intermediate","fluency_score":0.7,"complexity_score":0.5,"suggestions":[]}}

위 형식을 정확히 따라 JSON만 응답하세요:"""
                
                try:
                    retry_response = self.generate_response(retry_prompt, temperature=0.1)
                    retry_content = retry_response.content.strip()
                    if retry_content.startswith('```json'):
                        retry_content = retry_content[7:]
                    if retry_content.endswith('```'):
                        retry_content = retry_content[:-3]
                    analysis_data = json.loads(retry_content.strip())
                except:
                    # 재시도도 실패하면 기본값 반환
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
        """Gemini API 특화 오류 처리.
        
        Args:
            error: 발생한 원본 오류
            method: 오류가 발생한 메서드명
        """
        logger.error(f"Gemini API error in {method}: {error}")
        
        # Gemini 특화 오류 처리
        error_message = str(error).lower()
        
        if "api key" in error_message or "unauthorized" in error_message:
            raise AuthenticationError(f"Invalid Google API key: {error}")
        elif "quota" in error_message or "rate limit" in error_message:
            raise RateLimitError(f"Gemini API rate limit exceeded: {error}")
        elif "connection" in error_message or "timeout" in error_message:
            raise APIConnectionError(f"Gemini API connection error: {error}")
        elif "model" in error_message and "not found" in error_message:
            raise LanguageModelError(f"Gemini model not found: {self.model_name}")
        elif "safety" in error_message:
            raise LanguageModelError(f"Content blocked by Gemini safety filters: {error}")
        else:
            raise LanguageModelError(f"Gemini API error in {method}: {error}")
    
    def _calculate_usage(self, prompt: str, response_text: str, response) -> Dict[str, int]:
        """사용량 정보를 계산합니다.
        
        Args:
            prompt: 입력 프롬프트
            response_text: 응답 텍스트
            response: Gemini API 응답 객체
            
        Returns:
            Dict[str, int]: 토큰 사용량 정보
        """
        # Gemini API에서 실제 토큰 수를 제공하는지 확인
        if hasattr(response, 'usage_metadata'):
            return {
                'prompt_tokens': getattr(response.usage_metadata, 'prompt_token_count', 0),
                'completion_tokens': getattr(response.usage_metadata, 'candidates_token_count', 0),
                'total_tokens': getattr(response.usage_metadata, 'total_token_count', 0)
            }
        
        # 근사치 계산 (단어 기반)
        prompt_tokens = len(prompt.split())
        completion_tokens = len(response_text.split())
        
        return {
            'prompt_tokens': prompt_tokens,
            'completion_tokens': completion_tokens,
            'total_tokens': prompt_tokens + completion_tokens
        }
    
    def _extract_metadata(self, response) -> Dict[str, Any]:
        """응답에서 메타데이터를 추출합니다.
        
        Args:
            response: Gemini API 응답 객체
            
        Returns:
            Dict[str, Any]: 메타데이터
        """
        metadata = {}
        
        # 완료 이유
        if hasattr(response, 'candidates') and response.candidates:
            candidate = response.candidates[0]
            if hasattr(candidate, 'finish_reason'):
                metadata['finish_reason'] = str(candidate.finish_reason)
        
        # 안전성 등급
        if hasattr(response, 'candidates') and response.candidates:
            candidate = response.candidates[0]
            if hasattr(candidate, 'safety_ratings'):
                metadata['safety_ratings'] = [
                    {
                        'category': str(rating.category),
                        'probability': str(rating.probability)
                    }
                    for rating in candidate.safety_ratings
                ]
        
        # 프롬프트 피드백
        if hasattr(response, 'prompt_feedback'):
            metadata['prompt_feedback'] = {
                'block_reason': getattr(response.prompt_feedback, 'block_reason', None),
                'safety_ratings': [
                    {
                        'category': str(rating.category),
                        'probability': str(rating.probability)
                    }
                    for rating in getattr(response.prompt_feedback, 'safety_ratings', [])
                ]
            }
        
        return metadata
    
    def _get_available_models(self) -> List[str]:
        """Google Gemini API에서 사용 가능한 모델 목록을 가져옵니다.
        
        Returns:
            List[str]: 사용 가능한 모델명 목록
        """
        try:
            # Google Generative AI API를 통해 모델 목록 가져오기
            models = self.genai.list_models()
            
            # generateContent를 지원하는 모델만 필터링
            available_models = []
            for model in models:
                # 모델명에서 'models/' 접두사 제거
                model_name = model.name.replace('models/', '')
                
                # generateContent 메서드를 지원하는 모델만 포함
                if 'generateContent' in model.supported_generation_methods:
                    available_models.append(model_name)
            
            return available_models
            
        except Exception as e:
            logger.error(f"모델 목록 가져오기 실패: {e}")
            # 실패 시 기본 모델 목록 반환
            return self.supported_models
    
    def get_model_info(self) -> Dict[str, Any]:
        """Gemini 모델 정보를 반환합니다."""
        info = super().get_model_info()
        info.update({
            'provider': 'Google Gemini',
            'supported_models': self.supported_models,
            'has_api_key': bool(self.api_key),
            'temperature': self.temperature,
            'max_output_tokens': self.max_output_tokens
        })
        return info