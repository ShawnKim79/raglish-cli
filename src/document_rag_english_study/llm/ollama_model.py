"""
Ollama 로컬 언어 모델 구현체.

이 모듈은 Ollama 로컬 서버를 사용하는 언어 모델 구현을 제공합니다.
"""

import os
import json
import logging
import requests
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


class OllamaLanguageModel(LanguageModel):
    """Ollama 로컬 서버를 사용하는 언어 모델 구현체.
    
    Llama2, Mistral, CodeLlama 등의 로컬 모델들을 지원합니다.
    """
    
    def __init__(self, model_name: str = "llama2", host: str = "localhost:11434", **kwargs):
        """Ollama 언어 모델 초기화.
        
        Args:
            model_name: 사용할 Ollama 모델명 (기본값: "llama2")
            host: Ollama 서버 주소 (기본값: "localhost:11434")
            **kwargs: 추가 설정 (temperature, num_predict 등)
        """
        super().__init__(model_name, **kwargs)
        self.host = host
        self.base_url = f"http://{host}"
        self.temperature = kwargs.get('temperature', 0.7)
        self.num_predict = kwargs.get('num_predict', 1000)
        self.timeout = kwargs.get('timeout', 30)
        
        # 일반적인 Ollama 모델 목록
        self.common_models = [
            'llama2',
            'llama2:7b',
            'llama2:13b',
            'llama2:70b',
            'mistral',
            'mistral:7b',
            'codellama',
            'codellama:7b',
            'phi',
            'neural-chat',
            'starling-lm',
            'openchat',
            'wizardlm2'
        ]
    
    def initialize(self) -> None:
        """Ollama 서버 연결을 확인하고 모델 가용성을 검증합니다.
        
        Raises:
            APIConnectionError: Ollama 서버 연결 실패 시
            LanguageModelError: 모델을 찾을 수 없거나 기타 초기화 오류 시
        """
        try:
            # Ollama 서버 연결 확인
            self._check_server_connection()
            
            # 사용 가능한 모델 목록 확인
            available_models = self._get_available_models()
            
            # 요청한 모델이 사용 가능한지 확인
            if not self._is_model_available(available_models):
                # 모델이 없으면 다운로드 시도
                logger.info(f"Model {self.model_name} not found. Attempting to pull...")
                self._pull_model()
                
                # 다시 확인
                available_models = self._get_available_models()
                if not self._is_model_available(available_models):
                    raise LanguageModelError(
                        f"Model {self.model_name} is not available and could not be pulled. "
                        f"Available models: {[m['name'] for m in available_models]}"
                    )
            
            # 연결 테스트 - 간단한 요청으로 모델 동작 확인
            test_response = self._generate_completion("Hello", max_tokens=5)
            
            if not test_response or 'response' not in test_response:
                raise APIConnectionError("Failed to get response from Ollama server")
            
            self._is_initialized = True
            logger.info(f"Ollama model {self.model_name} initialized successfully")
            
        except (APIConnectionError, LanguageModelError):
            # 이미 처리된 예외는 다시 발생
            raise
        except Exception as e:
            self._handle_api_error(e, "initialize")
    
    def generate_response(self, prompt: str, context: str = "", **kwargs) -> LLMResponse:
        """Ollama를 사용하여 응답을 생성합니다.
        
        Args:
            prompt: 사용자 프롬프트
            context: 추가 컨텍스트 정보
            **kwargs: 추가 매개변수 (temperature, num_predict 등)
            
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
            
            # API 호출 매개변수 설정
            max_tokens = kwargs.get('num_predict', kwargs.get('max_tokens', self.num_predict))
            temperature = kwargs.get('temperature', self.temperature)
            
            # API 호출
            response_data = self._generate_completion(
                full_prompt,
                temperature=temperature,
                max_tokens=max_tokens
            )
            
            # 응답 처리
            content = response_data.get('response', '')
            if not content:
                raise LanguageModelError("Empty response from Ollama server")
            
            # 사용량 정보 계산
            usage = self._calculate_usage(full_prompt, content, response_data)
            
            # 메타데이터 추출
            metadata = self._extract_metadata(response_data)
            
            return LLMResponse(
                content=content,
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
    
    def _check_server_connection(self) -> None:
        """Ollama 서버 연결을 확인합니다.
        
        Raises:
            APIConnectionError: 서버 연결 실패 시
        """
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=self.timeout)
            response.raise_for_status()
        except requests.exceptions.ConnectionError:
            raise APIConnectionError(
                f"Cannot connect to Ollama server at {self.base_url}. "
                "Please ensure Ollama is running."
            )
        except requests.exceptions.Timeout:
            raise APIConnectionError(
                f"Timeout connecting to Ollama server at {self.base_url}"
            )
        except requests.exceptions.RequestException as e:
            raise APIConnectionError(f"Error connecting to Ollama server: {e}")
    
    def _get_available_models(self) -> List[Dict[str, Any]]:
        """사용 가능한 모델 목록을 가져옵니다.
        
        Returns:
            List[Dict[str, Any]]: 사용 가능한 모델 목록
            
        Raises:
            APIConnectionError: API 호출 실패 시
        """
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=self.timeout)
            response.raise_for_status()
            
            data = response.json()
            return data.get('models', [])
            
        except requests.exceptions.RequestException as e:
            raise APIConnectionError(f"Failed to get model list: {e}")
        except json.JSONDecodeError as e:
            raise APIConnectionError(f"Invalid JSON response from Ollama server: {e}")
    
    def _is_model_available(self, available_models: List[Dict[str, Any]]) -> bool:
        """모델이 사용 가능한지 확인합니다.
        
        Args:
            available_models: 사용 가능한 모델 목록
            
        Returns:
            bool: 모델이 사용 가능하면 True
        """
        model_names = [model.get('name', '') for model in available_models]
        
        # 정확한 이름 매치
        if self.model_name in model_names:
            return True
        
        # 태그가 없는 경우 기본 태그 추가해서 확인
        if ':' not in self.model_name:
            default_tagged = f"{self.model_name}:latest"
            if default_tagged in model_names:
                return True
        
        return False
    
    def _pull_model(self) -> None:
        """모델을 다운로드합니다.
        
        Raises:
            LanguageModelError: 모델 다운로드 실패 시
        """
        try:
            logger.info(f"Pulling model {self.model_name}...")
            
            response = requests.post(
                f"{self.base_url}/api/pull",
                json={"name": self.model_name},
                timeout=300,  # 모델 다운로드는 시간이 오래 걸릴 수 있음
                stream=True
            )
            response.raise_for_status()
            
            # 스트리밍 응답 처리
            for line in response.iter_lines():
                if line:
                    try:
                        data = json.loads(line)
                        if 'status' in data:
                            logger.info(f"Pull status: {data['status']}")
                        if data.get('error'):
                            raise LanguageModelError(f"Model pull error: {data['error']}")
                    except json.JSONDecodeError:
                        continue
            
            logger.info(f"Model {self.model_name} pulled successfully")
            
        except requests.exceptions.RequestException as e:
            raise LanguageModelError(f"Failed to pull model {self.model_name}: {e}")
    
    def _generate_completion(self, prompt: str, temperature: float = None, max_tokens: int = None) -> Dict[str, Any]:
        """Ollama API를 사용하여 텍스트 완성을 생성합니다.
        
        Args:
            prompt: 입력 프롬프트
            temperature: 생성 온도
            max_tokens: 최대 토큰 수
            
        Returns:
            Dict[str, Any]: API 응답 데이터
            
        Raises:
            APIConnectionError: API 호출 실패 시
        """
        try:
            payload = {
                "model": self.model_name,
                "prompt": prompt,
                "stream": False,
                "options": {}
            }
            
            if temperature is not None:
                payload["options"]["temperature"] = temperature
            
            if max_tokens is not None:
                payload["options"]["num_predict"] = max_tokens
            
            response = requests.post(
                f"{self.base_url}/api/generate",
                json=payload,
                timeout=self.timeout
            )
            response.raise_for_status()
            
            return response.json()
            
        except requests.exceptions.RequestException as e:
            raise APIConnectionError(f"Ollama API request failed: {e}")
        except json.JSONDecodeError as e:
            raise APIConnectionError(f"Invalid JSON response from Ollama: {e}")
    
    def _calculate_usage(self, prompt: str, response_text: str, response_data: Dict[str, Any]) -> Dict[str, int]:
        """사용량 정보를 계산합니다.
        
        Args:
            prompt: 입력 프롬프트
            response_text: 응답 텍스트
            response_data: Ollama API 응답 데이터
            
        Returns:
            Dict[str, int]: 토큰 사용량 정보
        """
        # Ollama 응답에서 실제 토큰 수 확인
        if 'prompt_eval_count' in response_data and 'eval_count' in response_data:
            return {
                'prompt_tokens': response_data['prompt_eval_count'],
                'completion_tokens': response_data['eval_count'],
                'total_tokens': response_data['prompt_eval_count'] + response_data['eval_count']
            }
        
        # 근사치 계산 (단어 기반)
        prompt_tokens = len(prompt.split())
        completion_tokens = len(response_text.split())
        
        return {
            'prompt_tokens': prompt_tokens,
            'completion_tokens': completion_tokens,
            'total_tokens': prompt_tokens + completion_tokens
        }
    
    def _extract_metadata(self, response_data: Dict[str, Any]) -> Dict[str, Any]:
        """응답에서 메타데이터를 추출합니다.
        
        Args:
            response_data: Ollama API 응답 데이터
            
        Returns:
            Dict[str, Any]: 메타데이터
        """
        metadata = {}
        
        # 성능 정보
        if 'total_duration' in response_data:
            metadata['total_duration'] = response_data['total_duration']
        
        if 'load_duration' in response_data:
            metadata['load_duration'] = response_data['load_duration']
        
        if 'prompt_eval_duration' in response_data:
            metadata['prompt_eval_duration'] = response_data['prompt_eval_duration']
        
        if 'eval_duration' in response_data:
            metadata['eval_duration'] = response_data['eval_duration']
        
        # 완료 상태
        if 'done' in response_data:
            metadata['done'] = response_data['done']
        
        # 컨텍스트 정보
        if 'context' in response_data:
            metadata['context_length'] = len(response_data['context'])
        
        return metadata
    
    def _handle_api_error(self, error: Exception, method: str) -> None:
        """Ollama API 특화 오류 처리.
        
        Args:
            error: 발생한 원본 오류
            method: 오류가 발생한 메서드명
        """
        logger.error(f"Ollama API error in {method}: {error}")
        
        # Ollama 특화 오류 처리
        error_message = str(error).lower()
        
        if "connection" in error_message or "timeout" in error_message:
            raise APIConnectionError(
                f"Ollama server connection error: {error}. "
                "Please ensure Ollama is running and accessible."
            )
        elif "model" in error_message and ("not found" in error_message or "not available" in error_message):
            raise LanguageModelError(
                f"Ollama model not found: {self.model_name}. "
                f"Try running: ollama pull {self.model_name}"
            )
        elif "out of memory" in error_message or "memory" in error_message:
            raise LanguageModelError(
                f"Insufficient memory to run model {self.model_name}: {error}"
            )
        else:
            raise LanguageModelError(f"Ollama API error in {method}: {error}")
    
    def get_model_info(self) -> Dict[str, Any]:
        """Ollama 모델 정보를 반환합니다."""
        info = super().get_model_info()
        info.update({
            'provider': 'Ollama',
            'host': self.host,
            'base_url': self.base_url,
            'common_models': self.common_models,
            'temperature': self.temperature,
            'num_predict': self.num_predict,
            'timeout': self.timeout
        })
        
        # 서버가 초기화되었다면 사용 가능한 모델 목록도 포함
        if self._is_initialized:
            try:
                available_models = self._get_available_models()
                info['available_models'] = [model.get('name', '') for model in available_models]
            except:
                # 오류가 발생해도 기본 정보는 반환
                pass
        
        return info
    
    def list_available_models(self) -> List[str]:
        """사용 가능한 모델 목록을 반환합니다.
        
        Returns:
            List[str]: 사용 가능한 모델명 목록
            
        Raises:
            LanguageModelError: 서버가 초기화되지 않았거나 모델 목록을 가져올 수 없는 경우
        """
        if not self._is_initialized:
            # 초기화되지 않았어도 서버 연결만 확인하고 모델 목록 가져오기 시도
            try:
                self._check_server_connection()
                available_models = self._get_available_models()
                return [model.get('name', '') for model in available_models]
            except Exception as e:
                raise LanguageModelError(f"Cannot get model list: {e}")
        
        try:
            available_models = self._get_available_models()
            return [model.get('name', '') for model in available_models]
        except Exception as e:
            raise LanguageModelError(f"Failed to get available models: {e}")
    
    def is_server_running(self) -> bool:
        """Ollama 서버가 실행 중인지 확인합니다.
        
        Returns:
            bool: 서버가 실행 중이면 True
        """
        try:
            self._check_server_connection()
            return True
        except APIConnectionError:
            return False