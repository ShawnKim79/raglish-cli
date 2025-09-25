"""
Ollama 언어 모델 구현체 테스트.

이 모듈은 OllamaLanguageModel 클래스의 기능을 테스트합니다.
"""

import pytest
import json
import sys
import os
from unittest.mock import Mock, patch, MagicMock
import requests

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from document_rag_english_study.llm.ollama_model import OllamaLanguageModel
from document_rag_english_study.llm.base import (
    LanguageModelError,
    APIConnectionError,
    AuthenticationError
)
from document_rag_english_study.models.llm import (
    LLMResponse,
    EnglishAnalysis,
    GrammarError,
    ErrorType,
    ImprovementSuggestion
)


class TestOllamaLanguageModel:
    """OllamaLanguageModel 클래스 테스트."""
    
    def setup_method(self):
        """각 테스트 메서드 실행 전 설정."""
        self.model = OllamaLanguageModel(
            model_name="llama2",
            host="localhost:11434",
            temperature=0.7
        )
    
    def test_init(self):
        """초기화 테스트."""
        assert self.model.model_name == "llama2"
        assert self.model.host == "localhost:11434"
        assert self.model.base_url == "http://localhost:11434"
        assert self.model.temperature == 0.7
        assert not self.model.is_available()
    
    def test_init_with_custom_params(self):
        """커스텀 매개변수로 초기화 테스트."""
        model = OllamaLanguageModel(
            model_name="mistral",
            host="192.168.1.100:11434",
            temperature=0.5,
            num_predict=500,
            timeout=60
        )
        
        assert model.model_name == "mistral"
        assert model.host == "192.168.1.100:11434"
        assert model.base_url == "http://192.168.1.100:11434"
        assert model.temperature == 0.5
        assert model.num_predict == 500
        assert model.timeout == 60
    
    @patch('requests.get')
    @patch('requests.post')
    def test_initialize_success(self, mock_post, mock_get):
        """성공적인 초기화 테스트."""
        # 서버 연결 확인 모킹
        mock_get.return_value.raise_for_status.return_value = None
        mock_get.return_value.json.return_value = {
            'models': [
                {'name': 'llama2:latest'},
                {'name': 'mistral:7b'}
            ]
        }
        
        # 테스트 요청 모킹
        mock_post.return_value.raise_for_status.return_value = None
        mock_post.return_value.json.return_value = {
            'response': 'Hello!',
            'done': True
        }
        
        self.model.initialize()
        
        assert self.model.is_available()
        mock_get.assert_called_with("http://localhost:11434/api/tags", timeout=30)
        mock_post.assert_called_once()
    
    @patch('requests.get')
    def test_initialize_server_connection_error(self, mock_get):
        """서버 연결 오류 시 초기화 테스트."""
        mock_get.side_effect = requests.exceptions.ConnectionError("Connection refused")
        
        with pytest.raises(APIConnectionError) as exc_info:
            self.model.initialize()
        
        assert "Cannot connect to Ollama server" in str(exc_info.value)
        assert not self.model.is_available()
    
    @patch('requests.get')
    @patch('requests.post')
    def test_initialize_model_not_available_pull_success(self, mock_post, mock_get):
        """모델이 없을 때 자동 다운로드 성공 테스트."""
        # 첫 번째 모델 목록 조회 - 모델 없음
        # 두 번째 모델 목록 조회 - 모델 있음
        mock_get.side_effect = [
            # 서버 연결 확인
            Mock(raise_for_status=Mock(), json=Mock(return_value={'models': []})),
            # 첫 번째 모델 목록 조회
            Mock(raise_for_status=Mock(), json=Mock(return_value={'models': []})),
            # 두 번째 모델 목록 조회 (pull 후)
            Mock(raise_for_status=Mock(), json=Mock(return_value={
                'models': [{'name': 'llama2:latest'}]
            }))
        ]
        
        # 모델 pull 모킹
        pull_response = Mock()
        pull_response.raise_for_status.return_value = None
        pull_response.iter_lines.return_value = [
            b'{"status": "downloading"}',
            b'{"status": "complete"}'
        ]
        
        # 테스트 요청 모킹
        test_response = Mock()
        test_response.raise_for_status.return_value = None
        test_response.json.return_value = {'response': 'Hello!', 'done': True}
        
        mock_post.side_effect = [pull_response, test_response]
        
        self.model.initialize()
        
        assert self.model.is_available()
        assert mock_post.call_count == 2  # pull + test
    
    @patch('requests.get')
    @patch('requests.post')
    def test_initialize_model_not_available_pull_fail(self, mock_post, mock_get):
        """모델이 없고 다운로드도 실패하는 경우 테스트."""
        # 서버 연결 확인
        mock_get.side_effect = [
            Mock(raise_for_status=Mock(), json=Mock(return_value={'models': []})),
            Mock(raise_for_status=Mock(), json=Mock(return_value={'models': []})),
            Mock(raise_for_status=Mock(), json=Mock(return_value={'models': []}))
        ]
        
        # 모델 pull 실패
        mock_post.side_effect = requests.exceptions.RequestException("Pull failed")
        
        with pytest.raises(LanguageModelError) as exc_info:
            self.model.initialize()
        
        assert "Failed to pull model" in str(exc_info.value)
        assert not self.model.is_available()
    
    @patch('requests.post')
    def test_generate_response_success(self, mock_post):
        """응답 생성 성공 테스트."""
        self.model._is_initialized = True
        
        mock_response = Mock()
        mock_response.raise_for_status.return_value = None
        mock_response.json.return_value = {
            'response': 'This is a test response.',
            'done': True,
            'prompt_eval_count': 10,
            'eval_count': 5,
            'total_duration': 1000000000
        }
        mock_post.return_value = mock_response
        
        result = self.model.generate_response("Hello, how are you?")
        
        assert isinstance(result, LLMResponse)
        assert result.content == 'This is a test response.'
        assert result.model == 'llama2'
        assert result.usage['prompt_tokens'] == 10
        assert result.usage['completion_tokens'] == 5
        assert result.usage['total_tokens'] == 15
        assert 'total_duration' in result.metadata
    
    @patch('requests.post')
    def test_generate_response_with_context(self, mock_post):
        """컨텍스트와 함께 응답 생성 테스트."""
        self.model._is_initialized = True
        
        mock_response = Mock()
        mock_response.raise_for_status.return_value = None
        mock_response.json.return_value = {
            'response': 'Response with context.',
            'done': True
        }
        mock_post.return_value = mock_response
        
        result = self.model.generate_response(
            "What is this about?",
            context="This is about machine learning."
        )
        
        assert isinstance(result, LLMResponse)
        assert result.content == 'Response with context.'
        
        # 호출된 프롬프트에 컨텍스트가 포함되었는지 확인
        call_args = mock_post.call_args[1]['json']
        assert "컨텍스트를 참고하여" in call_args['prompt']
        assert "This is about machine learning." in call_args['prompt']
    
    def test_generate_response_not_initialized(self):
        """초기화되지 않은 상태에서 응답 생성 시도 테스트."""
        with pytest.raises(LanguageModelError) as exc_info:
            self.model.generate_response("Hello")
        
        assert "not initialized" in str(exc_info.value)
    
    @patch('requests.post')
    def test_generate_response_api_error(self, mock_post):
        """API 오류 시 응답 생성 테스트."""
        self.model._is_initialized = True
        mock_post.side_effect = requests.exceptions.RequestException("API Error")
        
        with pytest.raises(LanguageModelError) as exc_info:
            self.model.generate_response("Hello")
        
        assert "Ollama API error in generate_response" in str(exc_info.value)
    
    @patch('requests.post')
    def test_translate_text_success(self, mock_post):
        """텍스트 번역 성공 테스트."""
        self.model._is_initialized = True
        
        mock_response = Mock()
        mock_response.raise_for_status.return_value = None
        mock_response.json.return_value = {
            'response': '안녕하세요, 어떻게 지내세요?',
            'done': True
        }
        mock_post.return_value = mock_response
        
        result = self.model.translate_text("Hello, how are you?", "korean")
        
        assert result == '안녕하세요, 어떻게 지내세요?'
        
        # 번역 프롬프트가 올바르게 구성되었는지 확인
        call_args = mock_post.call_args[1]['json']
        assert "korean로 번역해주세요" in call_args['prompt']
        assert "Hello, how are you?" in call_args['prompt']
    
    @patch('requests.post')
    def test_analyze_grammar_success(self, mock_post):
        """문법 분석 성공 테스트."""
        self.model._is_initialized = True
        
        analysis_json = {
            "grammar_errors": [
                {
                    "text": "I are",
                    "error_type": "grammar",
                    "position": [0, 5],
                    "suggestion": "I am",
                    "explanation": "주어가 I일 때는 am을 사용해야 합니다."
                }
            ],
            "vocabulary_level": "intermediate",
            "fluency_score": 0.8,
            "complexity_score": 0.6,
            "suggestions": [
                {
                    "category": "vocabulary",
                    "original": "good",
                    "improved": "excellent",
                    "reason": "더 정교한 어휘 선택",
                    "confidence": 0.9
                }
            ]
        }
        
        mock_response = Mock()
        mock_response.raise_for_status.return_value = None
        mock_response.json.return_value = {
            'response': json.dumps(analysis_json),
            'done': True
        }
        mock_post.return_value = mock_response
        
        result = self.model.analyze_grammar("I are good student.")
        
        assert isinstance(result, EnglishAnalysis)
        assert result.vocabulary_level == "intermediate"
        assert result.fluency_score == 0.8
        assert result.complexity_score == 0.6
        assert len(result.grammar_errors) == 1
        assert len(result.suggestions) == 1
        
        error = result.grammar_errors[0]
        assert error.text == "I are"
        assert error.error_type == ErrorType.GRAMMAR
        assert error.suggestion == "I am"
    
    @patch('requests.post')
    def test_analyze_grammar_json_parse_error_with_retry(self, mock_post):
        """JSON 파싱 오류 시 재시도 테스트."""
        self.model._is_initialized = True
        
        # 첫 번째 응답은 잘못된 JSON
        first_response = Mock()
        first_response.raise_for_status.return_value = None
        first_response.json.return_value = {
            'response': 'This is not valid JSON',
            'done': True
        }
        
        # 두 번째 응답은 올바른 JSON
        second_response = Mock()
        second_response.raise_for_status.return_value = None
        second_response.json.return_value = {
            'response': '{"grammar_errors":[],"vocabulary_level":"intermediate","fluency_score":0.7,"complexity_score":0.5,"suggestions":[]}',
            'done': True
        }
        
        mock_post.side_effect = [first_response, second_response]
        
        result = self.model.analyze_grammar("This is a test.")
        
        assert isinstance(result, EnglishAnalysis)
        assert result.vocabulary_level == "intermediate"
        assert mock_post.call_count == 2  # 재시도 확인
    
    @patch('requests.get')
    def test_list_available_models_success(self, mock_get):
        """사용 가능한 모델 목록 조회 성공 테스트."""
        mock_get.return_value.raise_for_status.return_value = None
        mock_get.return_value.json.return_value = {
            'models': [
                {'name': 'llama2:latest'},
                {'name': 'mistral:7b'},
                {'name': 'codellama:13b'}
            ]
        }
        
        models = self.model.list_available_models()
        
        assert models == ['llama2:latest', 'mistral:7b', 'codellama:13b']
        mock_get.assert_called_with("http://localhost:11434/api/tags", timeout=30)
    
    @patch('requests.get')
    def test_is_server_running_true(self, mock_get):
        """서버 실행 상태 확인 - 실행 중 테스트."""
        mock_get.return_value.raise_for_status.return_value = None
        
        assert self.model.is_server_running() is True
        mock_get.assert_called_with("http://localhost:11434/api/tags", timeout=30)
    
    @patch('requests.get')
    def test_is_server_running_false(self, mock_get):
        """서버 실행 상태 확인 - 실행 중이지 않음 테스트."""
        mock_get.side_effect = requests.exceptions.ConnectionError("Connection refused")
        
        assert self.model.is_server_running() is False
    
    def test_get_model_info(self):
        """모델 정보 반환 테스트."""
        info = self.model.get_model_info()
        
        assert info['model_name'] == 'llama2'
        assert info['provider'] == 'Ollama'
        assert info['host'] == 'localhost:11434'
        assert info['base_url'] == 'http://localhost:11434'
        assert info['temperature'] == 0.7
        assert 'common_models' in info
        assert 'llama2' in info['common_models']
    
    @patch('requests.get')
    def test_get_model_info_with_available_models(self, mock_get):
        """초기화된 상태에서 모델 정보 반환 테스트."""
        self.model._is_initialized = True
        
        mock_get.return_value.raise_for_status.return_value = None
        mock_get.return_value.json.return_value = {
            'models': [{'name': 'llama2:latest'}]
        }
        
        info = self.model.get_model_info()
        
        assert 'available_models' in info
        assert info['available_models'] == ['llama2:latest']
    
    def test_validate_input_valid(self):
        """유효한 입력 검증 테스트."""
        # 예외가 발생하지 않아야 함
        self.model.validate_input("This is a valid input.")
    
    def test_validate_input_empty(self):
        """빈 입력 검증 테스트."""
        with pytest.raises(ValueError) as exc_info:
            self.model.validate_input("")
        
        assert "cannot be empty" in str(exc_info.value)
    
    def test_validate_input_too_long(self):
        """너무 긴 입력 검증 테스트."""
        long_text = "a" * 10001
        
        with pytest.raises(ValueError) as exc_info:
            self.model.validate_input(long_text)
        
        assert "too long" in str(exc_info.value)
    
    def test_validate_input_not_string(self):
        """문자열이 아닌 입력 검증 테스트."""
        with pytest.raises(ValueError) as exc_info:
            self.model.validate_input(123)
        
        assert "must be a string" in str(exc_info.value)
    
    def test_is_model_available_exact_match(self):
        """모델 가용성 확인 - 정확한 이름 매치 테스트."""
        available_models = [
            {'name': 'llama2'},
            {'name': 'mistral:7b'}
        ]
        
        assert self.model._is_model_available(available_models) is True
    
    def test_is_model_available_with_tag(self):
        """모델 가용성 확인 - 태그 포함 매치 테스트."""
        self.model.model_name = "mistral"
        available_models = [
            {'name': 'llama2:latest'},
            {'name': 'mistral:latest'}
        ]
        
        assert self.model._is_model_available(available_models) is True
    
    def test_is_model_available_not_found(self):
        """모델 가용성 확인 - 모델 없음 테스트."""
        available_models = [
            {'name': 'codellama:7b'},
            {'name': 'mistral:7b'}
        ]
        
        assert self.model._is_model_available(available_models) is False
    
    def test_calculate_usage_with_ollama_data(self):
        """Ollama 응답 데이터로 사용량 계산 테스트."""
        response_data = {
            'prompt_eval_count': 15,
            'eval_count': 8
        }
        
        usage = self.model._calculate_usage("test prompt", "test response", response_data)
        
        assert usage['prompt_tokens'] == 15
        assert usage['completion_tokens'] == 8
        assert usage['total_tokens'] == 23
    
    def test_calculate_usage_without_ollama_data(self):
        """Ollama 데이터 없이 사용량 계산 테스트."""
        response_data = {}
        
        usage = self.model._calculate_usage("test prompt", "test response", response_data)
        
        # 단어 기반 근사치
        assert usage['prompt_tokens'] == 2  # "test prompt"
        assert usage['completion_tokens'] == 2  # "test response"
        assert usage['total_tokens'] == 4
    
    def test_extract_metadata(self):
        """메타데이터 추출 테스트."""
        response_data = {
            'total_duration': 1000000000,
            'load_duration': 100000000,
            'prompt_eval_duration': 200000000,
            'eval_duration': 700000000,
            'done': True,
            'context': [1, 2, 3, 4, 5]
        }
        
        metadata = self.model._extract_metadata(response_data)
        
        assert metadata['total_duration'] == 1000000000
        assert metadata['load_duration'] == 100000000
        assert metadata['prompt_eval_duration'] == 200000000
        assert metadata['eval_duration'] == 700000000
        assert metadata['done'] is True
        assert metadata['context_length'] == 5


@pytest.fixture
def mock_ollama_server():
    """Ollama 서버 모킹을 위한 픽스처."""
    with patch('requests.get') as mock_get, patch('requests.post') as mock_post:
        # 기본 서버 응답 설정
        mock_get.return_value.raise_for_status.return_value = None
        mock_get.return_value.json.return_value = {
            'models': [{'name': 'llama2:latest'}]
        }
        
        mock_post.return_value.raise_for_status.return_value = None
        mock_post.return_value.json.return_value = {
            'response': 'Test response',
            'done': True
        }
        
        yield mock_get, mock_post


def test_integration_full_workflow(mock_ollama_server):
    """전체 워크플로우 통합 테스트."""
    mock_get, mock_post = mock_ollama_server
    
    model = OllamaLanguageModel("llama2")
    
    # 초기화
    model.initialize()
    assert model.is_available()
    
    # 응답 생성
    response = model.generate_response("Hello, world!")
    assert isinstance(response, LLMResponse)
    assert response.content == "Test response"
    
    # 번역
    translation = model.translate_text("Hello", "korean")
    assert translation == "Test response"
    
    # 서버 상태 확인
    assert model.is_server_running()
    
    # 모델 정보
    info = model.get_model_info()
    assert info['provider'] == 'Ollama'
    assert info['initialized'] is True