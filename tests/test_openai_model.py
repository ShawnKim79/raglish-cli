"""
OpenAI Language Model 모듈 단위 테스트.

이 모듈은 OpenAILanguageModel 클래스의 모든 기능을 테스트합니다.
"""

import pytest
import json
from unittest.mock import Mock, patch, MagicMock

from src.document_rag_english_study.llm.openai_model import OpenAILanguageModel
from src.document_rag_english_study.models.llm import LLMResponse, EnglishAnalysis
from src.document_rag_english_study.llm.base import (
    LanguageModelError, AuthenticationError, RateLimitError, APIConnectionError
)


class TestOpenAILanguageModel:
    """OpenAILanguageModel 클래스 테스트."""

    @pytest.fixture
    def mock_openai(self):
        """OpenAI 클라이언트 모킹."""
        with patch('src.document_rag_english_study.llm.openai_model.OpenAI') as mock:
            yield mock

    @pytest.fixture
    def openai_model(self, mock_openai):
        """OpenAI 모델 인스턴스."""
        model = OpenAILanguageModel(api_key="test-api-key")
        return model

    def test_initialization_success(self, mock_openai):
        """성공적인 초기화 테스트."""
        model = OpenAILanguageModel(api_key="test-api-key")
        
        assert model.api_key == "test-api-key"
        assert model.model_name == "gpt-3.5-turbo"
        assert model.temperature == 0.7
        assert model.max_tokens == 1000
        mock_openai.assert_called_once_with(api_key="test-api-key")

    def test_initialization_custom_params(self, mock_openai):
        """사용자 지정 매개변수 초기화 테스트."""
        model = OpenAILanguageModel(
            api_key="custom-key",
            model="gpt-4",
            temperature=0.5,
            max_tokens=2000
        )
        
        assert model.api_key == "custom-key"
        assert model.model_name == "gpt-4"
        assert model.temperature == 0.5
        assert model.max_tokens == 2000

    def test_initialization_no_api_key(self):
        """API 키 없이 초기화 테스트."""
        with pytest.raises(LanguageModelError, match="OpenAI API key is required"):
            OpenAILanguageModel()

    def test_initialization_empty_api_key(self):
        """빈 API 키로 초기화 테스트."""
        with pytest.raises(LanguageModelError, match="OpenAI API key is required"):
            OpenAILanguageModel(api_key="")

    @patch('src.document_rag_english_study.llm.openai_model.OpenAI')
    def test_initialization_connection_error(self, mock_openai):
        """연결 오류 초기화 테스트."""
        mock_openai.side_effect = Exception("Connection failed")
        
        with pytest.raises(LanguageModelError, match="Failed to initialize OpenAI client"):
            OpenAILanguageModel(api_key="test-key")

    def test_generate_response_success(self, openai_model):
        """성공적인 응답 생성 테스트."""
        # Mock 설정
        mock_response = Mock()
        mock_choice = Mock()
        mock_message = Mock()
        mock_message.content = "Test response from OpenAI"
        mock_choice.message = mock_message
        mock_response.choices = [mock_choice]
        mock_response.usage = Mock()
        mock_response.usage.prompt_tokens = 10
        mock_response.usage.completion_tokens = 5
        mock_response.usage.total_tokens = 15
        
        openai_model.client.chat.completions.create.return_value = mock_response
        
        result = openai_model.generate_response("Test prompt")
        
        assert isinstance(result, LLMResponse)
        assert result.content == "Test response from OpenAI"
        assert result.usage["prompt_tokens"] == 10
        assert result.usage["completion_tokens"] == 5
        assert result.usage["total_tokens"] == 15

    def test_generate_response_with_context(self, openai_model):
        """컨텍스트 포함 응답 생성 테스트."""
        # Mock 설정
        mock_response = Mock()
        mock_choice = Mock()
        mock_message = Mock()
        mock_message.content = "Response with context"
        mock_choice.message = mock_message
        mock_response.choices = [mock_choice]
        mock_response.usage = Mock()
        mock_response.usage.prompt_tokens = 20
        mock_response.usage.completion_tokens = 10
        mock_response.usage.total_tokens = 30
        
        openai_model.client.chat.completions.create.return_value = mock_response
        
        result = openai_model.generate_response("Test prompt", context="Test context")
        
        assert isinstance(result, LLMResponse)
        assert result.content == "Response with context"
        
        # 호출 인자 확인
        call_args = openai_model.client.chat.completions.create.call_args
        messages = call_args[1]["messages"]
        assert len(messages) == 2  # system + user message
        assert "Test context" in messages[0]["content"]
        assert "Test prompt" in messages[1]["content"]

    def test_generate_response_empty_prompt(self, openai_model):
        """빈 프롬프트 응답 생성 테스트."""
        with pytest.raises(LanguageModelError, match="Prompt cannot be empty"):
            openai_model.generate_response("")

    def test_generate_response_api_error(self, openai_model):
        """API 오류 응답 생성 테스트."""
        openai_model.client.chat.completions.create.side_effect = Exception("API Error")
        
        with pytest.raises(LanguageModelError):
            openai_model.generate_response("Test prompt")

    def test_translate_text_success(self, openai_model):
        """성공적인 텍스트 번역 테스트."""
        # Mock 설정
        mock_response = Mock()
        mock_choice = Mock()
        mock_message = Mock()
        mock_message.content = "안녕하세요"
        mock_choice.message = mock_message
        mock_response.choices = [mock_choice]
        mock_response.usage = Mock()
        mock_response.usage.prompt_tokens = 15
        mock_response.usage.completion_tokens = 3
        mock_response.usage.total_tokens = 18
        
        openai_model.client.chat.completions.create.return_value = mock_response
        
        result = openai_model.translate_text("Hello", target_language="korean")
        
        assert isinstance(result, LLMResponse)
        assert result.content == "안녕하세요"

    def test_translate_text_empty_text(self, openai_model):
        """빈 텍스트 번역 테스트."""
        with pytest.raises(LanguageModelError, match="Text cannot be empty"):
            openai_model.translate_text("", target_language="korean")

    def test_analyze_grammar_success(self, openai_model):
        """성공적인 문법 분석 테스트."""
        # Mock 응답 데이터
        grammar_data = {
            "errors": [
                {
                    "type": "subject_verb_agreement",
                    "message": "Subject-verb disagreement",
                    "suggestion": "Use 'is' instead of 'are'",
                    "position": {"start": 5, "end": 8}
                }
            ],
            "score": 85,
            "level": "intermediate"
        }
        
        # Mock 설정
        mock_response = Mock()
        mock_choice = Mock()
        mock_message = Mock()
        mock_message.content = json.dumps(grammar_data)
        mock_choice.message = mock_message
        mock_response.choices = [mock_choice]
        mock_response.usage = Mock()
        mock_response.usage.prompt_tokens = 25
        mock_response.usage.completion_tokens = 50
        mock_response.usage.total_tokens = 75
        
        openai_model.client.chat.completions.create.return_value = mock_response
        
        result = openai_model.analyze_grammar("This are a test sentence")
        
        assert isinstance(result, GrammarAnalysis)
        assert len(result.errors) == 1
        assert result.errors[0].type == "subject_verb_agreement"
        assert result.score == 85
        assert result.level == "intermediate"

    def test_analyze_grammar_json_parse_error(self, openai_model):
        """JSON 파싱 오류 문법 분석 테스트."""
        # Mock 설정 - 잘못된 JSON 응답
        mock_response = Mock()
        mock_choice = Mock()
        mock_message = Mock()
        mock_message.content = "Invalid JSON response"
        mock_choice.message = mock_message
        mock_response.choices = [mock_choice]
        mock_response.usage = Mock()
        mock_response.usage.prompt_tokens = 20
        mock_response.usage.completion_tokens = 5
        mock_response.usage.total_tokens = 25
        
        openai_model.client.chat.completions.create.return_value = mock_response
        
        result = openai_model.analyze_grammar("Test sentence")
        
        # JSON 파싱 실패 시 기본값 반환
        assert isinstance(result, GrammarAnalysis)
        assert len(result.errors) == 0
        assert result.score == 0

    def test_analyze_grammar_empty_text(self, openai_model):
        """빈 텍스트 문법 분석 테스트."""
        with pytest.raises(LanguageModelError, match="Text cannot be empty"):
            openai_model.analyze_grammar("")

    def test_handle_api_error_authentication(self, openai_model):
        """인증 오류 처리 테스트."""
        error = Exception("Invalid API key")
        
        with pytest.raises(AuthenticationError):
            openai_model._handle_api_error(error, "test_method")

    def test_handle_api_error_rate_limit(self, openai_model):
        """속도 제한 오류 처리 테스트."""
        error = Exception("Rate limit exceeded")
        
        with pytest.raises(RateLimitError):
            openai_model._handle_api_error(error, "test_method")

    def test_handle_api_error_connection(self, openai_model):
        """연결 오류 처리 테스트."""
        error = Exception("Connection timeout")
        
        with pytest.raises(APIConnectionError):
            openai_model._handle_api_error(error, "test_method")

    def test_handle_api_error_generic(self, openai_model):
        """일반 오류 처리 테스트."""
        error = Exception("Unknown error")
        
        with pytest.raises(LanguageModelError):
            openai_model._handle_api_error(error, "test_method")

    def test_calculate_usage_with_usage_data(self, openai_model):
        """사용량 데이터 포함 계산 테스트."""
        mock_response = Mock()
        mock_response.usage = Mock()
        mock_response.usage.prompt_tokens = 100
        mock_response.usage.completion_tokens = 50
        mock_response.usage.total_tokens = 150
        
        usage = openai_model._calculate_usage("test prompt", "test response", mock_response)
        
        assert usage["prompt_tokens"] == 100
        assert usage["completion_tokens"] == 50
        assert usage["total_tokens"] == 150

    def test_calculate_usage_without_usage_data(self, openai_model):
        """사용량 데이터 없이 계산 테스트."""
        mock_response = Mock()
        # usage 속성이 없는 경우
        del mock_response.usage
        
        usage = openai_model._calculate_usage("test prompt", "test response", mock_response)
        
        # 근사치 계산 확인
        assert usage["prompt_tokens"] == 2  # "test prompt" = 2 words
        assert usage["completion_tokens"] == 2  # "test response" = 2 words
        assert usage["total_tokens"] == 4

    def test_extract_metadata(self, openai_model):
        """메타데이터 추출 테스트."""
        mock_response = Mock()
        mock_response.model = "gpt-3.5-turbo"
        mock_response.created = 1234567890
        
        metadata = openai_model._extract_metadata(mock_response)
        
        assert metadata["model"] == "gpt-3.5-turbo"
        assert metadata["created"] == 1234567890

    def test_get_model_info(self, openai_model):
        """모델 정보 조회 테스트."""
        info = openai_model.get_model_info()
        
        assert info["provider"] == "openai"
        assert info["model_name"] == "gpt-3.5-turbo"
        assert info["temperature"] == 0.7
        assert info["max_tokens"] == 1000

    def test_supported_models_list(self, openai_model):
        """지원 모델 목록 테스트."""
        models = openai_model.get_supported_models()
        
        assert isinstance(models, list)
        assert "gpt-3.5-turbo" in models
        assert "gpt-4" in models

    def test_validate_input_valid(self, openai_model):
        """유효한 입력 검증 테스트."""
        # 예외가 발생하지 않아야 함
        openai_model.validate_input("Valid input text")

    def test_validate_input_empty(self, openai_model):
        """빈 입력 검증 테스트."""
        with pytest.raises(LanguageModelError, match="Input cannot be empty"):
            openai_model.validate_input("")

    def test_validate_input_none(self, openai_model):
        """None 입력 검증 테스트."""
        with pytest.raises(LanguageModelError, match="Input cannot be empty"):
            openai_model.validate_input(None)

    def test_validate_input_too_long(self, openai_model):
        """너무 긴 입력 검증 테스트."""
        long_input = "x" * 10000  # 매우 긴 텍스트
        
        with pytest.raises(LanguageModelError, match="Input too long"):
            openai_model.validate_input(long_input)

    def test_build_messages_simple(self, openai_model):
        """간단한 메시지 구성 테스트."""
        messages = openai_model._build_messages("Test prompt")
        
        assert len(messages) == 1
        assert messages[0]["role"] == "user"
        assert messages[0]["content"] == "Test prompt"

    def test_build_messages_with_context(self, openai_model):
        """컨텍스트 포함 메시지 구성 테스트."""
        messages = openai_model._build_messages("Test prompt", context="Test context")
        
        assert len(messages) == 2
        assert messages[0]["role"] == "system"
        assert "Test context" in messages[0]["content"]
        assert messages[1]["role"] == "user"
        assert messages[1]["content"] == "Test prompt"

    def test_build_translation_prompt(self, openai_model):
        """번역 프롬프트 구성 테스트."""
        prompt = openai_model._build_translation_prompt("Hello", "korean")
        
        assert "Hello" in prompt
        assert "korean" in prompt.lower()
        assert "translate" in prompt.lower()

    def test_build_grammar_analysis_prompt(self, openai_model):
        """문법 분석 프롬프트 구성 테스트."""
        prompt = openai_model._build_grammar_analysis_prompt("Test sentence")
        
        assert "Test sentence" in prompt
        assert "grammar" in prompt.lower()
        assert "json" in prompt.lower()


class TestOpenAILanguageModelIntegration:
    """OpenAILanguageModel 통합 테스트."""

    @pytest.fixture
    def mock_openai(self):
        """OpenAI 클라이언트 모킹."""
        with patch('src.document_rag_english_study.llm.openai_model.OpenAI') as mock:
            yield mock

    @pytest.fixture
    def openai_model(self, mock_openai):
        """OpenAI 모델 인스턴스."""
        model = OpenAILanguageModel(api_key="test-api-key")
        return model

    def test_full_conversation_workflow(self, openai_model):
        """전체 대화 워크플로우 테스트."""
        # Mock 응답 설정
        mock_response = Mock()
        mock_choice = Mock()
        mock_message = Mock()
        mock_message.content = "This is a test response"
        mock_choice.message = mock_message
        mock_response.choices = [mock_choice]
        mock_response.usage = Mock()
        mock_response.usage.prompt_tokens = 10
        mock_response.usage.completion_tokens = 5
        mock_response.usage.total_tokens = 15
        
        openai_model.client.chat.completions.create.return_value = mock_response
        
        # 1. 기본 응답 생성
        response1 = openai_model.generate_response("Hello")
        assert response1.content == "This is a test response"
        
        # 2. 컨텍스트 포함 응답 생성
        response2 = openai_model.generate_response("Continue", context="Previous conversation")
        assert response2.content == "This is a test response"
        
        # 3. 번역
        response3 = openai_model.translate_text("Hello", target_language="korean")
        assert response3.content == "This is a test response"
        
        # 모든 호출이 성공적으로 완료되어야 함
        assert openai_model.client.chat.completions.create.call_count == 3

    def test_error_recovery_workflow(self, openai_model):
        """오류 복구 워크플로우 테스트."""
        # 첫 번째 호출은 실패
        openai_model.client.chat.completions.create.side_effect = [
            Exception("Temporary error"),
            Mock()  # 두 번째 호출은 성공
        ]
        
        # 첫 번째 호출 실패
        with pytest.raises(LanguageModelError):
            openai_model.generate_response("Test prompt")
        
        # Mock 응답 설정 (두 번째 호출용)
        mock_response = Mock()
        mock_choice = Mock()
        mock_message = Mock()
        mock_message.content = "Recovery successful"
        mock_choice.message = mock_message
        mock_response.choices = [mock_choice]
        mock_response.usage = Mock()
        mock_response.usage.prompt_tokens = 5
        mock_response.usage.completion_tokens = 3
        mock_response.usage.total_tokens = 8
        
        openai_model.client.chat.completions.create.side_effect = None
        openai_model.client.chat.completions.create.return_value = mock_response
        
        # 두 번째 호출 성공
        response = openai_model.generate_response("Test prompt")
        assert response.content == "Recovery successful"