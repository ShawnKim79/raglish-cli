"""
Google Gemini 언어 모델 구현체 테스트.

이 모듈은 GeminiLanguageModel 클래스의 기능을 테스트합니다.
"""

import pytest
import json
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, Any

from src.document_rag_english_study.llm.gemini_model import GeminiLanguageModel
from src.document_rag_english_study.llm.base import (
    LanguageModelError,
    APIConnectionError,
    AuthenticationError,
    RateLimitError
)
from src.document_rag_english_study.models.llm import (
    LLMResponse,
    EnglishAnalysis,
    GrammarError,
    ErrorType,
    ImprovementSuggestion
)


class TestGeminiLanguageModel:
    """GeminiLanguageModel 클래스 테스트."""
    
    @pytest.fixture
    def mock_genai(self):
        """Google Generative AI 모듈 모킹."""
        with patch('src.document_rag_english_study.llm.gemini_model.genai') as mock:
            yield mock
    
    @pytest.fixture
    def gemini_model(self):
        """테스트용 Gemini 모델 인스턴스."""
        return GeminiLanguageModel(
            model_name="gemini-pro",
            api_key="test-api-key",
            temperature=0.7
        )
    
    def test_initialization_success(self, gemini_model, mock_genai):
        """정상적인 초기화 테스트."""
        # Mock 설정
        mock_client = Mock()
        mock_response = Mock()
        mock_response.text = "Test response"
        mock_client.generate_content.return_value = mock_response
        mock_genai.GenerativeModel.return_value = mock_client
        
        # 초기화 실행
        gemini_model.initialize()
        
        # 검증
        assert gemini_model._is_initialized is True
        mock_genai.configure.assert_called_once_with(api_key="test-api-key")
        mock_genai.GenerativeModel.assert_called_once_with("gemini-pro")
    
    def test_initialization_no_api_key(self):
        """API 키 없이 초기화 시 오류 테스트."""
        model = GeminiLanguageModel(api_key=None)
        
        with pytest.raises(AuthenticationError, match="Google API key is required"):
            model.initialize()
    
    def test_initialization_import_error(self, gemini_model):
        """Google Generative AI 패키지 없을 때 오류 테스트."""
        with patch('builtins.__import__', side_effect=ImportError):
            with pytest.raises(LanguageModelError, match="Google Generative AI package not installed"):
                gemini_model.initialize()
    
    def test_initialization_connection_error(self, gemini_model, mock_genai):
        """연결 테스트 실패 시 오류 테스트."""
        # Mock 설정 - 연결 테스트 실패
        mock_client = Mock()
        mock_client.generate_content.side_effect = Exception("Connection failed")
        mock_genai.GenerativeModel.return_value = mock_client
        
        with pytest.raises(APIConnectionError, match="Failed to connect to Gemini API"):
            gemini_model.initialize()
    
    def test_generate_response_success(self, gemini_model, mock_genai):
        """정상적인 응답 생성 테스트."""
        # 초기화
        gemini_model._is_initialized = True
        
        # Mock 설정
        mock_client = Mock()
        mock_response = Mock()
        mock_response.text = "Generated response"
        mock_response.usage_metadata = Mock()
        mock_response.usage_metadata.prompt_token_count = 10
        mock_response.usage_metadata.candidates_token_count = 5
        mock_response.usage_metadata.total_token_count = 15
        mock_response.candidates = [Mock()]
        mock_response.candidates[0].finish_reason = "STOP"
        mock_response.candidates[0].safety_ratings = []
        
        mock_client.generate_content.return_value = mock_response
        gemini_model.client = mock_client
        
        # 응답 생성
        result = gemini_model.generate_response("Test prompt")
        
        # 검증
        assert isinstance(result, LLMResponse)
        assert result.content == "Generated response"
        assert result.model == "gemini-pro"
        assert result.usage['prompt_tokens'] == 10
        assert result.usage['completion_tokens'] == 5
        assert result.usage['total_tokens'] == 15
    
    def test_generate_response_not_initialized(self, gemini_model):
        """초기화되지 않은 상태에서 응답 생성 시 오류 테스트."""
        with pytest.raises(LanguageModelError, match="Model not initialized"):
            gemini_model.generate_response("Test prompt")
    
    def test_generate_response_empty_response(self, gemini_model):
        """빈 응답 처리 테스트."""
        gemini_model._is_initialized = True
        
        # Mock 설정 - 빈 응답
        mock_client = Mock()
        mock_response = Mock()
        mock_response.text = None
        mock_response.candidates = []
        mock_client.generate_content.return_value = mock_response
        gemini_model.client = mock_client
        
        with pytest.raises(LanguageModelError, match="No valid response from Gemini API"):
            gemini_model.generate_response("Test prompt")
    
    def test_generate_response_safety_filter(self, gemini_model):
        """안전성 필터로 차단된 응답 처리 테스트."""
        gemini_model._is_initialized = True
        
        # Mock 설정 - 안전성 필터 차단
        mock_client = Mock()
        mock_response = Mock()
        mock_response.prompt_feedback = Mock()
        mock_response.prompt_feedback.block_reason = "SAFETY"
        mock_client.generate_content.return_value = mock_response
        gemini_model.client = mock_client
        
        with pytest.raises(LanguageModelError, match="Content blocked by safety filter"):
            gemini_model.generate_response("Test prompt")
    
    def test_translate_text_success(self, gemini_model):
        """정상적인 번역 테스트."""
        gemini_model._is_initialized = True
        
        # Mock generate_response
        with patch.object(gemini_model, 'generate_response') as mock_generate:
            mock_response = LLMResponse(
                content="안녕하세요",
                model="gemini-pro",
                usage={'prompt_tokens': 10, 'completion_tokens': 5, 'total_tokens': 15}
            )
            mock_generate.return_value = mock_response
            
            result = gemini_model.translate_text("Hello", "Korean")
            
            assert result == "안녕하세요"
            mock_generate.assert_called_once()
    
    def test_analyze_grammar_success(self, gemini_model):
        """정상적인 문법 분석 테스트."""
        gemini_model._is_initialized = True
        
        # 테스트용 JSON 응답
        analysis_json = {
            "grammar_errors": [
                {
                    "text": "I are",
                    "error_type": "grammar",
                    "position": [0, 5],
                    "suggestion": "I am",
                    "explanation": "주어와 동사의 일치 오류"
                }
            ],
            "vocabulary_level": "beginner",
            "fluency_score": 0.6,
            "complexity_score": 0.4,
            "suggestions": [
                {
                    "category": "grammar",
                    "original": "I are happy",
                    "improved": "I am happy",
                    "reason": "주어-동사 일치",
                    "confidence": 0.9
                }
            ]
        }
        
        # Mock generate_response
        with patch.object(gemini_model, 'generate_response') as mock_generate:
            mock_response = LLMResponse(
                content=json.dumps(analysis_json),
                model="gemini-pro",
                usage={'prompt_tokens': 50, 'completion_tokens': 100, 'total_tokens': 150}
            )
            mock_generate.return_value = mock_response
            
            result = gemini_model.analyze_grammar("I are happy")
            
            # 검증
            assert isinstance(result, EnglishAnalysis)
            assert result.vocabulary_level == "beginner"
            assert result.fluency_score == 0.6
            assert result.complexity_score == 0.4
            assert len(result.grammar_errors) == 1
            assert len(result.suggestions) == 1
            
            # 문법 오류 검증
            error = result.grammar_errors[0]
            assert error.text == "I are"
            assert error.error_type == ErrorType.GRAMMAR
            assert error.position == (0, 5)
            assert error.suggestion == "I am"
    
    def test_analyze_grammar_json_parse_error(self, gemini_model):
        """JSON 파싱 오류 시 재시도 테스트."""
        gemini_model._is_initialized = True
        
        # Mock generate_response - 첫 번째 호출은 잘못된 JSON, 두 번째는 성공
        with patch.object(gemini_model, 'generate_response') as mock_generate:
            # 첫 번째 응답 - 잘못된 JSON
            invalid_response = LLMResponse(
                content="Invalid JSON response",
                model="gemini-pro",
                usage={'prompt_tokens': 50, 'completion_tokens': 20, 'total_tokens': 70}
            )
            
            # 두 번째 응답 - 유효한 JSON
            valid_json = {
                "grammar_errors": [],
                "vocabulary_level": "intermediate",
                "fluency_score": 0.7,
                "complexity_score": 0.5,
                "suggestions": []
            }
            valid_response = LLMResponse(
                content=json.dumps(valid_json),
                model="gemini-pro",
                usage={'prompt_tokens': 50, 'completion_tokens': 30, 'total_tokens': 80}
            )
            
            mock_generate.side_effect = [invalid_response, valid_response]
            
            result = gemini_model.analyze_grammar("Test text")
            
            # 검증 - 재시도 후 성공
            assert isinstance(result, EnglishAnalysis)
            assert result.vocabulary_level == "intermediate"
            assert mock_generate.call_count == 2
    
    def test_analyze_grammar_complete_failure(self, gemini_model):
        """JSON 파싱 완전 실패 시 기본값 반환 테스트."""
        gemini_model._is_initialized = True
        
        # Mock generate_response - 모든 호출이 실패
        with patch.object(gemini_model, 'generate_response') as mock_generate:
            invalid_response = LLMResponse(
                content="Completely invalid response",
                model="gemini-pro",
                usage={'prompt_tokens': 50, 'completion_tokens': 20, 'total_tokens': 70}
            )
            mock_generate.return_value = invalid_response
            
            result = gemini_model.analyze_grammar("Test text")
            
            # 검증 - 기본값 반환
            assert isinstance(result, EnglishAnalysis)
            assert result.vocabulary_level == "intermediate"
            assert result.fluency_score == 0.7
            assert result.complexity_score == 0.5
            assert len(result.grammar_errors) == 0
            assert len(result.suggestions) == 0
    
    def test_handle_api_error_authentication(self, gemini_model):
        """인증 오류 처리 테스트."""
        error = Exception("API key unauthorized")
        
        with pytest.raises(AuthenticationError, match="Invalid Google API key"):
            gemini_model._handle_api_error(error, "test_method")
    
    def test_handle_api_error_rate_limit(self, gemini_model):
        """속도 제한 오류 처리 테스트."""
        error = Exception("Rate limit exceeded")
        
        with pytest.raises(RateLimitError, match="Gemini API rate limit exceeded"):
            gemini_model._handle_api_error(error, "test_method")
    
    def test_handle_api_error_connection(self, gemini_model):
        """연결 오류 처리 테스트."""
        error = Exception("Connection timeout")
        
        with pytest.raises(APIConnectionError, match="Gemini API connection error"):
            gemini_model._handle_api_error(error, "test_method")
    
    def test_handle_api_error_safety_filter(self, gemini_model):
        """안전성 필터 오류 처리 테스트."""
        error = Exception("Content blocked by safety filters")
        
        with pytest.raises(LanguageModelError, match="Content blocked by Gemini safety filters"):
            gemini_model._handle_api_error(error, "test_method")
    
    def test_calculate_usage_with_metadata(self, gemini_model):
        """사용량 메타데이터가 있는 경우 계산 테스트."""
        mock_response = Mock()
        mock_response.usage_metadata = Mock()
        mock_response.usage_metadata.prompt_token_count = 20
        mock_response.usage_metadata.candidates_token_count = 15
        mock_response.usage_metadata.total_token_count = 35
        
        usage = gemini_model._calculate_usage("test prompt", "test response", mock_response)
        
        assert usage['prompt_tokens'] == 20
        assert usage['completion_tokens'] == 15
        assert usage['total_tokens'] == 35
    
    def test_calculate_usage_without_metadata(self, gemini_model):
        """사용량 메타데이터가 없는 경우 근사치 계산 테스트."""
        mock_response = Mock()
        # usage_metadata 속성이 없는 경우
        
        usage = gemini_model._calculate_usage("test prompt", "test response", mock_response)
        
        assert usage['prompt_tokens'] == 2  # "test prompt" = 2 words
        assert usage['completion_tokens'] == 2  # "test response" = 2 words
        assert usage['total_tokens'] == 4
    
    def test_extract_metadata(self, gemini_model):
        """메타데이터 추출 테스트."""
        mock_response = Mock()
        mock_candidate = Mock()
        mock_candidate.finish_reason = "STOP"
        mock_candidate.safety_ratings = []
        mock_response.candidates = [mock_candidate]
        
        mock_response.prompt_feedback = Mock()
        mock_response.prompt_feedback.block_reason = None
        mock_response.prompt_feedback.safety_ratings = []
        
        metadata = gemini_model._extract_metadata(mock_response)
        
        assert 'finish_reason' in metadata
        assert 'safety_ratings' in metadata
        assert 'prompt_feedback' in metadata
    
    def test_get_model_info(self, gemini_model):
        """모델 정보 반환 테스트."""
        info = gemini_model.get_model_info()
        
        assert info['provider'] == 'Google Gemini'
        assert info['model_name'] == 'gemini-pro'
        assert info['has_api_key'] is True
        assert info['temperature'] == 0.7
        assert 'supported_models' in info
        assert 'gemini-pro' in info['supported_models']
    
    def test_supported_models_list(self, gemini_model):
        """지원되는 모델 목록 테스트."""
        expected_models = [
            'gemini-pro',
            'gemini-pro-vision',
            'gemini-1.5-pro',
            'gemini-1.5-flash'
        ]
        
        assert gemini_model.supported_models == expected_models
    
    def test_model_name_validation_warning(self, gemini_model, mock_genai):
        """지원되지 않는 모델명에 대한 경고 테스트."""
        # 지원되지 않는 모델명으로 새 인스턴스 생성
        unsupported_model = GeminiLanguageModel(
            model_name="unsupported-model",
            api_key="test-key"
        )
        
        # Mock 설정
        mock_client = Mock()
        mock_response = Mock()
        mock_response.text = "Test response"
        mock_client.generate_content.return_value = mock_response
        mock_genai.GenerativeModel.return_value = mock_client
        
        # 로그 캡처를 위한 패치
        with patch('src.document_rag_english_study.llm.gemini_model.logger') as mock_logger:
            unsupported_model.initialize()
            
            # 경고 로그가 기록되었는지 확인
            mock_logger.warning.assert_called_once()
            warning_call = mock_logger.warning.call_args[0][0]
            assert "not in supported list" in warning_call