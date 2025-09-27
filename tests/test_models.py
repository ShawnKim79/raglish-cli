"""
Data Models 모듈 단위 테스트.

이 모듈은 모든 데이터 모델 클래스들의 기능을 테스트합니다.
"""

import pytest
import json
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import patch, MagicMock

# Configuration models
from src.document_rag_english_study.models.config import (
    LLMConfig, DocumentConfig, UserConfig, Configuration, SetupStatus
)

# Conversation models
from src.document_rag_english_study.models.conversation import (
    Message, LearningPoint, ConversationSession, Interaction, ConversationSummary
)

# Document models
from src.document_rag_english_study.models.document import (
    Document, IndexingResult, IndexingStatus, DocumentSummary
)

# LLM models
from src.document_rag_english_study.models.llm import (
    ErrorType, GrammarError, ImprovementSuggestion, EnglishAnalysis, LLMResponse
)

# Response models
from src.document_rag_english_study.models.response import (
    Correction, GrammarTip, VocabSuggestion, LearningFeedback, SearchResult, ConversationResponse
)


class TestLLMConfig:
    """LLMConfig 클래스 테스트."""

    def test_valid_openai_config(self):
        """유효한 OpenAI 설정 테스트."""
        config = LLMConfig(
            provider="openai",
            api_key="sk-test-key",
            model_name="gpt-3.5-turbo",
            temperature=0.7,
            max_tokens=1000
        )
        
        assert config.provider == "openai"
        assert config.api_key == "sk-test-key"
        assert config.model_name == "gpt-3.5-turbo"
        assert config.is_configured() is True

    def test_valid_gemini_config(self):
        """유효한 Gemini 설정 테스트."""
        config = LLMConfig(
            provider="gemini",
            api_key="test-gemini-key",
            model_name="gemini-pro"
        )
        
        assert config.provider == "gemini"
        assert config.is_configured() is True

    def test_valid_ollama_config(self):
        """유효한 Ollama 설정 테스트."""
        config = LLMConfig(
            provider="ollama",
            model_name="llama2",
            host="localhost:11434"
        )
        
        assert config.provider == "ollama"
        assert config.is_configured() is True

    def test_invalid_provider(self):
        """잘못된 제공업체 테스트."""
        with pytest.raises(ValueError, match="Unsupported LLM provider"):
            LLMConfig(provider="invalid_provider")

    def test_missing_api_key_openai(self):
        """OpenAI API 키 누락 테스트."""
        with pytest.raises(ValueError, match="API key is required"):
            LLMConfig(provider="openai", api_key="")

    def test_invalid_temperature(self):
        """잘못된 온도 설정 테스트."""
        with pytest.raises(ValueError, match="Temperature must be between"):
            LLMConfig(provider="openai", api_key="test", temperature=3.0)

    def test_invalid_max_tokens(self):
        """잘못된 최대 토큰 설정 테스트."""
        with pytest.raises(ValueError, match="Max tokens must be positive"):
            LLMConfig(provider="openai", api_key="test", max_tokens=-100)

    def test_default_model_names(self):
        """기본 모델명 설정 테스트."""
        openai_config = LLMConfig(provider="openai", api_key="test")
        assert openai_config.model_name == "gpt-3.5-turbo"
        
        gemini_config = LLMConfig(provider="gemini", api_key="test")
        assert gemini_config.model_name == "gemini-pro"
        
        ollama_config = LLMConfig(provider="ollama")
        assert ollama_config.model_name == "llama2"

    def test_to_dict_from_dict(self):
        """딕셔너리 변환 테스트."""
        config = LLMConfig(
            provider="openai",
            api_key="test-key",
            model_name="gpt-4",
            temperature=0.8,
            max_tokens=2000
        )
        
        config_dict = config.to_dict()
        restored_config = LLMConfig.from_dict(config_dict)
        
        assert restored_config.provider == config.provider
        assert restored_config.api_key == config.api_key
        assert restored_config.model_name == config.model_name
        assert restored_config.temperature == config.temperature
        assert restored_config.max_tokens == config.max_tokens


class TestDocumentConfig:
    """DocumentConfig 클래스 테스트."""

    def test_valid_config(self):
        """유효한 문서 설정 테스트."""
        config = DocumentConfig(
            document_directory="/test/docs",
            chunk_size=1000,
            chunk_overlap=200
        )
        
        assert config.document_directory == "/test/docs"
        assert config.chunk_size == 1000
        assert config.chunk_overlap == 200

    def test_invalid_chunk_size(self):
        """잘못된 청크 크기 테스트."""
        with pytest.raises(ValueError, match="Chunk size must be positive"):
            DocumentConfig(chunk_size=0)

    def test_invalid_chunk_overlap(self):
        """잘못된 청크 오버랩 테스트."""
        with pytest.raises(ValueError, match="Chunk overlap cannot be negative"):
            DocumentConfig(chunk_overlap=-10)

    def test_chunk_overlap_too_large(self):
        """청크 오버랩이 청크 크기보다 큰 경우 테스트."""
        with pytest.raises(ValueError, match="Chunk overlap must be less than chunk size"):
            DocumentConfig(chunk_size=100, chunk_overlap=150)

    @patch('pathlib.Path.exists')
    def test_is_configured(self, mock_exists):
        """설정 완료 상태 확인 테스트."""
        mock_exists.return_value = True
        
        config = DocumentConfig(document_directory="/test/docs")
        assert config.is_configured() is True
        
        config_no_dir = DocumentConfig()
        assert config_no_dir.is_configured() is False

    def test_to_dict_from_dict(self):
        """딕셔너리 변환 테스트."""
        config = DocumentConfig(
            document_directory="/test/docs",
            supported_formats=['pdf', 'txt'],
            chunk_size=500,
            chunk_overlap=100
        )
        
        config_dict = config.to_dict()
        restored_config = DocumentConfig.from_dict(config_dict)
        
        assert restored_config.document_directory == config.document_directory
        assert restored_config.supported_formats == config.supported_formats
        assert restored_config.chunk_size == config.chunk_size
        assert restored_config.chunk_overlap == config.chunk_overlap


class TestUserConfig:
    """UserConfig 클래스 테스트."""

    def test_valid_config(self):
        """유효한 사용자 설정 테스트."""
        config = UserConfig(
            native_language="korean",
            target_language="english",
            learning_level="intermediate"
        )
        
        assert config.native_language == "korean"
        assert config.target_language == "english"
        assert config.learning_level == "intermediate"

    def test_invalid_learning_level(self):
        """잘못된 학습 수준 테스트."""
        with pytest.raises(ValueError, match="Invalid learning level"):
            UserConfig(learning_level="expert")

    def test_invalid_feedback_level(self):
        """잘못된 피드백 수준 테스트."""
        with pytest.raises(ValueError, match="Invalid feedback level"):
            UserConfig(feedback_level="extreme")

    def test_empty_native_language(self):
        """빈 모국어 테스트."""
        with pytest.raises(ValueError, match="Native language cannot be empty"):
            UserConfig(native_language="")

    def test_to_dict_from_dict(self):
        """딕셔너리 변환 테스트."""
        config = UserConfig(
            native_language="english",
            learning_level="advanced",
            preferred_topics=["technology", "science"]
        )
        
        config_dict = config.to_dict()
        restored_config = UserConfig.from_dict(config_dict)
        
        assert restored_config.native_language == config.native_language
        assert restored_config.learning_level == config.learning_level
        assert restored_config.preferred_topics == config.preferred_topics


class TestConfiguration:
    """Configuration 클래스 테스트."""

    def test_complete_setup(self):
        """완전한 설정 테스트."""
        llm_config = LLMConfig(provider="openai", api_key="test-key")
        doc_config = DocumentConfig(document_directory="/test/docs")
        
        with patch('pathlib.Path.exists', return_value=True):
            config = Configuration(llm=llm_config, document=doc_config)
            assert config.is_setup_complete() is True

    def test_incomplete_setup(self):
        """불완전한 설정 테스트."""
        config = Configuration()
        assert config.is_setup_complete() is False

    def test_setup_status(self):
        """설정 상태 확인 테스트."""
        config = Configuration()
        status = config.get_setup_status()
        
        assert isinstance(status, SetupStatus)
        assert status.overall_complete is False

    def test_to_dict_from_dict(self):
        """딕셔너리 변환 테스트."""
        llm_config = LLMConfig(provider="openai", api_key="test-key")
        config = Configuration(llm=llm_config, version="2.0.0")
        
        config_dict = config.to_dict()
        restored_config = Configuration.from_dict(config_dict)
        
        assert restored_config.version == config.version
        assert restored_config.llm.provider == config.llm.provider


class TestSetupStatus:
    """SetupStatus 클래스 테스트."""

    def test_completion_percentage(self):
        """완료 비율 계산 테스트."""
        status = SetupStatus(
            llm_configured=True,
            documents_configured=True,
            user_configured=False
        )
        
        assert status.get_completion_percentage() == 66.66666666666666

    def test_missing_steps(self):
        """누락된 단계 확인 테스트."""
        status = SetupStatus(
            llm_configured=False,
            documents_configured=True,
            user_configured=False
        )
        
        missing = status.get_missing_steps()
        assert "LLM configuration" in missing
        assert "User preferences" in missing
        assert "Document directory setup" not in missing

    def test_to_dict(self):
        """딕셔너리 변환 테스트."""
        status = SetupStatus(llm_configured=True, documents_configured=False)
        status_dict = status.to_dict()
        
        assert 'completion_percentage' in status_dict
        assert 'missing_steps' in status_dict
        assert status_dict['llm_configured'] is True
        assert status_dict['documents_configured'] is False


class TestMessage:
    """Message 클래스 테스트."""

    def test_valid_message(self):
        """유효한 메시지 테스트."""
        message = Message(role="user", content="Hello, how are you?")
        
        assert message.role == "user"
        assert message.content == "Hello, how are you?"
        assert isinstance(message.timestamp, datetime)

    def test_invalid_role(self):
        """잘못된 역할 테스트."""
        with pytest.raises(ValueError, match="Invalid role"):
            Message(role="invalid", content="test")

    def test_empty_content(self):
        """빈 내용 테스트."""
        with pytest.raises(ValueError, match="Message content cannot be empty"):
            Message(role="user", content="")

    def test_to_dict_from_dict(self):
        """딕셔너리 변환 테스트."""
        message = Message(
            role="assistant",
            content="Hello there!",
            metadata={"source": "test"}
        )
        
        message_dict = message.to_dict()
        restored_message = Message.from_dict(message_dict)
        
        assert restored_message.role == message.role
        assert restored_message.content == message.content
        assert restored_message.metadata == message.metadata


class TestLearningPoint:
    """LearningPoint 클래스 테스트."""

    def test_valid_learning_point(self):
        """유효한 학습 포인트 테스트."""
        point = LearningPoint(
            topic="Present Perfect",
            description="Used for actions that started in the past and continue to the present",
            example="I have lived here for 5 years"
        )
        
        assert point.topic == "Present Perfect"
        assert point.difficulty_level == "intermediate"  # 기본값

    def test_invalid_difficulty_level(self):
        """잘못된 난이도 테스트."""
        with pytest.raises(ValueError, match="Invalid difficulty level"):
            LearningPoint(
                topic="test",
                description="test",
                difficulty_level="expert"
            )

    def test_empty_topic(self):
        """빈 주제 테스트."""
        with pytest.raises(ValueError, match="Learning point topic cannot be empty"):
            LearningPoint(topic="", description="test")

    def test_to_dict_from_dict(self):
        """딕셔너리 변환 테스트."""
        point = LearningPoint(
            topic="Conditionals",
            description="If-then statements",
            difficulty_level="advanced"
        )
        
        point_dict = point.to_dict()
        restored_point = LearningPoint.from_dict(point_dict)
        
        assert restored_point.topic == point.topic
        assert restored_point.description == point.description
        assert restored_point.difficulty_level == point.difficulty_level


class TestConversationSession:
    """ConversationSession 클래스 테스트."""

    def test_session_creation(self):
        """세션 생성 테스트."""
        session = ConversationSession()
        
        assert session.session_id is not None
        assert isinstance(session.start_time, datetime)
        assert session.is_active() is True
        assert session.end_time is None

    def test_add_message(self):
        """메시지 추가 테스트."""
        session = ConversationSession()
        message = Message(role="user", content="Hello")
        
        session.add_message(message)
        
        assert len(session.messages) == 1
        assert session.messages[0] == message

    def test_end_session(self):
        """세션 종료 테스트."""
        session = ConversationSession()
        session.end_session()
        
        assert session.is_active() is False
        assert session.end_time is not None
        assert isinstance(session.get_duration(), float)

    def test_to_dict_from_dict(self):
        """딕셔너리 변환 테스트."""
        session = ConversationSession(user_language="english")
        message = Message(role="user", content="Test message")
        session.add_message(message)
        
        session_dict = session.to_dict()
        restored_session = ConversationSession.from_dict(session_dict)
        
        assert restored_session.session_id == session.session_id
        assert restored_session.user_language == session.user_language
        assert len(restored_session.messages) == 1
        assert restored_session.messages[0].content == "Test message"

    def test_json_serialization(self):
        """JSON 직렬화 테스트."""
        session = ConversationSession()
        json_str = session.to_json()
        restored_session = ConversationSession.from_json(json_str)
        
        assert restored_session.session_id == session.session_id


class TestDocument:
    """Document 클래스 테스트."""

    def test_valid_document(self):
        """유효한 문서 테스트."""
        doc = Document(
            id="doc1",
            title="Test Document",
            file_path="/test/doc.txt",
            content="This is a test document with some content.",
            file_type="txt"
        )
        
        assert doc.id == "doc1"
        assert doc.word_count == 8  # 자동 계산됨 (실제 단어 수)
        assert doc.file_hash != ""  # 자동 생성됨

    def test_invalid_file_type(self):
        """잘못된 파일 타입 테스트."""
        with pytest.raises(ValueError, match="Unsupported file type"):
            Document(
                id="doc1",
                title="Test",
                file_path="/test/doc.xyz",
                content="test",
                file_type="xyz"
            )

    def test_empty_content(self):
        """빈 내용 테스트."""
        with pytest.raises(ValueError, match="Document content cannot be empty"):
            Document(
                id="doc1",
                title="Test",
                file_path="/test/doc.txt",
                content="",
                file_type="txt"
            )

    def test_get_summary(self):
        """요약 생성 테스트."""
        long_content = "This is a very long document " * 20
        doc = Document(
            id="doc1",
            title="Long Document",
            file_path="/test/long.txt",
            content=long_content,
            file_type="txt"
        )
        
        summary = doc.get_summary(max_length=50)
        assert len(summary) <= 53  # 50 + "..."
        assert summary.endswith("...")

    @patch('pathlib.Path.stat')
    def test_get_file_size(self, mock_stat):
        """파일 크기 조회 테스트."""
        mock_stat.return_value.st_size = 1024
        
        doc = Document(
            id="doc1",
            title="Test",
            file_path="/test/doc.txt",
            content="test content",
            file_type="txt"
        )
        
        assert doc.get_file_size() == 1024

    def test_to_dict_from_dict(self):
        """딕셔너리 변환 테스트."""
        doc = Document(
            id="doc1",
            title="Test Document",
            file_path="/test/doc.txt",
            content="Test content",
            file_type="txt",
            language="english"
        )
        
        doc_dict = doc.to_dict()
        restored_doc = Document.from_dict(doc_dict)
        
        assert restored_doc.id == doc.id
        assert restored_doc.title == doc.title
        assert restored_doc.content == doc.content
        assert restored_doc.language == doc.language


class TestIndexingResult:
    """IndexingResult 클래스 테스트."""

    def test_successful_result(self):
        """성공적인 인덱싱 결과 테스트."""
        result = IndexingResult(success=True)
        result.add_indexed_file("/test/doc1.txt", chunks=5)
        result.add_indexed_file("/test/doc2.txt", chunks=3)
        
        assert result.success is True
        assert result.documents_processed == 2
        assert result.total_chunks == 8
        assert result.get_success_rate() == 100.0

    def test_failed_result(self):
        """실패한 인덱싱 결과 테스트."""
        result = IndexingResult(success=True)
        result.add_indexed_file("/test/doc1.txt", chunks=5)
        result.add_failed_file("/test/doc2.txt", "File not found")
        
        assert result.success is False  # 오류 추가 시 False로 변경됨
        assert len(result.errors) == 1
        assert len(result.failed_files) == 1
        assert result.get_success_rate() == 50.0

    def test_to_dict_from_dict(self):
        """딕셔너리 변환 테스트."""
        result = IndexingResult(
            success=True,
            documents_processed=5,
            processing_time=10.5
        )
        
        result_dict = result.to_dict()
        restored_result = IndexingResult.from_dict(result_dict)
        
        assert restored_result.success == result.success
        assert restored_result.documents_processed == result.documents_processed
        assert restored_result.processing_time == result.processing_time


class TestIndexingStatus:
    """IndexingStatus 클래스 테스트."""

    def test_progress_calculation(self):
        """진행률 계산 테스트."""
        status = IndexingStatus(
            total_documents=10,
            processed_documents=3
        )
        
        assert status.get_progress_percentage() == 30.0

    def test_elapsed_time(self):
        """경과 시간 계산 테스트."""
        start_time = datetime.now() - timedelta(seconds=30)
        status = IndexingStatus(start_time=start_time)
        
        elapsed = status.get_elapsed_time()
        assert elapsed is not None
        assert elapsed >= 29  # 약간의 오차 허용

    def test_to_dict(self):
        """딕셔너리 변환 테스트."""
        status = IndexingStatus(
            is_indexing=True,
            total_documents=5,
            processed_documents=2,
            current_file="/test/current.txt"
        )
        
        status_dict = status.to_dict()
        
        assert status_dict['is_indexing'] is True
        assert status_dict['progress_percentage'] == 40.0
        assert status_dict['current_file'] == "/test/current.txt"


class TestDocumentSummary:
    """DocumentSummary 클래스 테스트."""

    def test_add_document(self):
        """문서 추가 테스트."""
        summary = DocumentSummary()
        doc = Document(
            id="doc1",
            title="Test",
            file_path="/test/doc.txt",
            content="This is a test document",
            file_type="txt",
            language="english"
        )
        
        summary.add_document(doc)
        
        assert summary.total_documents == 1
        assert summary.total_words == doc.word_count
        assert summary.file_types["txt"] == 1
        assert summary.languages["english"] == 1

    def test_average_calculation(self):
        """평균 계산 테스트."""
        summary = DocumentSummary()
        
        # 빈 요약의 평균
        assert summary.get_average_words_per_document() == 0.0
        
        # 문서 추가 후 평균
        doc1 = Document(id="1", title="T1", file_path="/1", content="one two", file_type="txt")
        doc2 = Document(id="2", title="T2", file_path="/2", content="one two three four", file_type="txt")
        
        summary.add_document(doc1)
        summary.add_document(doc2)
        
        assert summary.get_average_words_per_document() == 3.0  # (2 + 4) / 2

    def test_to_dict_from_dict(self):
        """딕셔너리 변환 테스트."""
        summary = DocumentSummary(
            total_documents=5,
            total_words=1000,
            file_types={"txt": 3, "pdf": 2}
        )
        
        summary_dict = summary.to_dict()
        restored_summary = DocumentSummary.from_dict(summary_dict)
        
        assert restored_summary.total_documents == summary.total_documents
        assert restored_summary.total_words == summary.total_words
        assert restored_summary.file_types == summary.file_types


class TestErrorType:
    """ErrorType 열거형 테스트."""

    def test_error_types(self):
        """오류 타입 확인 테스트."""
        assert ErrorType.GRAMMAR.value == "grammar"
        assert ErrorType.VOCABULARY.value == "vocabulary"
        assert ErrorType.SPELLING.value == "spelling"
        assert ErrorType.PUNCTUATION.value == "punctuation"
        assert ErrorType.SYNTAX.value == "syntax"


class TestGrammarError:
    """GrammarError 클래스 테스트."""

    def test_valid_error(self):
        """유효한 문법 오류 테스트."""
        error = GrammarError(
            text="I are happy",
            error_type=ErrorType.GRAMMAR,
            position=(2, 5),
            suggestion="I am happy",
            explanation="Subject-verb agreement error"
        )
        
        assert error.text == "I are happy"
        assert error.error_type == ErrorType.GRAMMAR
        assert error.position == (2, 5)

    def test_invalid_position(self):
        """잘못된 위치 테스트."""
        with pytest.raises(ValueError, match="Invalid position range"):
            GrammarError(
                text="test",
                error_type=ErrorType.GRAMMAR,
                position=(5, 2),  # 끝이 시작보다 앞
                suggestion="test",
                explanation="test"
            )

    def test_empty_fields(self):
        """빈 필드 테스트."""
        with pytest.raises(ValueError, match="Error text cannot be empty"):
            GrammarError(
                text="",
                error_type=ErrorType.GRAMMAR,
                position=(0, 1),
                suggestion="test",
                explanation="test"
            )

    def test_to_dict_from_dict(self):
        """딕셔너리 변환 테스트."""
        error = GrammarError(
            text="wrong text",
            error_type=ErrorType.SPELLING,
            position=(0, 5),
            suggestion="right text",
            explanation="Spelling mistake"
        )
        
        error_dict = error.to_dict()
        restored_error = GrammarError.from_dict(error_dict)
        
        assert restored_error.text == error.text
        assert restored_error.error_type == error.error_type
        assert restored_error.position == error.position


class TestImprovementSuggestion:
    """ImprovementSuggestion 클래스 테스트."""

    def test_valid_suggestion(self):
        """유효한 개선 제안 테스트."""
        suggestion = ImprovementSuggestion(
            category="vocabulary",
            original="big",
            improved="enormous",
            reason="More descriptive word",
            confidence=0.9
        )
        
        assert suggestion.category == "vocabulary"
        assert suggestion.confidence == 0.9

    def test_invalid_confidence(self):
        """잘못된 신뢰도 테스트."""
        with pytest.raises(ValueError, match="Confidence must be between 0.0 and 1.0"):
            ImprovementSuggestion(
                category="test",
                original="test",
                improved="test",
                reason="test",
                confidence=1.5
            )

    def test_to_dict_from_dict(self):
        """딕셔너리 변환 테스트."""
        suggestion = ImprovementSuggestion(
            category="style",
            original="good",
            improved="excellent",
            reason="More emphatic"
        )
        
        suggestion_dict = suggestion.to_dict()
        restored_suggestion = ImprovementSuggestion.from_dict(suggestion_dict)
        
        assert restored_suggestion.category == suggestion.category
        assert restored_suggestion.original == suggestion.original
        assert restored_suggestion.improved == suggestion.improved


class TestEnglishAnalysis:
    """EnglishAnalysis 클래스 테스트."""

    def test_valid_analysis(self):
        """유효한 영어 분석 테스트."""
        analysis = EnglishAnalysis(
            vocabulary_level="advanced",
            fluency_score=0.8,
            complexity_score=0.6
        )
        
        assert analysis.vocabulary_level == "advanced"
        assert analysis.fluency_score == 0.8
        assert analysis.has_errors() is False
        assert analysis.has_suggestions() is False

    def test_invalid_vocabulary_level(self):
        """잘못된 어휘 수준 테스트."""
        with pytest.raises(ValueError, match="Invalid vocabulary level"):
            EnglishAnalysis(vocabulary_level="expert")

    def test_invalid_scores(self):
        """잘못된 점수 테스트."""
        with pytest.raises(ValueError, match="Fluency score must be between"):
            EnglishAnalysis(fluency_score=1.5)
        
        with pytest.raises(ValueError, match="Complexity score must be between"):
            EnglishAnalysis(complexity_score=-0.1)

    def test_with_errors_and_suggestions(self):
        """오류와 제안이 있는 분석 테스트."""
        error = GrammarError(
            text="test",
            error_type=ErrorType.GRAMMAR,
            position=(0, 4),
            suggestion="corrected",
            explanation="test error"
        )
        
        suggestion = ImprovementSuggestion(
            category="vocabulary",
            original="good",
            improved="excellent",
            reason="better word"
        )
        
        analysis = EnglishAnalysis(
            grammar_errors=[error],
            suggestions=[suggestion]
        )
        
        assert analysis.has_errors() is True
        assert analysis.has_suggestions() is True

    def test_to_dict_from_dict(self):
        """딕셔너리 변환 테스트."""
        analysis = EnglishAnalysis(
            vocabulary_level="beginner",
            fluency_score=0.5
        )
        
        analysis_dict = analysis.to_dict()
        restored_analysis = EnglishAnalysis.from_dict(analysis_dict)
        
        assert restored_analysis.vocabulary_level == analysis.vocabulary_level
        assert restored_analysis.fluency_score == analysis.fluency_score


class TestLLMResponse:
    """LLMResponse 클래스 테스트."""

    def test_valid_response(self):
        """유효한 LLM 응답 테스트."""
        response = LLMResponse(
            content="This is a response from the LLM",
            model="gpt-3.5-turbo",
            usage={"prompt_tokens": 10, "completion_tokens": 20},
            metadata={"temperature": 0.7}
        )
        
        assert response.content == "This is a response from the LLM"
        assert response.model == "gpt-3.5-turbo"
        assert response.usage["prompt_tokens"] == 10

    def test_empty_content(self):
        """빈 내용 테스트."""
        with pytest.raises(ValueError, match="Response content cannot be empty"):
            LLMResponse(content="", model="test-model")

    def test_empty_model(self):
        """빈 모델명 테스트."""
        with pytest.raises(ValueError, match="Model name cannot be empty"):
            LLMResponse(content="test content", model="")

    def test_to_dict_from_dict(self):
        """딕셔너리 변환 테스트."""
        response = LLMResponse(
            content="Test response",
            model="test-model",
            usage={"tokens": 50}
        )
        
        response_dict = response.to_dict()
        restored_response = LLMResponse.from_dict(response_dict)
        
        assert restored_response.content == response.content
        assert restored_response.model == response.model
        assert restored_response.usage == response.usage


class TestCorrection:
    """Correction 클래스 테스트."""

    def test_valid_correction(self):
        """유효한 교정 테스트."""
        correction = Correction(
            original_text="I are happy",
            corrected_text="I am happy",
            explanation="Subject-verb agreement",
            error_type="grammar"
        )
        
        assert correction.original_text == "I are happy"
        assert correction.corrected_text == "I am happy"
        assert correction.error_type == "grammar"

    def test_empty_fields(self):
        """빈 필드 테스트."""
        with pytest.raises(ValueError, match="Original text cannot be empty"):
            Correction(original_text="", corrected_text="test", explanation="test")

    def test_to_dict_from_dict(self):
        """딕셔너리 변환 테스트."""
        correction = Correction(
            original_text="wrong",
            corrected_text="right",
            explanation="correction explanation"
        )
        
        correction_dict = correction.to_dict()
        restored_correction = Correction.from_dict(correction_dict)
        
        assert restored_correction.original_text == correction.original_text
        assert restored_correction.corrected_text == correction.corrected_text


class TestGrammarTip:
    """GrammarTip 클래스 테스트."""

    def test_valid_tip(self):
        """유효한 문법 팁 테스트."""
        tip = GrammarTip(
            rule="Present Perfect",
            explanation="Used for actions that started in the past",
            examples=["I have lived here for 5 years"],
            difficulty_level="intermediate"
        )
        
        assert tip.rule == "Present Perfect"
        assert len(tip.examples) == 1
        assert tip.difficulty_level == "intermediate"

    def test_invalid_difficulty(self):
        """잘못된 난이도 테스트."""
        with pytest.raises(ValueError, match="Invalid difficulty level"):
            GrammarTip(
                rule="test",
                explanation="test",
                difficulty_level="expert"
            )

    def test_to_dict_from_dict(self):
        """딕셔너리 변환 테스트."""
        tip = GrammarTip(
            rule="Conditionals",
            explanation="If-then statements",
            examples=["If it rains, I will stay home"]
        )
        
        tip_dict = tip.to_dict()
        restored_tip = GrammarTip.from_dict(tip_dict)
        
        assert restored_tip.rule == tip.rule
        assert restored_tip.examples == tip.examples


class TestVocabSuggestion:
    """VocabSuggestion 클래스 테스트."""

    def test_valid_suggestion(self):
        """유효한 어휘 제안 테스트."""
        suggestion = VocabSuggestion(
            word="enormous",
            definition="extremely large",
            usage_example="The elephant was enormous",
            synonyms=["huge", "gigantic"],
            difficulty_level="advanced"
        )
        
        assert suggestion.word == "enormous"
        assert len(suggestion.synonyms) == 2
        assert suggestion.difficulty_level == "advanced"

    def test_empty_word(self):
        """빈 단어 테스트."""
        with pytest.raises(ValueError, match="Vocabulary word cannot be empty"):
            VocabSuggestion(word="", definition="test")

    def test_to_dict_from_dict(self):
        """딕셔너리 변환 테스트."""
        suggestion = VocabSuggestion(
            word="magnificent",
            definition="very beautiful",
            synonyms=["beautiful", "gorgeous"]
        )
        
        suggestion_dict = suggestion.to_dict()
        restored_suggestion = VocabSuggestion.from_dict(suggestion_dict)
        
        assert restored_suggestion.word == suggestion.word
        assert restored_suggestion.synonyms == suggestion.synonyms


class TestLearningFeedback:
    """LearningFeedback 클래스 테스트."""

    def test_empty_feedback(self):
        """빈 피드백 테스트."""
        feedback = LearningFeedback()
        
        assert feedback.has_feedback() is False
        assert len(feedback.corrections) == 0

    def test_feedback_with_content(self):
        """내용이 있는 피드백 테스트."""
        correction = Correction(
            original_text="wrong",
            corrected_text="right",
            explanation="test correction"
        )
        
        feedback = LearningFeedback(
            corrections=[correction],
            encouragement="Good job!"
        )
        
        assert feedback.has_feedback() is True
        assert len(feedback.corrections) == 1
        assert feedback.encouragement == "Good job!"

    def test_to_dict_from_dict(self):
        """딕셔너리 변환 테스트."""
        tip = GrammarTip(rule="test rule", explanation="test explanation")
        feedback = LearningFeedback(
            grammar_tips=[tip],
            encouragement="Keep it up!"
        )
        
        feedback_dict = feedback.to_dict()
        restored_feedback = LearningFeedback.from_dict(feedback_dict)
        
        assert len(restored_feedback.grammar_tips) == 1
        assert restored_feedback.encouragement == feedback.encouragement


class TestSearchResult:
    """SearchResult 클래스 테스트."""

    def test_valid_result(self):
        """유효한 검색 결과 테스트."""
        result = SearchResult(
            content="This is relevant content",
            source_file="/docs/file.txt",
            relevance_score=0.85,
            metadata={"chunk_id": 123}
        )
        
        assert result.content == "This is relevant content"
        assert result.relevance_score == 0.85
        assert result.metadata["chunk_id"] == 123

    def test_invalid_relevance_score(self):
        """잘못된 관련성 점수 테스트."""
        with pytest.raises(ValueError, match="Relevance score must be between"):
            SearchResult(
                content="test",
                source_file="test.txt",
                relevance_score=1.5
            )

    def test_to_dict_from_dict(self):
        """딕셔너리 변환 테스트."""
        result = SearchResult(
            content="search result content",
            source_file="source.txt",
            relevance_score=0.7
        )
        
        result_dict = result.to_dict()
        restored_result = SearchResult.from_dict(result_dict)
        
        assert restored_result.content == result.content
        assert restored_result.source_file == result.source_file
        assert restored_result.relevance_score == result.relevance_score


class TestConversationResponse:
    """ConversationResponse 클래스 테스트."""

    def test_simple_response(self):
        """간단한 응답 테스트."""
        response = ConversationResponse(
            response_text="Hello! How can I help you today?",
            suggested_topics=["grammar", "vocabulary"]
        )
        
        assert response.response_text == "Hello! How can I help you today?"
        assert len(response.suggested_topics) == 2
        assert response.learning_feedback is None

    def test_response_with_feedback(self):
        """피드백이 있는 응답 테스트."""
        feedback = LearningFeedback(encouragement="Great work!")
        search_result = SearchResult(
            content="relevant content",
            source_file="doc.txt",
            relevance_score=0.8
        )
        
        response = ConversationResponse(
            response_text="Here's your answer",
            learning_feedback=feedback,
            context_sources=[search_result]
        )
        
        assert response.learning_feedback is not None
        assert len(response.context_sources) == 1
        assert response.context_sources[0].relevance_score == 0.8

    def test_empty_response_text(self):
        """빈 응답 텍스트 테스트."""
        with pytest.raises(ValueError, match="Response text cannot be empty"):
            ConversationResponse(response_text="")

    def test_to_dict_from_dict(self):
        """딕셔너리 변환 테스트."""
        response = ConversationResponse(
            response_text="Test response",
            suggested_topics=["topic1", "topic2"]
        )
        
        response_dict = response.to_dict()
        restored_response = ConversationResponse.from_dict(response_dict)
        
        assert restored_response.response_text == response.response_text
        assert restored_response.suggested_topics == response.suggested_topics


class TestInteraction:
    """Interaction 클래스 테스트."""

    def test_valid_interaction(self):
        """유효한 상호작용 테스트."""
        user_msg = Message(role="user", content="Hello")
        assistant_msg = Message(role="assistant", content="Hi there!")
        
        interaction = Interaction(
            user_message=user_msg,
            assistant_message=assistant_msg,
            topics=["greeting"]
        )
        
        assert interaction.user_message.content == "Hello"
        assert interaction.assistant_message.content == "Hi there!"
        assert "greeting" in interaction.topics

    def test_to_dict_from_dict(self):
        """딕셔너리 변환 테스트."""
        user_msg = Message(role="user", content="Test user message")
        assistant_msg = Message(role="assistant", content="Test assistant response")
        
        interaction = Interaction(
            user_message=user_msg,
            assistant_message=assistant_msg
        )
        
        interaction_dict = interaction.to_dict()
        restored_interaction = Interaction.from_dict(interaction_dict)
        
        assert restored_interaction.user_message.content == interaction.user_message.content
        assert restored_interaction.assistant_message.content == interaction.assistant_message.content


class TestConversationSummary:
    """ConversationSummary 클래스 테스트."""

    def test_valid_summary(self):
        """유효한 대화 요약 테스트."""
        learning_point = LearningPoint(
            topic="Present Perfect",
            description="Grammar point learned"
        )
        
        summary = ConversationSummary(
            session_id="session123",
            duration_seconds=300.5,
            total_messages=10,
            topics_covered=["grammar", "vocabulary"],
            learning_points=[learning_point],
            key_vocabulary=["enormous", "magnificent"],
            user_progress="Good improvement in grammar"
        )
        
        assert summary.session_id == "session123"
        assert summary.duration_seconds == 300.5
        assert len(summary.learning_points) == 1
        assert len(summary.key_vocabulary) == 2

    def test_to_dict_from_dict(self):
        """딕셔너리 변환 테스트."""
        summary = ConversationSummary(
            session_id="test_session",
            duration_seconds=120.0,
            total_messages=5,
            topics_covered=["test_topic"],
            learning_points=[]  # 필수 인자 추가
        )
        
        summary_dict = summary.to_dict()
        restored_summary = ConversationSummary.from_dict(summary_dict)
        
        assert restored_summary.session_id == summary.session_id
        assert restored_summary.duration_seconds == summary.duration_seconds
        assert restored_summary.topics_covered == summary.topics_covered

    def test_json_serialization(self):
        """JSON 직렬화 테스트."""
        summary = ConversationSummary(
            session_id="json_test",
            duration_seconds=60.0,
            total_messages=3,
            topics_covered=["json"],
            learning_points=[]  # 필수 인자 추가
        )
        
        json_str = summary.to_json()
        restored_summary = ConversationSummary.from_json(json_str)
        
        assert restored_summary.session_id == summary.session_id
        assert isinstance(json_str, str)
        assert "json_test" in json_str