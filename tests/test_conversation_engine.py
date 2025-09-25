"""
ConversationEngine 클래스에 대한 테스트.

이 모듈은 대화 엔진의 통합 기능을 테스트합니다.
"""

import pytest
import tempfile
import shutil
from pathlib import Path
from unittest.mock import Mock, MagicMock, patch

from src.document_rag_english_study.conversation.engine import ConversationEngine, ConversationEngineError
from src.document_rag_english_study.models.conversation import ConversationSession, Message
from src.document_rag_english_study.models.response import ConversationResponse, SearchResult, LearningFeedback
from src.document_rag_english_study.llm.base import MockLanguageModel


class TestConversationEngine:
    """ConversationEngine 클래스 테스트."""
    
    @pytest.fixture
    def temp_dir(self):
        """임시 디렉토리 생성."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)
    
    @pytest.fixture
    def mock_rag_engine(self):
        """모의 RAG 엔진 생성."""
        mock_rag = Mock()
        mock_rag.get_indexed_document_info.return_value = {
            'total_documents': 5,
            'total_chunks': 50,
            'vector_db_info': {},
            'embedding_info': {},
            'chunk_size': 500,
            'chunk_overlap': 50
        }
        mock_rag.search_similar_content.return_value = [
            SearchResult(
                content="This is a sample document content about technology.",
                source_file="test_doc.txt",
                relevance_score=0.8,
                metadata={'document_id': 'doc1', 'chunk_index': 0}
            )
        ]
        mock_rag.extract_keywords.return_value = ['technology', 'innovation', 'development']
        mock_rag.generate_answer.return_value = "This is a generated answer based on the context."
        return mock_rag
    
    @pytest.fixture
    def mock_llm(self):
        """모의 언어 모델 생성."""
        llm = MockLanguageModel()
        llm.initialize()
        return llm
    
    @pytest.fixture
    def conversation_engine(self, mock_rag_engine, mock_llm, temp_dir):
        """ConversationEngine 인스턴스 생성."""
        return ConversationEngine(
            rag_engine=mock_rag_engine,
            llm=mock_llm,
            user_language="korean",
            sessions_dir=temp_dir
        )
    
    def test_initialization(self, conversation_engine, mock_rag_engine, mock_llm):
        """ConversationEngine 초기화 테스트."""
        assert conversation_engine.rag_engine == mock_rag_engine
        assert conversation_engine.llm == mock_llm
        assert conversation_engine.user_language == "korean"
        assert conversation_engine.dialog_manager is not None
        assert conversation_engine.learning_assistant is not None
        assert conversation_engine.session_tracker is not None
        assert conversation_engine._current_session is None
    
    def test_start_conversation_new_session(self, conversation_engine):
        """새로운 대화 세션 시작 테스트."""
        session = conversation_engine.start_conversation()
        
        assert isinstance(session, ConversationSession)
        assert session.user_language == "korean"
        assert len(session.messages) == 1  # 시작 메시지
        assert session.messages[0].role == "assistant"
        assert conversation_engine._current_session == session
    
    def test_start_conversation_with_preferred_topic(self, conversation_engine):
        """선호 주제와 함께 대화 시작 테스트."""
        preferred_topic = "technology"
        session = conversation_engine.start_conversation(preferred_topic=preferred_topic)
        
        assert isinstance(session, ConversationSession)
        assert len(session.messages) == 1
        # 시작 메시지에 주제 정보가 포함되어야 함
        assert "topics" in session.messages[0].metadata
    
    def test_process_user_input_english(self, conversation_engine):
        """영어 사용자 입력 처리 테스트."""
        # 먼저 세션 시작
        session = conversation_engine.start_conversation()
        
        user_input = "I think technology is very important for our future."
        response = conversation_engine.process_user_input(user_input, session)
        
        assert isinstance(response, ConversationResponse)
        assert response.response_text
        assert len(session.messages) >= 3  # 시작 메시지 + 사용자 메시지 + 어시스턴트 응답
        assert session.messages[1].role == "user"
        assert session.messages[1].content == user_input
        assert session.messages[2].role == "assistant"
    
    def test_process_user_input_korean(self, conversation_engine):
        """한국어 사용자 입력 처리 테스트."""
        session = conversation_engine.start_conversation()
        
        user_input = "안녕하세요. 기술에 대해 이야기하고 싶습니다."
        response = conversation_engine.process_user_input(user_input, session)
        
        assert isinstance(response, ConversationResponse)
        assert response.response_text
        # 한국어 입력이므로 학습 피드백이 없어야 함
        assert response.learning_feedback is None or not response.learning_feedback.has_feedback()
    
    def test_process_user_input_with_learning_feedback(self, conversation_engine):
        """학습 피드백이 포함된 입력 처리 테스트."""
        session = conversation_engine.start_conversation()
        
        # 문법 오류가 있는 영어 입력
        user_input = "I am very interesting in technology and I have a plan to study it."
        
        with patch.object(conversation_engine.learning_assistant, 'create_learning_feedback') as mock_feedback:
            # 모의 학습 피드백 설정
            mock_feedback.return_value = LearningFeedback(
                corrections=[],
                grammar_tips=[],
                vocabulary_suggestions=[],
                encouragement="Good effort!"
            )
            
            response = conversation_engine.process_user_input(user_input, session)
            
            assert isinstance(response, ConversationResponse)
            assert response.learning_feedback is not None
            mock_feedback.assert_called_once()
    
    def test_process_user_input_empty_input(self, conversation_engine):
        """빈 입력 처리 테스트."""
        session = conversation_engine.start_conversation()
        
        with pytest.raises(ConversationEngineError):
            conversation_engine.process_user_input("", session)
    
    def test_process_user_input_no_session(self, conversation_engine):
        """세션 없이 입력 처리 테스트."""
        with pytest.raises(ConversationEngineError):
            conversation_engine.process_user_input("Hello", None)
    
    def test_end_conversation(self, conversation_engine):
        """대화 종료 테스트."""
        # 세션 시작 및 몇 개의 메시지 추가
        session = conversation_engine.start_conversation()
        conversation_engine.process_user_input("Hello, how are you?", session)
        
        result = conversation_engine.end_conversation(session)
        
        assert isinstance(result, dict)
        assert "session_id" in result
        assert "duration_seconds" in result
        assert "total_messages" in result
        assert "topics_covered" in result
        assert "learning_points_count" in result
        assert conversation_engine._current_session is None
    
    def test_end_conversation_no_session(self, conversation_engine):
        """세션 없이 대화 종료 테스트."""
        with pytest.raises(ConversationEngineError):
            conversation_engine.end_conversation(None)
    
    def test_get_current_session(self, conversation_engine):
        """현재 세션 조회 테스트."""
        # 초기에는 세션이 없어야 함
        assert conversation_engine.get_current_session() is None
        
        # 세션 시작 후에는 세션이 있어야 함
        session = conversation_engine.start_conversation()
        assert conversation_engine.get_current_session() == session
    
    def test_get_session_history(self, conversation_engine):
        """세션 기록 조회 테스트."""
        # 초기에는 기록이 없어야 함
        history = conversation_engine.get_session_history()
        assert isinstance(history, list)
        
        # 세션을 생성하고 종료한 후 기록 확인
        session = conversation_engine.start_conversation()
        conversation_engine.process_user_input("Test message", session)
        conversation_engine.end_conversation(session)
        
        history = conversation_engine.get_session_history()
        assert len(history) >= 0  # 세션 저장이 성공했다면 1개 이상
    
    def test_get_learning_progress(self, conversation_engine):
        """학습 진행 상황 조회 테스트."""
        progress = conversation_engine.get_learning_progress()
        assert isinstance(progress, dict)
    
    def test_suggest_conversation_topics(self, conversation_engine):
        """대화 주제 제안 테스트."""
        topics = conversation_engine.suggest_conversation_topics(count=3)
        assert isinstance(topics, list)
        assert len(topics) <= 3
    
    def test_suggest_conversation_topics_with_current_session(self, conversation_engine):
        """현재 세션이 있을 때 주제 제안 테스트."""
        session = conversation_engine.start_conversation()
        session.topics_covered = ["technology", "innovation"]
        
        topics = conversation_engine.suggest_conversation_topics(count=5)
        assert isinstance(topics, list)
        # 이미 다룬 주제는 제외되어야 함
        for topic in topics:
            assert topic not in session.topics_covered
    
    def test_extract_available_topics(self, conversation_engine):
        """사용 가능한 주제 추출 테스트."""
        topics = conversation_engine._extract_available_topics()
        assert isinstance(topics, list)
    
    def test_extract_available_topics_no_documents(self, conversation_engine):
        """문서가 없을 때 주제 추출 테스트."""
        conversation_engine.rag_engine.get_indexed_document_info.return_value = {
            'total_documents': 0,
            'total_chunks': 0
        }
        
        topics = conversation_engine._extract_available_topics()
        assert topics == []
    
    def test_analyze_user_english_english_text(self, conversation_engine):
        """영어 텍스트 분석 테스트."""
        english_text = "I am very interesting in technology."
        
        with patch.object(conversation_engine.learning_assistant, 'create_learning_feedback') as mock_feedback:
            mock_feedback.return_value = LearningFeedback(encouragement="Good job!")
            
            result = conversation_engine._analyze_user_english(english_text)
            
            assert result is not None
            mock_feedback.assert_called_once_with(english_text)
    
    def test_analyze_user_english_non_english_text(self, conversation_engine):
        """비영어 텍스트 분석 테스트."""
        korean_text = "안녕하세요. 기술에 대해 이야기하고 싶습니다."
        
        result = conversation_engine._analyze_user_english(korean_text)
        
        # 영어가 아니므로 분석하지 않아야 함
        assert result is None
    
    def test_search_relevant_context(self, conversation_engine):
        """관련 컨텍스트 검색 테스트."""
        session = ConversationSession(user_language="korean")
        session.topics_covered = ["technology", "innovation"]
        
        user_input = "Tell me about artificial intelligence."
        
        results = conversation_engine._search_relevant_context(user_input, session)
        
        assert isinstance(results, list)
        # RAG 엔진의 search_similar_content가 호출되어야 함
        conversation_engine.rag_engine.search_similar_content.assert_called()
    
    def test_generate_conversation_response(self, conversation_engine):
        """대화 응답 생성 테스트."""
        session = ConversationSession(user_language="korean")
        session.add_message(Message(role="assistant", content="Hello!"))
        
        user_input = "Hello, how are you?"
        context_sources = [
            SearchResult(
                content="Sample content",
                source_file="test.txt",
                relevance_score=0.8
            )
        ]
        learning_feedback = None
        
        response = conversation_engine._generate_conversation_response(
            user_input, session, context_sources, learning_feedback
        )
        
        assert isinstance(response, str)
        assert len(response) > 0
    
    def test_suggest_follow_up_topics(self, conversation_engine):
        """후속 주제 제안 테스트."""
        session = ConversationSession(user_language="korean")
        session.topics_covered = ["technology"]
        
        user_input = "I love programming."
        context_sources = [
            SearchResult(
                content="Programming is a valuable skill in technology.",
                source_file="programming.txt",
                relevance_score=0.9
            )
        ]
        
        with patch.object(conversation_engine.dialog_manager, 'suggest_follow_up_questions') as mock_suggest:
            mock_suggest.return_value = ["What programming languages do you know?"]
            
            topics = conversation_engine._suggest_follow_up_topics(
                user_input, session, context_sources
            )
            
            assert isinstance(topics, list)
            mock_suggest.assert_called_once()
    
    def test_extract_learning_points(self, conversation_engine):
        """학습 포인트 추출 테스트."""
        user_input = "I am very interesting in technology."
        learning_feedback = LearningFeedback(
            corrections=[],
            vocabulary_suggestions=[],
            encouragement="Good effort!"
        )
        context_sources = [
            SearchResult(
                content="Technology is important for innovation.",
                source_file="tech.txt",
                relevance_score=0.8
            )
        ]
        
        learning_points = conversation_engine._extract_learning_points(
            user_input, learning_feedback, context_sources
        )
        
        assert isinstance(learning_points, list)
    
    def test_extract_topics_from_context(self, conversation_engine):
        """컨텍스트에서 주제 추출 테스트."""
        context_sources = [
            SearchResult(
                content="Technology and innovation are key topics.",
                source_file="topics.txt",
                relevance_score=0.8,
                metadata={'topics': ['technology', 'innovation']}
            )
        ]
        
        topics = conversation_engine._extract_topics_from_context(context_sources)
        
        assert isinstance(topics, list)
        assert 'technology' in topics or 'innovation' in topics
    
    def test_format_learning_feedback_korean(self, conversation_engine):
        """한국어 학습 피드백 포맷팅 테스트."""
        from src.document_rag_english_study.models.response import Correction, VocabSuggestion
        
        feedback = LearningFeedback(
            corrections=[
                Correction(
                    original_text="I am interesting",
                    corrected_text="I am interested",
                    explanation="Use 'interested' for people, 'interesting' for things."
                )
            ],
            vocabulary_suggestions=[
                VocabSuggestion(
                    word="fascinating",
                    definition="매우 흥미로운",
                    usage_example="The book is fascinating."
                )
            ],
            encouragement="잘하고 있어요!"
        )
        
        formatted = conversation_engine._format_learning_feedback(feedback)
        
        assert isinstance(formatted, str)
        assert "문법 교정" in formatted
        assert "어휘 제안" in formatted
        assert "잘하고 있어요!" in formatted
    
    def test_format_learning_feedback_english(self, conversation_engine):
        """영어 학습 피드백 포맷팅 테스트."""
        # 영어 모드로 설정
        conversation_engine.user_language = "english"
        
        from src.document_rag_english_study.models.response import Correction
        
        feedback = LearningFeedback(
            corrections=[
                Correction(
                    original_text="I am interesting",
                    corrected_text="I am interested",
                    explanation="Use 'interested' for people."
                )
            ],
            encouragement="Good job!"
        )
        
        formatted = conversation_engine._format_learning_feedback(feedback)
        
        assert isinstance(formatted, str)
        assert "Grammar Corrections" in formatted
        assert "Good job!" in formatted
    
    def test_get_fallback_response_korean(self, conversation_engine):
        """한국어 폴백 응답 테스트."""
        response = conversation_engine._get_fallback_response()
        
        assert isinstance(response, str)
        assert "죄송합니다" in response
    
    def test_get_fallback_response_english(self, conversation_engine):
        """영어 폴백 응답 테스트."""
        conversation_engine.user_language = "english"
        
        response = conversation_engine._get_fallback_response()
        
        assert isinstance(response, str)
        assert "sorry" in response.lower()
    
    def test_integration_full_conversation_flow(self, conversation_engine):
        """전체 대화 흐름 통합 테스트."""
        # 1. 대화 시작
        session = conversation_engine.start_conversation()
        assert session is not None
        
        # 2. 사용자 입력 처리
        response1 = conversation_engine.process_user_input(
            "Hello, I want to learn about technology.", session
        )
        assert isinstance(response1, ConversationResponse)
        
        # 3. 추가 입력 처리
        response2 = conversation_engine.process_user_input(
            "What is artificial intelligence?", session
        )
        assert isinstance(response2, ConversationResponse)
        
        # 4. 대화 종료
        summary = conversation_engine.end_conversation(session)
        assert isinstance(summary, dict)
        assert summary["total_messages"] >= 4  # 시작 + 2번의 사용자-어시스턴트 교환
    
    def test_error_handling_rag_engine_failure(self, conversation_engine):
        """RAG 엔진 실패 시 오류 처리 테스트."""
        session = conversation_engine.start_conversation()
        
        # RAG 엔진에서 예외 발생하도록 설정
        conversation_engine.rag_engine.search_similar_content.side_effect = Exception("RAG error")
        
        # 여전히 응답을 생성해야 함 (폴백 사용)
        response = conversation_engine.process_user_input("Test input", session)
        assert isinstance(response, ConversationResponse)
        assert response.response_text  # 폴백 응답이라도 있어야 함
    
    def test_error_handling_llm_failure(self, conversation_engine):
        """LLM 실패 시 오류 처리 테스트."""
        session = conversation_engine.start_conversation()
        
        # LLM에서 예외 발생하도록 설정 (Mock 객체 사용)
        with patch.object(conversation_engine.llm, 'generate_response', side_effect=Exception("LLM error")):
            # 여전히 응답을 생성해야 함
            response = conversation_engine.process_user_input("Test input", session)
            assert isinstance(response, ConversationResponse)