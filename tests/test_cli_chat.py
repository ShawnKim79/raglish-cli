"""
CLI chat 명령어 테스트.

이 모듈은 대화형 학습 명령어의 기능을 테스트합니다.
"""

import pytest
import tempfile
import os
from unittest.mock import Mock, patch, MagicMock
from click.testing import CliRunner

from src.document_rag_english_study.cli.interface import cli
from src.document_rag_english_study.models.conversation import ConversationSession, Message
from src.document_rag_english_study.models.response import ConversationResponse, LearningFeedback


class TestChatCommand:
    """Chat 명령어 테스트 클래스."""
    
    def setup_method(self):
        """각 테스트 메서드 실행 전 설정."""
        self.runner = CliRunner()
        self.temp_dir = tempfile.mkdtemp()
    
    def teardown_method(self):
        """각 테스트 메서드 실행 후 정리."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    @patch('src.document_rag_english_study.cli.interface.ConfigurationManager')
    def test_chat_command_not_configured(self, mock_config_manager):
        """설정이 완료되지 않은 경우 테스트."""
        # Mock 설정 - 설정이 완료되지 않음
        mock_manager = Mock()
        mock_setup_status = Mock()
        mock_setup_status.overall_complete = False
        mock_manager.get_setup_status.return_value = mock_setup_status
        mock_config_manager.return_value = mock_manager
        
        # chat 명령어 실행
        result = self.runner.invoke(cli, ['chat'])
        
        # 결과 검증
        assert result.exit_code == 0
        assert "설정이 완료되지 않았습니다" in result.output
        assert "setup" in result.output
    
    @patch('src.document_rag_english_study.cli.interface._start_interactive_chat_session')
    @patch('src.document_rag_english_study.cli.interface.ConfigurationManager')
    def test_chat_command_configured(self, mock_config_manager, mock_start_session):
        """설정이 완료된 경우 테스트."""
        # Mock 설정 - 설정이 완료됨
        mock_manager = Mock()
        mock_setup_status = Mock()
        mock_setup_status.overall_complete = True
        mock_manager.get_setup_status.return_value = mock_setup_status
        mock_config_manager.return_value = mock_manager
        
        # chat 명령어 실행
        result = self.runner.invoke(cli, ['chat'])
        
        # 결과 검증
        assert result.exit_code == 0
        mock_start_session.assert_called_once()
    
    @patch('src.document_rag_english_study.cli.interface._start_interactive_chat_session')
    @patch('src.document_rag_english_study.cli.interface.ConfigurationManager')
    def test_chat_command_with_options(self, mock_config_manager, mock_start_session):
        """옵션과 함께 chat 명령어 테스트."""
        # Mock 설정
        mock_manager = Mock()
        mock_setup_status = Mock()
        mock_setup_status.overall_complete = True
        mock_manager.get_setup_status.return_value = mock_setup_status
        mock_config_manager.return_value = mock_manager
        
        # 옵션과 함께 chat 명령어 실행
        result = self.runner.invoke(cli, [
            'chat', 
            '--session-id', 'test-session-123',
            '--topic', 'artificial intelligence',
            '--no-save-session'
        ])
        
        # 결과 검증
        assert result.exit_code == 0
        mock_start_session.assert_called_once_with(
            mock_manager, 'test-session-123', 'artificial intelligence', False
        )
    
    @patch('src.document_rag_english_study.cli.interface._start_interactive_chat_session')
    @patch('src.document_rag_english_study.cli.interface.ConfigurationManager')
    def test_chat_command_keyboard_interrupt(self, mock_config_manager, mock_start_session):
        """키보드 인터럽트 처리 테스트."""
        # Mock 설정
        mock_manager = Mock()
        mock_setup_status = Mock()
        mock_setup_status.overall_complete = True
        mock_manager.get_setup_status.return_value = mock_setup_status
        mock_config_manager.return_value = mock_manager
        
        # KeyboardInterrupt 발생 시뮬레이션
        mock_start_session.side_effect = KeyboardInterrupt()
        
        # chat 명령어 실행
        result = self.runner.invoke(cli, ['chat'])
        
        # 결과 검증
        assert result.exit_code == 0
        assert "대화를 종료합니다" in result.output
    
    @patch('src.document_rag_english_study.cli.interface._start_interactive_chat_session')
    @patch('src.document_rag_english_study.cli.interface.ConfigurationManager')
    def test_chat_command_exception(self, mock_config_manager, mock_start_session):
        """예외 처리 테스트."""
        # Mock 설정
        mock_manager = Mock()
        mock_setup_status = Mock()
        mock_setup_status.overall_complete = True
        mock_manager.get_setup_status.return_value = mock_setup_status
        mock_config_manager.return_value = mock_manager
        
        # 예외 발생 시뮬레이션
        mock_start_session.side_effect = Exception("Test error")
        
        # chat 명령어 실행
        result = self.runner.invoke(cli, ['chat'])
        
        # 결과 검증
        assert result.exit_code == 1
        assert "대화 시작 중 오류 발생" in result.output


class TestInteractiveChatSession:
    """대화형 세션 테스트 클래스."""
    
    def setup_method(self):
        """각 테스트 메서드 실행 전 설정."""
        self.temp_dir = tempfile.mkdtemp()
    
    def teardown_method(self):
        """각 테스트 메서드 실행 후 정리."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    @patch('src.document_rag_english_study.cli.interface.ConversationEngine')
    @patch('src.document_rag_english_study.cli.interface.RAGEngine')
    @patch('src.document_rag_english_study.cli.interface.create_language_model')
    @patch('src.document_rag_english_study.cli.interface._run_conversation_loop')
    @patch('src.document_rag_english_study.cli.interface._show_chat_welcome_message')
    def test_start_interactive_chat_session_new(
        self, mock_welcome, mock_run_loop, mock_create_llm, 
        mock_rag_engine, mock_conversation_engine
    ):
        """새로운 대화 세션 시작 테스트."""
        from src.document_rag_english_study.cli.interface import _start_interactive_chat_session
        from src.document_rag_english_study.config import ConfigurationManager
        
        # Mock 설정
        mock_config_manager = Mock()
        mock_config = Mock()
        mock_config.llm = Mock()
        mock_config.user.native_language = "korean"
        mock_config_manager.get_config.return_value = mock_config
        
        mock_llm = Mock()
        mock_create_llm.return_value = mock_llm
        
        mock_rag = Mock()
        mock_rag_engine.return_value = mock_rag
        
        mock_engine = Mock()
        mock_session = Mock()
        mock_session.messages = [Mock(role="assistant", content="Hello!")]
        mock_engine.start_conversation.return_value = mock_session
        mock_conversation_engine.return_value = mock_engine
        
        # 함수 실행
        _start_interactive_chat_session(mock_config_manager)
        
        # 검증
        mock_create_llm.assert_called_once_with(mock_config.llm)
        mock_rag_engine.assert_called_once()
        mock_conversation_engine.assert_called_once_with(
            rag_engine=mock_rag,
            llm=mock_llm,
            user_language="korean"
        )
        mock_engine.start_conversation.assert_called_once_with(
            preferred_topic=None,
            session_id=None
        )
        mock_welcome.assert_called_once_with("korean")
        mock_run_loop.assert_called_once()
    
    @patch('src.document_rag_english_study.cli.interface.ConversationEngine')
    @patch('src.document_rag_english_study.cli.interface.RAGEngine')
    @patch('src.document_rag_english_study.cli.interface.create_language_model')
    @patch('src.document_rag_english_study.cli.interface._run_conversation_loop')
    @patch('src.document_rag_english_study.cli.interface._show_chat_welcome_message')
    def test_start_interactive_chat_session_with_options(
        self, mock_welcome, mock_run_loop, mock_create_llm, 
        mock_rag_engine, mock_conversation_engine
    ):
        """옵션과 함께 대화 세션 시작 테스트."""
        from src.document_rag_english_study.cli.interface import _start_interactive_chat_session
        
        # Mock 설정
        mock_config_manager = Mock()
        mock_config = Mock()
        mock_config.llm = Mock()
        mock_config.user.native_language = "korean"
        mock_config_manager.get_config.return_value = mock_config
        
        mock_llm = Mock()
        mock_create_llm.return_value = mock_llm
        
        mock_rag = Mock()
        mock_rag_engine.return_value = mock_rag
        
        mock_engine = Mock()
        mock_session = Mock()
        mock_session.messages = []
        mock_engine.start_conversation.return_value = mock_session
        mock_conversation_engine.return_value = mock_engine
        
        # 함수 실행
        _start_interactive_chat_session(
            mock_config_manager,
            session_id="test-session",
            preferred_topic="AI",
            save_session=False
        )
        
        # 검증
        mock_engine.start_conversation.assert_called_once_with(
            preferred_topic="AI",
            session_id="test-session"
        )
        mock_run_loop.assert_called_once()
    
    def test_show_chat_welcome_message_korean(self):
        """한국어 환영 메시지 테스트."""
        from src.document_rag_english_study.cli.interface import _show_chat_welcome_message
        from click.testing import CliRunner
        import click
        
        runner = CliRunner()
        
        @click.command()
        def test_command():
            _show_chat_welcome_message("korean")
        
        result = runner.invoke(test_command)
        
        assert "Document RAG English Study - 대화형 학습" in result.output
        assert "사용법:" in result.output
        assert "/help" in result.output
        assert "/quit" in result.output
    
    def test_show_chat_welcome_message_english(self):
        """영어 환영 메시지 테스트."""
        from src.document_rag_english_study.cli.interface import _show_chat_welcome_message
        from click.testing import CliRunner
        import click
        
        runner = CliRunner()
        
        @click.command()
        def test_command():
            _show_chat_welcome_message("english")
        
        result = runner.invoke(test_command)
        
        assert "Document RAG English Study - Interactive Learning" in result.output
        assert "How to use:" in result.output
        assert "/help" in result.output
        assert "/quit" in result.output


class TestSpecialCommands:
    """특수 명령어 테스트 클래스."""
    
    def test_handle_special_command_quit(self):
        """종료 명령어 테스트."""
        from src.document_rag_english_study.cli.interface import _handle_special_command
        
        mock_engine = Mock()
        mock_session = Mock()
        
        # 다양한 종료 명령어 테스트
        assert _handle_special_command("/quit", mock_engine, mock_session) is True
        assert _handle_special_command("/exit", mock_engine, mock_session) is True
        assert _handle_special_command("/q", mock_engine, mock_session) is True
    
    def test_handle_special_command_help(self):
        """도움말 명령어 테스트."""
        from src.document_rag_english_study.cli.interface import _handle_special_command
        from click.testing import CliRunner
        import click
        
        runner = CliRunner()
        
        @click.command()
        def test_command():
            mock_engine = Mock()
            mock_session = Mock()
            result = _handle_special_command("/help", mock_engine, mock_session)
            click.echo(f"Result: {result}")
        
        result = runner.invoke(test_command)
        assert "Result: False" in result.output
    
    @patch('src.document_rag_english_study.cli.interface._show_topic_suggestions')
    def test_handle_special_command_topics(self, mock_show_topics):
        """주제 제안 명령어 테스트."""
        from src.document_rag_english_study.cli.interface import _handle_special_command
        
        mock_engine = Mock()
        mock_session = Mock()
        
        result = _handle_special_command("/topics", mock_engine, mock_session)
        
        assert result is False
        mock_show_topics.assert_called_once_with(mock_engine)
    
    @patch('src.document_rag_english_study.cli.interface._show_learning_progress')
    def test_handle_special_command_progress(self, mock_show_progress):
        """학습 진행 상황 명령어 테스트."""
        from src.document_rag_english_study.cli.interface import _handle_special_command
        
        mock_engine = Mock()
        mock_session = Mock()
        
        result = _handle_special_command("/progress", mock_engine, mock_session)
        
        assert result is False
        mock_show_progress.assert_called_once_with(mock_engine, mock_session)
    
    def test_handle_special_command_unknown(self):
        """알 수 없는 명령어 테스트."""
        from src.document_rag_english_study.cli.interface import _handle_special_command
        from click.testing import CliRunner
        import click
        
        runner = CliRunner()
        
        @click.command()
        def test_command():
            mock_engine = Mock()
            mock_session = Mock()
            result = _handle_special_command("/unknown", mock_engine, mock_session)
            click.echo(f"Result: {result}")
        
        result = runner.invoke(test_command)
        assert "Result: False" in result.output
        assert "알 수 없는 명령어" in result.output


class TestDisplayFunctions:
    """표시 함수 테스트 클래스."""
    
    def test_display_assistant_message(self):
        """어시스턴트 메시지 표시 테스트."""
        from src.document_rag_english_study.cli.interface import _display_assistant_message
        from click.testing import CliRunner
        import click
        
        runner = CliRunner()
        
        @click.command()
        def test_command():
            _display_assistant_message("Hello, how can I help you?")
        
        result = runner.invoke(test_command)
        
        assert "🤖 Assistant: Hello, how can I help you?" in result.output
    
    def test_display_conversation_response_simple(self):
        """간단한 대화 응답 표시 테스트."""
        from src.document_rag_english_study.cli.interface import _display_conversation_response
        from src.document_rag_english_study.models.response import ConversationResponse
        from click.testing import CliRunner
        import click
        
        runner = CliRunner()
        
        @click.command()
        def test_command():
            response = ConversationResponse(
                response_text="This is a test response.",
                suggested_topics=["AI", "Machine Learning", "Technology"]
            )
            _display_conversation_response(response)
        
        result = runner.invoke(test_command)
        
        assert "🤖 Assistant: This is a test response." in result.output
        assert "💡 다음 주제도 이야기해보세요: AI, Machine Learning, Technology" in result.output
    
    def test_display_conversation_response_with_feedback(self):
        """피드백이 있는 대화 응답 표시 테스트."""
        from src.document_rag_english_study.cli.interface import _display_conversation_response
        from src.document_rag_english_study.models.response import (
            ConversationResponse, LearningFeedback, Correction, VocabSuggestion
        )
        from click.testing import CliRunner
        import click
        
        runner = CliRunner()
        
        @click.command()
        def test_command():
            # 학습 피드백 생성
            feedback = LearningFeedback(
                corrections=[
                    Correction(
                        original_text="I are happy",
                        corrected_text="I am happy",
                        explanation="Use 'am' with 'I'"
                    )
                ],
                vocabulary_suggestions=[
                    VocabSuggestion(
                        word="delighted",
                        definition="feeling very pleased",
                        usage_example="I am delighted to meet you."
                    )
                ],
                encouragement="Great job practicing English!"
            )
            
            response = ConversationResponse(
                response_text="That's wonderful!",
                learning_feedback=feedback
            )
            _display_conversation_response(response)
        
        result = runner.invoke(test_command)
        
        assert "🤖 Assistant: That's wonderful!" in result.output
        assert "📚 학습 피드백:" in result.output
        assert "📝 문법 교정:" in result.output
        assert "I are happy → I am happy" in result.output
        assert "📖 어휘 제안:" in result.output
        assert "delighted: feeling very pleased" in result.output
        assert "💪 Great job practicing English!" in result.output