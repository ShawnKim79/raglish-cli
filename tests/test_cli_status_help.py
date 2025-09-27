"""
CLI Status and Help Commands 테스트.

이 모듈은 CLI의 status와 help 명령어 기능을 테스트합니다.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from click.testing import CliRunner
import json

from src.document_rag_english_study.cli.interface import cli
from src.document_rag_english_study.models.config import Configuration, SetupStatus


class TestStatusCommand:
    """Status 명령어 테스트 클래스."""
    
    def setup_method(self):
        """각 테스트 메서드 실행 전 설정."""
        self.runner = CliRunner()
    
    @patch('src.document_rag_english_study.cli.interface.ConfigurationManager')
    def test_status_command_basic(self, mock_config_manager):
        """기본 status 명령어 테스트."""
        # Mock 설정
        mock_manager = Mock()
        mock_config_manager.return_value = mock_manager
        
        mock_setup_status = SetupStatus(
            llm_configured=True,
            documents_configured=True,
            user_configured=True,
            overall_complete=True
        )
        mock_manager.get_setup_status.return_value = mock_setup_status
        
        mock_config = Configuration()
        mock_config.user.native_language = "korean"
        mock_config.document.document_directory = "/test/docs"
        mock_config.llm = Mock()
        mock_config.llm.provider = "openai"
        mock_manager.get_config.return_value = mock_config
        
        # 명령어 실행
        result = self.runner.invoke(cli, ['status'])
        
        # 검증
        assert result.exit_code == 0
        assert "시스템 상태" in result.output
        assert "모든 설정이 완료되었습니다" in result.output
        assert "모국어" in result.output
        assert "문서 디렉토리" in result.output
        assert "LLM 제공업체" in result.output
    
    @patch('src.document_rag_english_study.cli.interface.DocumentManager')
    @patch('src.document_rag_english_study.cli.interface.ConfigurationManager')
    def test_status_command_detailed(self, mock_config_manager, mock_doc_manager):
        """상세 status 명령어 테스트."""
        # Mock 설정
        mock_manager = Mock()
        mock_config_manager.return_value = mock_manager
        
        mock_setup_status = SetupStatus(
            llm_configured=True,
            documents_configured=True,
            user_configured=True,
            overall_complete=True
        )
        mock_manager.get_setup_status.return_value = mock_setup_status
        
        mock_config = Configuration()
        mock_config.user.native_language = "korean"
        mock_config.document.document_directory = "/test/docs"
        mock_config.llm = Mock()
        mock_config.llm.provider = "openai"
        mock_config.llm.model_name = "gpt-3.5-turbo"
        mock_config.llm.temperature = 0.7
        mock_config.llm.max_tokens = 1000
        mock_config.llm.api_key = "sk-test123456789"
        mock_config.version = "1.0.0"
        mock_manager.get_config.return_value = mock_config
        mock_manager.config_path = "/test/config.yaml"
        
        # DocumentManager Mock
        mock_doc_mgr = Mock()
        mock_doc_manager.return_value = mock_doc_mgr
        mock_summary = Mock()
        mock_summary.total_documents = 5
        mock_summary.total_words = 1000
        mock_summary.file_types = {"pdf": 3, "txt": 2}
        mock_doc_mgr.get_document_summary.return_value = mock_summary
        
        mock_indexing_status = Mock()
        mock_indexing_status.is_indexing = False
        mock_doc_mgr.get_indexing_status.return_value = mock_indexing_status
        
        # 명령어 실행
        result = self.runner.invoke(cli, ['status', '--detailed'])
        
        # 검증
        if result.exit_code != 0:
            print(f"Error output: {result.output}")
        assert result.exit_code == 0
        assert "시스템 상태" in result.output
        assert "학습 수준" in result.output
        assert "피드백 수준" in result.output
        assert "지원 형식" in result.output
        assert "청크 크기" in result.output
        assert "인덱싱된 문서: 5개" in result.output
        assert "시스템 정보" in result.output
        assert "설정 파일" in result.output
    
    @patch('src.document_rag_english_study.cli.interface.DocumentManager')
    @patch('src.document_rag_english_study.cli.interface.ConfigurationManager')
    def test_status_command_json(self, mock_config_manager, mock_doc_manager):
        """JSON 형식 status 명령어 테스트."""
        # Mock 설정
        mock_manager = Mock()
        mock_config_manager.return_value = mock_manager
        
        mock_setup_status = SetupStatus(
            llm_configured=True,
            documents_configured=True,
            user_configured=True,
            overall_complete=True
        )
        mock_manager.get_setup_status.return_value = mock_setup_status
        
        mock_config = Configuration()
        mock_config.user.native_language = "korean"
        mock_config.document.document_directory = "/test/docs"
        mock_config.llm = Mock()
        mock_config.llm.provider = "openai"
        mock_config.llm.model_name = "gpt-3.5-turbo"
        mock_config.llm.temperature = 0.7
        mock_config.llm.max_tokens = 1000
        mock_config.llm.api_key = "sk-test123456789"
        mock_config.version = "1.0.0"
        mock_config.created_at = "2024-01-01T00:00:00"
        mock_config.updated_at = "2024-01-01T00:00:00"
        mock_manager.get_config.return_value = mock_config
        mock_manager.config_path = "/test/config.yaml"
        
        # DocumentManager Mock
        mock_doc_mgr = Mock()
        mock_doc_manager.return_value = mock_doc_mgr
        mock_summary = Mock()
        mock_summary.total_documents = 5
        mock_summary.total_words = 1000
        mock_summary.file_types = {"pdf": 3, "txt": 2}
        # Mock last_indexed attribute to avoid JSON serialization issues
        from datetime import datetime
        mock_summary.last_indexed = datetime(2024, 1, 1, 0, 0, 0)
        mock_doc_mgr.get_document_summary.return_value = mock_summary
        
        # 명령어 실행
        result = self.runner.invoke(cli, ['status', '--json'])
        
        # 검증
        if result.exit_code != 0:
            print(f"Error output: {result.output}")
        assert result.exit_code == 0
        
        # JSON 파싱 테스트
        try:
            json_data = json.loads(result.output)
            assert json_data["overall_complete"] is True
            assert json_data["completion_percentage"] == 100.0
            assert json_data["user_config"]["native_language"] == "korean"
            assert json_data["llm_config"]["provider"] == "openai"
            assert json_data["indexing_info"]["total_documents"] == 5
        except json.JSONDecodeError:
            pytest.fail("출력이 유효한 JSON이 아닙니다")
    
    @patch('src.document_rag_english_study.cli.interface.ConfigurationManager')
    def test_status_command_incomplete_setup(self, mock_config_manager):
        """설정이 완료되지 않은 경우 status 명령어 테스트."""
        # Mock 설정
        mock_manager = Mock()
        mock_config_manager.return_value = mock_manager
        
        mock_setup_status = SetupStatus(
            llm_configured=False,
            documents_configured=True,
            user_configured=True,
            overall_complete=False
        )
        mock_setup_status.get_completion_percentage = Mock(return_value=66.7)
        mock_setup_status.get_missing_steps = Mock(return_value=["LLM configuration"])
        mock_manager.get_setup_status.return_value = mock_setup_status
        
        mock_config = Configuration()
        mock_config.user.native_language = "korean"
        mock_config.document.document_directory = "/test/docs"
        mock_config.llm = None
        mock_manager.get_config.return_value = mock_config
        
        # 명령어 실행
        result = self.runner.invoke(cli, ['status'])
        
        # 검증
        assert result.exit_code == 0
        assert "설정 진행률: 66.7%" in result.output
        assert "미완료 항목" in result.output
        assert "다음 단계" in result.output
        assert "setup" in result.output


class TestHelpCommand:
    """Help 명령어 테스트 클래스."""
    
    def setup_method(self):
        """각 테스트 메서드 실행 전 설정."""
        self.runner = CliRunner()
    
    def test_help_command_basic(self):
        """기본 help 명령어 테스트."""
        result = self.runner.invoke(cli, ['help'])
        
        assert result.exit_code == 0
        assert "Document RAG English Study 상세 도움말" in result.output
        assert "프로그램 개요" in result.output
        assert "빠른 시작" in result.output
        assert "주요 명령어" in result.output
        assert "설정 명령어" in result.output
        assert "학습 명령어" in result.output
        assert "정보 명령어" in result.output
        assert "지원 기능" in result.output
        assert "문제 해결" in result.output
    
    def test_help_command_examples(self):
        """예제 포함 help 명령어 테스트."""
        result = self.runner.invoke(cli, ['help', '--examples'])
        
        assert result.exit_code == 0
        assert "사용 예제 모음" in result.output
        assert "처음 사용하는 경우" in result.output
        assert "다양한 문서 형식 활용" in result.output
        assert "다양한 LLM 제공업체 사용" in result.output
        assert "맞춤형 학습 설정" in result.output
        assert "대화형 학습 활용" in result.output
        assert "상태 확인 및 문제 해결" in result.output
        assert "고급 사용법" in result.output
        assert "실제 학습 시나리오" in result.output
        assert "팁" in result.output
    
    def test_help_command_specific_command(self):
        """특정 명령어 help 테스트."""
        result = self.runner.invoke(cli, ['help', '--command', 'chat'])
        
        assert result.exit_code == 0
        assert "'chat' 명령어 상세 도움말" in result.output
        assert "설명: 대화형 영어 학습 시작" in result.output
        assert "사용법: english-study chat" in result.output
        assert "옵션:" in result.output
        assert "상세 기능:" in result.output
        assert "사용 예제:" in result.output
        assert "대화 중 명령어:" in result.output
    
    def test_help_command_unknown_command(self):
        """알 수 없는 명령어에 대한 help 테스트."""
        result = self.runner.invoke(cli, ['help', '--command', 'unknown'])
        
        assert result.exit_code == 0
        assert "알 수 없는 명령어: unknown" in result.output
        assert "사용 가능한 명령어" in result.output
    
    def test_help_command_all_supported_commands(self):
        """지원되는 모든 명령어에 대한 help 테스트."""
        commands = ['setup', 'set-docs', 'set-llm', 'set-language', 'chat', 'status']
        
        for command in commands:
            result = self.runner.invoke(cli, ['help', '--command', command])
            assert result.exit_code == 0
            assert f"'{command}' 명령어 상세 도움말" in result.output
            assert "설명:" in result.output
            assert "사용법:" in result.output


class TestCLIIntegration:
    """CLI 통합 테스트 클래스."""
    
    def setup_method(self):
        """각 테스트 메서드 실행 전 설정."""
        self.runner = CliRunner()
    
    def test_cli_main_help_display(self):
        """메인 CLI 도움말 표시 테스트."""
        result = self.runner.invoke(cli, [])
        
        assert result.exit_code == 0
        assert "Document RAG English Study" in result.output
        assert "관심사 기반 대화형 영어 학습 프로그램" in result.output
        assert "주요 명령어" in result.output
    
    def test_cli_version_display(self):
        """버전 정보 표시 테스트."""
        result = self.runner.invoke(cli, ['--version'])
        
        assert result.exit_code == 0
        assert "Document RAG English Study v0.1.0" in result.output
    
    @patch('src.document_rag_english_study.cli.interface.ConfigurationManager')
    def test_status_error_handling(self, mock_config_manager):
        """Status 명령어 오류 처리 테스트."""
        # Mock에서 예외 발생
        mock_config_manager.side_effect = Exception("설정 파일 오류")
        
        result = self.runner.invoke(cli, ['status'])
        
        assert result.exit_code == 1
        assert "상태 확인 중 오류 발생" in result.output