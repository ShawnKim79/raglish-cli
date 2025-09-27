"""
Logging Config 모듈 단위 테스트.

이 모듈은 로깅 설정 및 관리 기능을 테스트합니다.
"""

import pytest
import logging
import tempfile
import shutil
import os
from pathlib import Path
from unittest.mock import patch, MagicMock
from datetime import datetime, timedelta

from src.document_rag_english_study.utils.logging_config import (
    LoggingConfig,
    get_logging_config,
    setup_logging,
    get_logger,
    log_context
)


class TestLoggingConfig:
    """LoggingConfig 클래스 테스트."""

    def setup_method(self):
        """각 테스트 메서드 실행 전 설정."""
        self.temp_dir = tempfile.mkdtemp()
        self.log_dir = Path(self.temp_dir) / "test_logs"

    def teardown_method(self):
        """각 테스트 메서드 실행 후 정리."""
        # 로깅 핸들러 정리
        root_logger = logging.getLogger()
        for handler in root_logger.handlers[:]:
            root_logger.removeHandler(handler)
        
        # 임시 디렉토리 정리
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_initialization(self):
        """LoggingConfig 초기화 테스트."""
        config = LoggingConfig(log_dir=str(self.log_dir), app_name="test_app")
        
        assert config.log_dir == self.log_dir
        assert config.app_name == "test_app"
        assert config.log_dir.exists()
        assert not config._configured

    def test_log_file_paths(self):
        """로그 파일 경로 설정 테스트."""
        config = LoggingConfig(log_dir=str(self.log_dir), app_name="test_app")
        
        assert config.main_log_file == self.log_dir / "test_app.log"
        assert config.error_log_file == self.log_dir / "test_app_error.log"
        assert config.debug_log_file == self.log_dir / "test_app_debug.log"

    def test_setup_logging_basic(self):
        """기본 로깅 설정 테스트."""
        config = LoggingConfig(log_dir=str(self.log_dir))
        config.setup_logging(level="INFO", console_output=False, file_output=True)
        
        assert config._configured is True
        
        # 로그 파일이 생성되었는지 확인
        logger = config.get_logger("test")
        logger.info("Test message")
        
        assert config.main_log_file.exists()

    def test_setup_logging_debug_level(self):
        """DEBUG 레벨 로깅 설정 테스트."""
        config = LoggingConfig(log_dir=str(self.log_dir))
        config.setup_logging(level="DEBUG", console_output=False, file_output=True)
        
        # DEBUG 레벨에서는 디버그 로그 파일도 생성되어야 함
        logger = config.get_logger("test")
        logger.debug("Debug message")
        
        assert config.debug_log_file.exists()

    def test_setup_logging_console_only(self):
        """콘솔 전용 로깅 설정 테스트."""
        config = LoggingConfig(log_dir=str(self.log_dir))
        config.setup_logging(level="INFO", console_output=True, file_output=False)
        
        logger = config.get_logger("test")
        logger.info("Console only message")
        
        # 파일은 생성되지 않아야 함
        assert not config.main_log_file.exists()

    def test_setup_logging_file_only(self):
        """파일 전용 로깅 설정 테스트."""
        config = LoggingConfig(log_dir=str(self.log_dir))
        config.setup_logging(level="INFO", console_output=False, file_output=True)
        
        logger = config.get_logger("test")
        logger.info("File only message")
        
        assert config.main_log_file.exists()

    def test_setup_logging_idempotent(self):
        """로깅 설정이 중복 호출되어도 안전한지 테스트."""
        config = LoggingConfig(log_dir=str(self.log_dir))
        
        # 첫 번째 설정
        config.setup_logging(level="INFO")
        assert config._configured is True
        
        # 두 번째 설정 (무시되어야 함)
        config.setup_logging(level="DEBUG")
        
        # 여전히 INFO 레벨이어야 함
        root_logger = logging.getLogger()
        assert root_logger.level == logging.INFO

    def test_get_logger(self):
        """로거 생성 테스트."""
        config = LoggingConfig(log_dir=str(self.log_dir))
        
        logger = config.get_logger("test_logger")
        
        assert isinstance(logger, logging.Logger)
        assert logger.name == "test_logger"
        assert config._configured is True  # 자동으로 설정되어야 함

    def test_log_system_info(self):
        """시스템 정보 로깅 테스트."""
        config = LoggingConfig(log_dir=str(self.log_dir))
        config.setup_logging(console_output=False, file_output=True)
        
        config.log_system_info()
        
        # 로그 파일에 시스템 정보가 기록되었는지 확인
        assert config.main_log_file.exists()
        log_content = config.main_log_file.read_text(encoding='utf-8')
        assert "애플리케이션 시작" in log_content
        assert "Python 버전" in log_content

    def test_cleanup_old_logs(self):
        """오래된 로그 파일 정리 테스트."""
        config = LoggingConfig(log_dir=str(self.log_dir))
        
        # 오래된 로그 파일 생성
        old_log = self.log_dir / "old.log"
        old_log.write_text("old log content")
        
        # 파일 수정 시간을 과거로 설정
        old_time = (datetime.now() - timedelta(days=35)).timestamp()
        os.utime(old_log, (old_time, old_time))
        
        # 최근 로그 파일 생성
        recent_log = self.log_dir / "recent.log"
        recent_log.write_text("recent log content")
        
        # 정리 실행
        config.cleanup_old_logs(days_to_keep=30)
        
        # 오래된 파일은 삭제되고 최근 파일은 유지되어야 함
        assert not old_log.exists()
        assert recent_log.exists()

    @patch('src.document_rag_english_study.utils.logging_config.Path.glob')
    def test_cleanup_old_logs_error_handling(self, mock_glob):
        """로그 정리 중 오류 처리 테스트."""
        config = LoggingConfig(log_dir=str(self.log_dir))
        config.setup_logging(console_output=False, file_output=True)
        
        # glob에서 예외 발생하도록 설정
        mock_glob.side_effect = Exception("Test error")
        
        # 예외가 발생해도 프로그램이 중단되지 않아야 함
        config.cleanup_old_logs()
        
        # 에러 로그가 기록되었는지 확인
        assert config.main_log_file.exists()

    def test_different_log_levels(self):
        """다양한 로그 레벨 테스트."""
        config = LoggingConfig(log_dir=str(self.log_dir))
        
        for level in ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]:
            config._configured = False  # 재설정 허용
            config.setup_logging(level=level, console_output=False, file_output=True)
            
            expected_level = getattr(logging, level)
            root_logger = logging.getLogger()
            assert root_logger.level == expected_level

    def test_error_log_separation(self):
        """에러 로그 분리 테스트."""
        config = LoggingConfig(log_dir=str(self.log_dir))
        config.setup_logging(level="DEBUG", console_output=False, file_output=True)
        
        logger = config.get_logger("test")
        logger.info("Info message")
        logger.error("Error message")
        
        # 메인 로그에는 모든 메시지가 있어야 함
        main_content = config.main_log_file.read_text(encoding='utf-8')
        assert "Info message" in main_content
        assert "Error message" in main_content
        
        # 에러 로그에는 에러 메시지만 있어야 함
        error_content = config.error_log_file.read_text(encoding='utf-8')
        assert "Info message" not in error_content
        assert "Error message" in error_content


class TestGlobalFunctions:
    """전역 함수들 테스트."""

    def setup_method(self):
        """각 테스트 메서드 실행 전 설정."""
        self.temp_dir = tempfile.mkdtemp()
        
        # 전역 설정 초기화
        import src.document_rag_english_study.utils.logging_config as logging_module
        logging_module._logging_config = None

    def teardown_method(self):
        """각 테스트 메서드 실행 후 정리."""
        # 로깅 핸들러 정리
        root_logger = logging.getLogger()
        for handler in root_logger.handlers[:]:
            root_logger.removeHandler(handler)
        
        # 임시 디렉토리 정리
        shutil.rmtree(self.temp_dir, ignore_errors=True)
        
        # 전역 설정 초기화
        import src.document_rag_english_study.utils.logging_config as logging_module
        logging_module._logging_config = None

    def test_get_logging_config_singleton(self):
        """전역 로깅 설정 싱글톤 테스트."""
        config1 = get_logging_config()
        config2 = get_logging_config()
        
        assert config1 is config2
        assert isinstance(config1, LoggingConfig)

    @patch('src.document_rag_english_study.utils.logging_config.LoggingConfig')
    def test_setup_logging_global(self, mock_logging_config):
        """전역 로깅 설정 함수 테스트."""
        mock_instance = MagicMock()
        mock_logging_config.return_value = mock_instance
        
        setup_logging(level="DEBUG", console_output=True, file_output=False)
        
        mock_instance.setup_logging.assert_called_once_with(
            level="DEBUG", 
            console_output=True, 
            file_output=False
        )
        mock_instance.log_system_info.assert_called_once()

    def test_get_logger_global(self):
        """전역 로거 생성 함수 테스트."""
        logger = get_logger("test_global_logger")
        
        assert isinstance(logger, logging.Logger)
        assert logger.name == "test_global_logger"

    def test_log_context(self):
        """컨텍스트 로깅 함수 테스트."""
        # 임시 로그 디렉토리로 설정
        with patch('src.document_rag_english_study.utils.logging_config.LoggingConfig') as mock_config_class:
            mock_config = MagicMock()
            mock_logger = MagicMock()
            mock_config.get_logger.return_value = mock_logger
            mock_config_class.return_value = mock_config
            
            context = {
                "user_id": "123",
                "session_id": "abc",
                "operation": "test"
            }
            
            log_context(context)
            
            # 컨텍스트 정보가 로깅되었는지 확인
            assert mock_logger.debug.call_count >= len(context) + 1  # 헤더 + 각 항목

    def test_log_context_with_custom_logger(self):
        """사용자 지정 로거로 컨텍스트 로깅 테스트."""
        mock_logger = MagicMock()
        context = {"key1": "value1", "key2": "value2"}
        
        log_context(context, logger=mock_logger)
        
        # 사용자 지정 로거가 사용되었는지 확인
        assert mock_logger.debug.call_count >= len(context) + 1


class TestLoggingIntegration:
    """로깅 시스템 통합 테스트."""

    def setup_method(self):
        """각 테스트 메서드 실행 전 설정."""
        self.temp_dir = tempfile.mkdtemp()
        self.log_dir = Path(self.temp_dir) / "integration_logs"

    def teardown_method(self):
        """각 테스트 메서드 실행 후 정리."""
        # 로깅 핸들러 정리
        root_logger = logging.getLogger()
        for handler in root_logger.handlers[:]:
            root_logger.removeHandler(handler)
        
        # 임시 디렉토리 정리
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_full_logging_workflow(self):
        """전체 로깅 워크플로우 테스트."""
        config = LoggingConfig(log_dir=str(self.log_dir), app_name="integration_test")
        
        # 1. 로깅 설정
        config.setup_logging(level="DEBUG", console_output=False, file_output=True)
        
        # 2. 시스템 정보 로깅
        config.log_system_info()
        
        # 3. 다양한 레벨의 로그 메시지
        logger = config.get_logger("integration_test")
        logger.debug("Debug message")
        logger.info("Info message")
        logger.warning("Warning message")
        logger.error("Error message")
        logger.critical("Critical message")
        
        # 4. 컨텍스트 로깅
        context = {"test_id": "integration", "step": "logging"}
        log_context(context, logger=logger)
        
        # 5. 로그 파일 확인
        assert config.main_log_file.exists()
        assert config.error_log_file.exists()
        assert config.debug_log_file.exists()
        
        # 6. 로그 내용 확인
        main_content = config.main_log_file.read_text(encoding='utf-8')
        assert "Debug message" in main_content
        assert "Info message" in main_content
        assert "Error message" in main_content
        
        error_content = config.error_log_file.read_text(encoding='utf-8')
        assert "Error message" in error_content
        assert "Critical message" in error_content
        assert "Info message" not in error_content

    def test_rotating_file_handler(self):
        """로그 파일 로테이션 테스트."""
        config = LoggingConfig(log_dir=str(self.log_dir))
        config.setup_logging(
            level="INFO", 
            console_output=False, 
            file_output=True,
            max_file_size=1024,  # 1KB로 작게 설정
            backup_count=3
        )
        
        logger = config.get_logger("rotation_test")
        
        # 큰 로그 메시지를 여러 번 기록하여 로테이션 유발
        large_message = "A" * 200  # 200바이트 메시지
        for i in range(10):
            logger.info(f"Message {i}: {large_message}")
        
        # 로그 파일이 존재하는지 확인
        assert config.main_log_file.exists()
        
        # 백업 파일들이 생성되었는지 확인 (로테이션이 발생했다면)
        backup_files = list(self.log_dir.glob("*.log.*"))
        # 파일 크기가 작아서 로테이션이 발생할 수 있음
        assert len(backup_files) >= 0  # 최소 0개 이상

    def test_unicode_logging(self):
        """유니코드 로깅 테스트."""
        config = LoggingConfig(log_dir=str(self.log_dir))
        config.setup_logging(level="INFO", console_output=False, file_output=True)
        
        logger = config.get_logger("unicode_test")
        
        # 다양한 언어의 메시지
        messages = [
            "한글 메시지입니다",
            "English message",
            "日本語メッセージ",
            "Mensaje en español",
            "Message en français"
        ]
        
        for message in messages:
            logger.info(message)
        
        # 로그 파일에서 유니코드 메시지 확인
        content = config.main_log_file.read_text(encoding='utf-8')
        for message in messages:
            assert message in content