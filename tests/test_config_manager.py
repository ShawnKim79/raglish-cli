"""
Configuration Manager 모듈 단위 테스트.

이 모듈은 ConfigurationManager 클래스의 모든 기능을 테스트합니다.
"""

import pytest
import tempfile
import shutil
import os
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

from src.document_rag_english_study.config.manager import ConfigurationManager
from src.document_rag_english_study.models.config import (
    Configuration, LLMConfig, SetupStatus
)
from src.document_rag_english_study.utils.exceptions import ConfigurationError


class TestConfigurationManager:
    """ConfigurationManager 클래스 테스트."""

    def setup_method(self):
        """각 테스트 메서드 실행 전 설정."""
        self.temp_dir = tempfile.mkdtemp()
        self.config_path = os.path.join(self.temp_dir, "test_config.yaml")
        self.config_manager = ConfigurationManager(config_path=self.config_path)

    def teardown_method(self):
        """각 테스트 메서드 실행 후 정리."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_init_default(self):
        """기본 초기화 테스트."""
        manager = ConfigurationManager()
        assert manager.config_dir is not None
        assert manager.config_file is not None

    def test_init_custom_dir(self):
        """사용자 지정 디렉토리 초기화 테스트."""
        assert self.config_manager.config_dir == Path(self.temp_dir)
        assert self.config_manager.config_file == Path(self.temp_dir) / "config.yaml"

    def test_load_config_new_file(self):
        """새 설정 파일 로드 테스트."""
        config = self.config_manager.load_config()
        
        assert isinstance(config, Configuration)
        assert config.user_language == "korean"
        assert config.document_directory is None
        assert config.llm_config is None

    def test_save_and_load_config(self):
        """설정 저장 및 로드 테스트."""
        # 설정 생성
        llm_config = LLMConfig(
            provider="openai",
            api_key="test-key",
            model_name="gpt-3.5-turbo"
        )
        config = Configuration(
            user_language="english",
            document_directory="/test/docs",
            llm_config=llm_config
        )
        
        # 저장
        self.config_manager.save_config(config)
        
        # 로드 및 검증
        loaded_config = self.config_manager.load_config()
        assert loaded_config.user_language == "english"
        assert loaded_config.document_directory == "/test/docs"
        assert loaded_config.llm_config.provider == "openai"
        assert loaded_config.llm_config.api_key == "test-key"

    def test_get_user_language_default(self):
        """기본 사용자 언어 조회 테스트."""
        language = self.config_manager.get_user_language()
        assert language == "korean"

    def test_set_user_language(self):
        """사용자 언어 설정 테스트."""
        self.config_manager.set_user_language("english")
        
        language = self.config_manager.get_user_language()
        assert language == "english"

    def test_set_user_language_invalid(self):
        """잘못된 사용자 언어 설정 테스트."""
        with pytest.raises(ConfigurationError, match="Unsupported language"):
            self.config_manager.set_user_language("invalid_language")

    def test_get_document_directory_none(self):
        """문서 디렉토리 조회 (없음) 테스트."""
        directory = self.config_manager.get_document_directory()
        assert directory is None

    def test_set_document_directory(self):
        """문서 디렉토리 설정 테스트."""
        test_dir = self.temp_dir
        self.config_manager.set_document_directory(test_dir)
        
        directory = self.config_manager.get_document_directory()
        assert directory == test_dir

    def test_set_document_directory_invalid(self):
        """존재하지 않는 문서 디렉토리 설정 테스트."""
        with pytest.raises(ConfigurationError, match="Directory does not exist"):
            self.config_manager.set_document_directory("/nonexistent/path")

    def test_get_llm_provider_none(self):
        """LLM 제공업체 조회 (없음) 테스트."""
        provider = self.config_manager.get_llm_provider()
        assert provider is None

    def test_set_llm_provider_openai(self):
        """OpenAI LLM 제공업체 설정 테스트."""
        self.config_manager.set_llm_provider("openai", api_key="test-key")
        
        provider = self.config_manager.get_llm_provider()
        assert provider == "openai"
        
        config = self.config_manager.load_config()
        assert config.llm_config.provider == "openai"
        assert config.llm_config.api_key == "test-key"

    def test_set_llm_provider_gemini(self):
        """Gemini LLM 제공업체 설정 테스트."""
        self.config_manager.set_llm_provider("gemini", api_key="test-key")
        
        provider = self.config_manager.get_llm_provider()
        assert provider == "gemini"

    def test_set_llm_provider_ollama(self):
        """Ollama LLM 제공업체 설정 테스트."""
        self.config_manager.set_llm_provider("ollama", host="localhost:11434")
        
        provider = self.config_manager.get_llm_provider()
        assert provider == "ollama"
        
        config = self.config_manager.load_config()
        assert config.llm_config.host == "localhost:11434"

    def test_set_llm_provider_invalid(self):
        """잘못된 LLM 제공업체 설정 테스트."""
        with pytest.raises(ConfigurationError, match="Unsupported LLM provider"):
            self.config_manager.set_llm_provider("invalid_provider")

    def test_set_llm_provider_missing_api_key(self):
        """API 키 누락 테스트."""
        with pytest.raises(ConfigurationError, match="API key is required"):
            self.config_manager.set_llm_provider("openai")

    def test_is_setup_complete_false(self):
        """설정 완료 확인 (미완료) 테스트."""
        assert not self.config_manager.is_setup_complete()

    def test_is_setup_complete_true(self):
        """설정 완료 확인 (완료) 테스트."""
        # 필요한 설정 모두 완료
        self.config_manager.set_user_language("korean")
        self.config_manager.set_document_directory(self.temp_dir)
        self.config_manager.set_llm_provider("openai", api_key="test-key")
        
        assert self.config_manager.is_setup_complete()

    def test_get_setup_status_incomplete(self):
        """설정 상태 조회 (미완료) 테스트."""
        status = self.config_manager.get_setup_status()
        
        assert isinstance(status, SetupStatus)
        assert not status.is_complete
        assert not status.has_document_directory
        assert not status.has_llm_config
        assert status.has_user_language  # 기본값 있음

    def test_get_setup_status_complete(self):
        """설정 상태 조회 (완료) 테스트."""
        # 필요한 설정 모두 완료
        self.config_manager.set_user_language("korean")
        self.config_manager.set_document_directory(self.temp_dir)
        self.config_manager.set_llm_provider("openai", api_key="test-key")
        
        status = self.config_manager.get_setup_status()
        
        assert status.is_complete
        assert status.has_document_directory
        assert status.has_llm_config
        assert status.has_user_language

    def test_validate_config_valid(self):
        """유효한 설정 검증 테스트."""
        llm_config = LLMConfig(
            provider="openai",
            api_key="test-key",
            model_name="gpt-3.5-turbo"
        )
        config = Configuration(
            user_language="korean",
            document_directory=self.temp_dir,
            llm_config=llm_config
        )
        
        # 예외가 발생하지 않아야 함
        self.config_manager._validate_config(config)

    def test_validate_config_invalid_language(self):
        """잘못된 언어 설정 검증 테스트."""
        config = Configuration(user_language="invalid")
        
        with pytest.raises(ConfigurationError, match="Invalid user language"):
            self.config_manager._validate_config(config)

    def test_validate_config_invalid_directory(self):
        """잘못된 디렉토리 설정 검증 테스트."""
        config = Configuration(
            user_language="korean",
            document_directory="/nonexistent/path"
        )
        
        with pytest.raises(ConfigurationError, match="Document directory does not exist"):
            self.config_manager._validate_config(config)

    def test_get_supported_languages(self):
        """지원 언어 목록 조회 테스트."""
        languages = self.config_manager.get_supported_languages()
        
        assert isinstance(languages, list)
        assert "korean" in languages
        assert "english" in languages

    def test_get_supported_llm_providers(self):
        """지원 LLM 제공업체 목록 조회 테스트."""
        providers = self.config_manager.get_supported_llm_providers()
        
        assert isinstance(providers, list)
        assert "openai" in providers
        assert "gemini" in providers
        assert "ollama" in providers

    @patch('src.document_rag_english_study.config.manager.yaml.safe_load')
    def test_load_config_yaml_error(self, mock_yaml_load):
        """YAML 로드 오류 테스트."""
        # 설정 파일 생성
        config_file = Path(self.temp_dir) / "config.yaml"
        config_file.write_text("invalid: yaml: content:")
        
        mock_yaml_load.side_effect = Exception("YAML parse error")
        
        with pytest.raises(ConfigurationError, match="Failed to load configuration"):
            self.config_manager.load_config()

    @patch('builtins.open', side_effect=PermissionError("Permission denied"))
    def test_save_config_permission_error(self, mock_open):
        """설정 저장 권한 오류 테스트."""
        config = Configuration()
        
        with pytest.raises(ConfigurationError, match="Failed to save configuration"):
            self.config_manager.save_config(config)

    def test_reset_config(self):
        """설정 초기화 테스트."""
        # 설정 완료
        self.config_manager.set_user_language("english")
        self.config_manager.set_document_directory(self.temp_dir)
        self.config_manager.set_llm_provider("openai", api_key="test-key")
        
        # 초기화
        self.config_manager.reset_config()
        
        # 기본값으로 복원 확인
        config = self.config_manager.load_config()
        assert config.user_language == "korean"
        assert config.document_directory is None
        assert config.llm_config is None

    def test_backup_and_restore_config(self):
        """설정 백업 및 복원 테스트."""
        # 설정 완료
        self.config_manager.set_user_language("english")
        self.config_manager.set_document_directory(self.temp_dir)
        self.config_manager.set_llm_provider("openai", api_key="test-key")
        
        # 백업
        backup_path = self.config_manager.backup_config()
        assert backup_path.exists()
        
        # 설정 변경
        self.config_manager.set_user_language("korean")
        
        # 복원
        self.config_manager.restore_config(backup_path)
        
        # 복원 확인
        config = self.config_manager.load_config()
        assert config.user_language == "english"


class TestConfigurationManagerIntegration:
    """ConfigurationManager 통합 테스트."""

    def setup_method(self):
        """각 테스트 메서드 실행 전 설정."""
        self.temp_dir = tempfile.mkdtemp()
        self.config_path = os.path.join(self.temp_dir, "test_config.yaml")
        self.config_manager = ConfigurationManager(config_path=self.config_path)

    def teardown_method(self):
        """각 테스트 메서드 실행 후 정리."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_full_setup_workflow(self):
        """전체 설정 워크플로우 테스트."""
        # 초기 상태 확인
        assert not self.config_manager.is_setup_complete()
        
        # 단계별 설정
        self.config_manager.set_user_language("korean")
        self.config_manager.set_document_directory(self.temp_dir)
        self.config_manager.set_llm_provider("openai", api_key="test-key")
        
        # 완료 상태 확인
        assert self.config_manager.is_setup_complete()
        
        # 설정 재로드 후 확인
        new_config_path = os.path.join(self.temp_dir, "test_config.yaml")
        new_manager = ConfigurationManager(config_path=new_config_path)
        config = new_manager.load_config()
        
        assert config.user_language == "korean"
        assert config.document_directory == self.temp_dir
        assert config.llm_config.provider == "openai"

    def test_concurrent_config_access(self):
        """동시 설정 접근 테스트."""
        import threading
        import time
        
        results = []
        errors = []
        
        def worker(worker_id):
            try:
                config_path = os.path.join(self.temp_dir, f"test_config_{worker_id}.yaml")
                manager = ConfigurationManager(config_path=config_path)
                manager.set_user_language("korean")
                time.sleep(0.1)  # 동시성 테스트를 위한 지연
                language = manager.get_user_language()
                results.append((worker_id, language))
            except Exception as e:
                errors.append((worker_id, str(e)))
        
        # 여러 스레드에서 동시 접근
        threads = []
        for i in range(5):
            thread = threading.Thread(target=worker, args=(i,))
            threads.append(thread)
            thread.start()
        
        for thread in threads:
            thread.join()
        
        # 오류 없이 모든 작업 완료 확인
        assert len(errors) == 0
        assert len(results) == 5
        
        # 모든 결과가 일관성 있는지 확인
        for worker_id, language in results:
            assert language == "korean"