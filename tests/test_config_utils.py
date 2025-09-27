"""
Configuration Utils 모듈 단위 테스트.

이 모듈은 config.utils 모듈의 유틸리티 함수들을 테스트합니다.
"""

import pytest
import tempfile
import shutil
import os
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

from src.document_rag_english_study.config.utils import (
    validate_document_directory,
    get_supported_file_extensions,
    scan_document_directory,
    validate_llm_config,
    get_default_llm_config,
    create_default_config_file,
    get_environment_variables,
    setup_from_environment
)
from src.document_rag_english_study.utils.exceptions import ConfigurationError


class TestConfigUtils:
    """Config utils 함수들 테스트."""

    def setup_method(self):
        """각 테스트 메서드 실행 전 설정."""
        self.temp_dir = tempfile.mkdtemp()

    def teardown_method(self):
        """각 테스트 메서드 실행 후 정리."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_validate_document_directory_valid(self):
        """유효한 문서 디렉토리 검증 테스트."""
        # 임시 디렉토리 생성
        test_dir = Path(self.temp_dir) / "test_docs"
        test_dir.mkdir()
        
        result = validate_document_directory(str(test_dir))
        assert result is True

    def test_validate_document_directory_invalid(self):
        """잘못된 문서 디렉토리 검증 테스트."""
        # 존재하지 않는 디렉토리
        result = validate_document_directory("/nonexistent/directory")
        assert result is False
        
        # 파일 경로 (디렉토리가 아님)
        test_file = Path(self.temp_dir) / "test_file.txt"
        test_file.write_text("test")
        result = validate_document_directory(str(test_file))
        assert result is False

    def test_get_supported_file_extensions(self):
        """지원되는 파일 확장자 목록 조회 테스트."""
        extensions = get_supported_file_extensions()
        
        assert isinstance(extensions, list)
        assert '.pdf' in extensions
        assert '.docx' in extensions
        assert '.txt' in extensions
        assert '.md' in extensions

    def test_scan_document_directory_valid(self):
        """유효한 문서 디렉토리 스캔 테스트."""
        # 테스트 파일들 생성
        test_dir = Path(self.temp_dir) / "docs"
        test_dir.mkdir()
        
        (test_dir / "test.txt").write_text("test content")
        (test_dir / "test.md").write_text("# Test")
        (test_dir / "test.pdf").write_text("fake pdf")  # 실제 PDF가 아니어도 확장자만 확인
        (test_dir / "other.xyz").write_text("unsupported")
        
        result = scan_document_directory(str(test_dir))
        
        assert result['valid'] is True
        assert result['total_files'] == 4
        assert result['supported_files'] == 3
        assert 'txt' in result['files_by_type']
        assert 'md' in result['files_by_type']
        assert 'pdf' in result['files_by_type']

    def test_scan_document_directory_invalid(self):
        """잘못된 문서 디렉토리 스캔 테스트."""
        result = scan_document_directory("/nonexistent/directory")
        
        assert result['valid'] is False
        assert 'error' in result
        assert result['total_files'] == 0
        assert result['supported_files'] == 0

    def test_validate_llm_config_valid_openai(self):
        """유효한 OpenAI LLM 설정 검증 테스트."""
        from src.document_rag_english_study.models.config import LLMConfig
        
        config = LLMConfig(
            provider="openai",
            api_key="sk-test-key-123456789",
            model_name="gpt-3.5-turbo",
            temperature=0.7,
            max_tokens=1000
        )
        
        result = validate_llm_config(config)
        
        assert result['valid'] is True
        assert len(result['errors']) == 0

    def test_validate_llm_config_invalid_provider(self):
        """잘못된 제공업체 LLM 설정 검증 테스트."""
        from src.document_rag_english_study.models.config import LLMConfig
        
        config = LLMConfig(
            provider="invalid_provider",
            api_key="test-key",
            model_name="test-model",
            temperature=0.7,
            max_tokens=1000
        )
        
        result = validate_llm_config(config)
        
        assert result['valid'] is False
        assert any("지원되지 않는 LLM 제공업체" in error for error in result['errors'])

    def test_validate_llm_config_missing_api_key(self):
        """API 키 누락 LLM 설정 검증 테스트."""
        from src.document_rag_english_study.models.config import LLMConfig
        
        config = LLMConfig(
            provider="openai",
            api_key="",
            model_name="gpt-3.5-turbo",
            temperature=0.7,
            max_tokens=1000
        )
        
        result = validate_llm_config(config)
        
        assert result['valid'] is False
        assert any("API 키가 필요합니다" in error for error in result['errors'])

    def test_validate_llm_config_invalid_temperature(self):
        """잘못된 온도 설정 검증 테스트."""
        from src.document_rag_english_study.models.config import LLMConfig
        
        config = LLMConfig(
            provider="openai",
            api_key="sk-test-key",
            model_name="gpt-3.5-turbo",
            temperature=3.0,  # 유효 범위 초과
            max_tokens=1000
        )
        
        result = validate_llm_config(config)
        
        assert result['valid'] is False
        assert any("온도 설정은 0.0과 2.0 사이여야 합니다" in error for error in result['errors'])

    def test_get_default_llm_config_openai(self):
        """OpenAI 기본 LLM 설정 조회 테스트."""
        config = get_default_llm_config("openai")
        
        assert config['model_name'] == 'gpt-3.5-turbo'
        assert config['temperature'] == 0.7
        assert config['max_tokens'] == 1000
        assert config['host'] == ''

    def test_get_default_llm_config_gemini(self):
        """Gemini 기본 LLM 설정 조회 테스트."""
        config = get_default_llm_config("gemini")
        
        assert config['model_name'] == 'gemini-pro'
        assert config['temperature'] == 0.7
        assert config['max_tokens'] == 1000

    def test_get_default_llm_config_ollama(self):
        """Ollama 기본 LLM 설정 조회 테스트."""
        config = get_default_llm_config("ollama")
        
        assert config['model_name'] == 'llama2'
        assert config['host'] == 'localhost:11434'

    def test_get_default_llm_config_invalid(self):
        """잘못된 제공업체 기본 설정 조회 테스트."""
        config = get_default_llm_config("invalid_provider")
        
        # 기본값으로 OpenAI 설정 반환
        assert config['model_name'] == 'gpt-3.5-turbo'

    @patch('src.document_rag_english_study.config.utils.os.getenv')
    def test_get_environment_variables(self, mock_getenv):
        """환경 변수 조회 테스트."""
        mock_getenv.side_effect = lambda key: {
            'OPENAI_API_KEY': 'test-openai-key',
            'GEMINI_API_KEY': 'test-gemini-key',
            'DOCUMENT_DIR': '/test/docs',
            'NATIVE_LANGUAGE': 'korean',
            'CONFIG_PATH': '/test/config.yaml'
        }.get(key)
        
        env_vars = get_environment_variables()
        
        assert env_vars['OPENAI_API_KEY'] == 'test-openai-key'
        assert env_vars['GEMINI_API_KEY'] == 'test-gemini-key'
        assert env_vars['DOCUMENT_DIR'] == '/test/docs'
        assert env_vars['NATIVE_LANGUAGE'] == 'korean'
        assert env_vars['CONFIG_PATH'] == '/test/config.yaml'

    @patch('src.document_rag_english_study.config.utils.os.getenv')
    def test_setup_from_environment_openai(self, mock_getenv):
        """OpenAI 환경 변수 설정 테스트."""
        mock_getenv.side_effect = lambda key: {
            'OPENAI_API_KEY': 'test-openai-key',
            'DOCUMENT_DIR': '/test/docs',
            'NATIVE_LANGUAGE': 'english'
        }.get(key)
        
        config_info = setup_from_environment()
        
        assert config_info['llm_provider'] == 'openai'
        assert config_info['api_key'] == 'test-openai-key'
        assert config_info['document_directory'] == '/test/docs'
        assert config_info['native_language'] == 'english'

    @patch('src.document_rag_english_study.config.utils.os.getenv')
    def test_setup_from_environment_gemini(self, mock_getenv):
        """Gemini 환경 변수 설정 테스트."""
        mock_getenv.side_effect = lambda key: {
            'GEMINI_API_KEY': 'test-gemini-key',
            'DOCUMENT_DIR': '/test/docs'
        }.get(key)
        
        config_info = setup_from_environment()
        
        assert config_info['llm_provider'] == 'gemini'
        assert config_info['api_key'] == 'test-gemini-key'
        assert config_info['native_language'] == 'korean'  # 기본값

    @patch('src.document_rag_english_study.config.utils.os.getenv')
    def test_setup_from_environment_no_api_key(self, mock_getenv):
        """API 키 없는 환경 변수 설정 테스트."""
        mock_getenv.side_effect = lambda key: {
            'DOCUMENT_DIR': '/test/docs',
            'NATIVE_LANGUAGE': 'korean'
        }.get(key)
        
        config_info = setup_from_environment()
        
        assert config_info['llm_provider'] is None
        assert config_info['api_key'] is None
        assert config_info['document_directory'] == '/test/docs'

    def test_create_default_config_file(self):
        """기본 설정 파일 생성 테스트."""
        config_path = Path(self.temp_dir) / "test_config.yaml"
        
        # 함수 실행
        create_default_config_file(str(config_path))
        
        # 설정 파일이 생성되었는지 확인
        assert config_path.exists()


class TestConfigUtilsIntegration:
    """Config utils 통합 테스트."""

    def setup_method(self):
        """각 테스트 메서드 실행 전 설정."""
        self.temp_dir = tempfile.mkdtemp()

    def teardown_method(self):
        """각 테스트 메서드 실행 후 정리."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_full_document_setup_workflow(self):
        """전체 문서 설정 워크플로우 테스트."""
        # 1. 문서 디렉토리 생성
        docs_dir = Path(self.temp_dir) / "documents"
        docs_dir.mkdir()
        
        # 2. 테스트 문서 파일들 생성
        (docs_dir / "test.txt").write_text("Test content")
        (docs_dir / "guide.md").write_text("# Guide")
        (docs_dir / "manual.pdf").write_text("Fake PDF")
        
        # 3. 디렉토리 검증
        assert validate_document_directory(str(docs_dir)) is True
        
        # 4. 디렉토리 스캔
        scan_result = scan_document_directory(str(docs_dir))
        
        # 5. 결과 검증
        assert scan_result['valid'] is True
        assert scan_result['supported_files'] == 3
        assert 'txt' in scan_result['files_by_type']
        assert 'md' in scan_result['files_by_type']
        assert 'pdf' in scan_result['files_by_type']

    def test_llm_config_validation_workflow(self):
        """LLM 설정 검증 워크플로우 테스트."""
        from src.document_rag_english_study.models.config import LLMConfig
        
        # 1. 기본 설정 조회
        default_config = get_default_llm_config("openai")
        
        # 2. LLM 설정 객체 생성
        llm_config = LLMConfig(
            provider="openai",
            api_key="sk-test-key-123456789",
            model_name=default_config['model_name'],
            temperature=default_config['temperature'],
            max_tokens=default_config['max_tokens']
        )
        
        # 3. 설정 검증
        validation_result = validate_llm_config(llm_config)
        
        # 4. 결과 확인
        assert validation_result['valid'] is True
        assert len(validation_result['errors']) == 0

    @patch('src.document_rag_english_study.config.utils.os.getenv')
    def test_environment_setup_workflow(self, mock_getenv):
        """환경 변수 설정 워크플로우 테스트."""
        # 1. 환경 변수 설정
        mock_getenv.side_effect = lambda key: {
            'OPENAI_API_KEY': 'test-openai-key',
            'DOCUMENT_DIR': self.temp_dir,
            'NATIVE_LANGUAGE': 'korean'
        }.get(key)
        
        # 2. 환경 변수 조회
        env_vars = get_environment_variables()
        
        # 3. 환경 변수에서 설정 생성
        config_info = setup_from_environment()
        
        # 4. 결과 검증
        assert env_vars['OPENAI_API_KEY'] == 'test-openai-key'
        assert config_info['llm_provider'] == 'openai'
        assert config_info['api_key'] == 'test-openai-key'
        assert config_info['document_directory'] == self.temp_dir
        assert config_info['native_language'] == 'korean'

    def test_error_handling_workflow(self):
        """오류 처리 워크플로우 테스트."""
        from src.document_rag_english_study.models.config import LLMConfig
        
        # 1. 잘못된 디렉토리 검증
        assert validate_document_directory("/nonexistent/directory") is False
        
        # 2. 잘못된 디렉토리 스캔
        scan_result = scan_document_directory("/nonexistent/directory")
        assert scan_result['valid'] is False
        assert 'error' in scan_result
        
        # 3. 잘못된 LLM 설정 검증
        invalid_config = LLMConfig(
            provider="invalid_provider",
            api_key="",
            model_name="",
            temperature=5.0,  # 유효 범위 초과
            max_tokens=-100   # 음수
        )
        
        validation_result = validate_llm_config(invalid_config)
        assert validation_result['valid'] is False
        assert len(validation_result['errors']) > 0

    def test_file_extension_and_scanning_integration(self):
        """파일 확장자 및 스캔 통합 테스트."""
        # 1. 지원되는 확장자 조회
        supported_extensions = get_supported_file_extensions()
        
        # 2. 테스트 디렉토리 및 파일 생성
        test_dir = Path(self.temp_dir) / "mixed_files"
        test_dir.mkdir()
        
        # 지원되는 파일들
        for ext in supported_extensions:
            filename = f"test{ext}"
            (test_dir / filename).write_text(f"Content for {filename}")
        
        # 지원되지 않는 파일들
        (test_dir / "test.xyz").write_text("Unsupported file")
        (test_dir / "test.abc").write_text("Another unsupported file")
        
        # 3. 디렉토리 스캔
        scan_result = scan_document_directory(str(test_dir))
        
        # 4. 결과 검증
        assert scan_result['valid'] is True
        assert scan_result['total_files'] == len(supported_extensions) + 2
        assert scan_result['supported_files'] == len(supported_extensions)
        
        # 각 지원되는 확장자별로 파일이 인식되었는지 확인
        for ext in supported_extensions:
            ext_name = ext.lstrip('.')
            assert ext_name in scan_result['files_by_type']
            assert scan_result['files_by_type'][ext_name] == 1