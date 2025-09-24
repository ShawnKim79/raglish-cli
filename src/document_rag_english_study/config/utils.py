"""
설정 관리를 위한 유틸리티 함수들.

이 모듈은 설정 검증, 기본값 설정 등의 유틸리티 기능을 제공합니다.
"""

import os
from pathlib import Path
from typing import List, Dict, Any, Optional

from ..models.config import LLMConfig, DocumentConfig, UserConfig


def validate_document_directory(directory_path: str) -> bool:
    """문서 디렉토리가 유효한지 검증합니다.
    
    Args:
        directory_path: 검증할 디렉토리 경로
        
    Returns:
        bool: 유효한 디렉토리인지 여부
    """
    try:
        path = Path(directory_path)
        return path.exists() and path.is_dir()
    except Exception:
        return False


def get_supported_file_extensions() -> List[str]:
    """지원되는 파일 확장자 목록을 반환합니다.
    
    Returns:
        List[str]: 지원되는 파일 확장자 목록
    """
    return ['.pdf', '.docx', '.txt', '.md']


def scan_document_directory(directory_path: str, 
                          supported_formats: Optional[List[str]] = None) -> Dict[str, Any]:
    """문서 디렉토리를 스캔하여 파일 정보를 반환합니다.
    
    Args:
        directory_path: 스캔할 디렉토리 경로
        supported_formats: 지원되는 파일 형식 목록
        
    Returns:
        Dict[str, Any]: 스캔 결과 정보
    """
    if not supported_formats:
        supported_formats = ['pdf', 'docx', 'txt', 'md']
    
    path = Path(directory_path)
    if not path.exists() or not path.is_dir():
        return {
            'valid': False,
            'error': '디렉토리가 존재하지 않거나 유효하지 않습니다.',
            'total_files': 0,
            'supported_files': 0,
            'files_by_type': {}
        }
    
    total_files = 0
    supported_files = 0
    files_by_type = {}
    
    try:
        for file_path in path.rglob('*'):
            if file_path.is_file():
                total_files += 1
                file_ext = file_path.suffix.lower().lstrip('.')
                
                if file_ext in supported_formats:
                    supported_files += 1
                    files_by_type[file_ext] = files_by_type.get(file_ext, 0) + 1
        
        return {
            'valid': True,
            'total_files': total_files,
            'supported_files': supported_files,
            'files_by_type': files_by_type,
            'directory_path': str(path.absolute())
        }
    
    except Exception as e:
        return {
            'valid': False,
            'error': f'디렉토리 스캔 중 오류 발생: {str(e)}',
            'total_files': 0,
            'supported_files': 0,
            'files_by_type': {}
        }


def validate_llm_config(llm_config: LLMConfig) -> Dict[str, Any]:
    """LLM 설정의 유효성을 검증합니다.
    
    Args:
        llm_config: 검증할 LLM 설정
        
    Returns:
        Dict[str, Any]: 검증 결과
    """
    errors = []
    warnings = []
    
    # 제공업체 검증
    if llm_config.provider not in ['openai', 'gemini', 'ollama']:
        errors.append(f"지원되지 않는 LLM 제공업체: {llm_config.provider}")
    
    # API 키 검증
    if llm_config.provider in ['openai', 'gemini']:
        if not llm_config.api_key:
            errors.append(f"{llm_config.provider}에는 API 키가 필요합니다.")
        elif len(llm_config.api_key) < 10:
            warnings.append("API 키가 너무 짧습니다. 올바른 키인지 확인해주세요.")
    
    # 모델명 검증
    if not llm_config.model_name:
        warnings.append("모델명이 설정되지 않았습니다. 기본값이 사용됩니다.")
    
    # 온도 설정 검증
    if not 0.0 <= llm_config.temperature <= 2.0:
        errors.append("온도 설정은 0.0과 2.0 사이여야 합니다.")
    
    # 최대 토큰 검증
    if llm_config.max_tokens <= 0:
        errors.append("최대 토큰 수는 양수여야 합니다.")
    elif llm_config.max_tokens > 4000:
        warnings.append("최대 토큰 수가 매우 큽니다. 비용이 많이 발생할 수 있습니다.")
    
    return {
        'valid': len(errors) == 0,
        'errors': errors,
        'warnings': warnings,
        'is_configured': llm_config.is_configured()
    }


def get_default_llm_config(provider: str) -> Dict[str, Any]:
    """제공업체별 기본 LLM 설정을 반환합니다.
    
    Args:
        provider: LLM 제공업체
        
    Returns:
        Dict[str, Any]: 기본 설정 딕셔너리
    """
    defaults = {
        'openai': {
            'model_name': 'gpt-3.5-turbo',
            'temperature': 0.7,
            'max_tokens': 1000,
            'host': ''
        },
        'gemini': {
            'model_name': 'gemini-pro',
            'temperature': 0.7,
            'max_tokens': 1000,
            'host': ''
        },
        'ollama': {
            'model_name': 'llama2',
            'temperature': 0.7,
            'max_tokens': 1000,
            'host': 'localhost:11434'
        }
    }
    
    return defaults.get(provider, defaults['openai'])


def create_default_config_file(config_path: str) -> None:
    """기본 설정 파일을 생성합니다.
    
    Args:
        config_path: 생성할 설정 파일 경로
    """
    from .manager import ConfigurationManager
    
    manager = ConfigurationManager(config_path)
    # 기본 설정으로 저장
    manager.save_config()


def get_environment_variables() -> Dict[str, Optional[str]]:
    """설정 관련 환경 변수들을 반환합니다.
    
    Returns:
        Dict[str, Optional[str]]: 환경 변수 딕셔너리
    """
    return {
        'OPENAI_API_KEY': os.getenv('OPENAI_API_KEY'),
        'GEMINI_API_KEY': os.getenv('GEMINI_API_KEY'),
        'DOCUMENT_DIR': os.getenv('DOCUMENT_DIR'),
        'NATIVE_LANGUAGE': os.getenv('NATIVE_LANGUAGE'),
        'CONFIG_PATH': os.getenv('CONFIG_PATH')
    }


def setup_from_environment() -> Dict[str, Any]:
    """환경 변수에서 설정을 읽어와 설정 정보를 반환합니다.
    
    Returns:
        Dict[str, Any]: 환경 변수에서 읽은 설정 정보
    """
    env_vars = get_environment_variables()
    
    config_info = {
        'llm_provider': None,
        'api_key': None,
        'document_directory': env_vars.get('DOCUMENT_DIR'),
        'native_language': env_vars.get('NATIVE_LANGUAGE', 'korean'),
        'config_path': env_vars.get('CONFIG_PATH')
    }
    
    # API 키 기반으로 제공업체 결정
    if env_vars.get('OPENAI_API_KEY'):
        config_info['llm_provider'] = 'openai'
        config_info['api_key'] = env_vars['OPENAI_API_KEY']
    elif env_vars.get('GEMINI_API_KEY'):
        config_info['llm_provider'] = 'gemini'
        config_info['api_key'] = env_vars['GEMINI_API_KEY']
    
    return config_info