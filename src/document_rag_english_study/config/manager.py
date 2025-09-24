"""
설정 관리 시스템 구현.

이 모듈은 YAML 기반 설정 파일의 로드/저장 기능과 
시스템 설정 관리를 담당하는 ConfigurationManager를 포함합니다.
"""

import os
import yaml
from pathlib import Path
from typing import Optional, Dict, Any
from datetime import datetime

from ..models.config import Configuration, LLMConfig, DocumentConfig, UserConfig, SetupStatus


class ConfigurationManager:
    """시스템 설정을 관리하는 클래스.
    
    YAML 파일을 통해 설정을 저장하고 로드하며, 설정 완료 상태를 확인합니다.
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """ConfigurationManager를 초기화합니다.
        
        Args:
            config_path: 설정 파일 경로. None이면 기본 경로 사용
        """
        if config_path:
            self.config_path = Path(config_path)
        else:
            # 기본 설정 파일 경로: 프로젝트 루트의 config.yaml
            self.config_path = Path.cwd() / "config.yaml"
        
        self._config: Optional[Configuration] = None
        self._ensure_config_directory()
    
    def _ensure_config_directory(self) -> None:
        """설정 파일 디렉토리가 존재하는지 확인하고 없으면 생성합니다."""
        self.config_path.parent.mkdir(parents=True, exist_ok=True)
    
    def load_config(self) -> Configuration:
        """설정 파일에서 설정을 로드합니다.
        
        Returns:
            Configuration: 로드된 설정 객체
            
        Raises:
            FileNotFoundError: 설정 파일이 존재하지 않는 경우
            yaml.YAMLError: YAML 파싱 오류가 발생한 경우
        """
        if not self.config_path.exists():
            # 설정 파일이 없으면 기본 설정 생성
            self._config = Configuration()
            return self._config
        
        try:
            with open(self.config_path, 'r', encoding='utf-8') as file:
                data = yaml.safe_load(file) or {}
                self._config = Configuration.from_dict(data)
                return self._config
        except yaml.YAMLError as e:
            raise yaml.YAMLError(f"설정 파일 파싱 오류: {e}")
        except Exception as e:
            raise Exception(f"설정 파일 로드 중 오류 발생: {e}")
    
    def save_config(self, config: Optional[Configuration] = None) -> None:
        """설정을 파일에 저장합니다.
        
        Args:
            config: 저장할 설정 객체. None이면 현재 로드된 설정 사용
            
        Raises:
            ValueError: 저장할 설정이 없는 경우
            yaml.YAMLError: YAML 저장 오류가 발생한 경우
        """
        if config:
            self._config = config
        
        if not self._config:
            raise ValueError("저장할 설정이 없습니다. 먼저 설정을 로드하거나 제공해주세요.")
        
        # 업데이트 시간 설정
        self._config.updated_at = datetime.now().isoformat()
        if not self._config.created_at:
            self._config.created_at = self._config.updated_at
        
        try:
            with open(self.config_path, 'w', encoding='utf-8') as file:
                yaml.dump(
                    self._config.to_dict(), 
                    file, 
                    default_flow_style=False, 
                    allow_unicode=True,
                    indent=2
                )
        except yaml.YAMLError as e:
            raise yaml.YAMLError(f"설정 파일 저장 오류: {e}")
        except Exception as e:
            raise Exception(f"설정 파일 저장 중 오류 발생: {e}")
    
    def get_config(self) -> Configuration:
        """현재 설정을 반환합니다. 로드되지 않았으면 로드합니다.
        
        Returns:
            Configuration: 현재 설정 객체
        """
        if not self._config:
            return self.load_config()
        return self._config
    
    def update_llm_config(self, llm_config: LLMConfig) -> None:
        """LLM 설정을 업데이트합니다.
        
        Args:
            llm_config: 새로운 LLM 설정
        """
        config = self.get_config()
        config.llm = llm_config
        self.save_config(config)
    
    def update_document_config(self, document_config: DocumentConfig) -> None:
        """문서 설정을 업데이트합니다.
        
        Args:
            document_config: 새로운 문서 설정
        """
        config = self.get_config()
        config.document = document_config
        self.save_config(config)
    
    def update_user_config(self, user_config: UserConfig) -> None:
        """사용자 설정을 업데이트합니다.
        
        Args:
            user_config: 새로운 사용자 설정
        """
        config = self.get_config()
        config.user = user_config
        self.save_config(config)
    
    def get_llm_config(self) -> Optional[LLMConfig]:
        """LLM 설정을 반환합니다.
        
        Returns:
            Optional[LLMConfig]: LLM 설정 또는 None
        """
        return self.get_config().llm
    
    def get_document_config(self) -> DocumentConfig:
        """문서 설정을 반환합니다.
        
        Returns:
            DocumentConfig: 문서 설정
        """
        return self.get_config().document
    
    def get_user_config(self) -> UserConfig:
        """사용자 설정을 반환합니다.
        
        Returns:
            UserConfig: 사용자 설정
        """
        return self.get_config().user
    
    def is_setup_complete(self) -> bool:
        """초기 설정이 완료되었는지 확인합니다.
        
        Returns:
            bool: 설정 완료 여부
        """
        return self.get_config().is_setup_complete()
    
    def get_setup_status(self) -> SetupStatus:
        """상세한 설정 상태를 반환합니다.
        
        Returns:
            SetupStatus: 설정 상태 정보
        """
        return self.get_config().get_setup_status()
    
    def get_native_language(self) -> str:
        """사용자의 모국어를 반환합니다.
        
        Returns:
            str: 모국어 코드
        """
        return self.get_user_config().native_language
    
    def get_document_directory(self) -> Optional[str]:
        """문서 디렉토리 경로를 반환합니다.
        
        Returns:
            Optional[str]: 문서 디렉토리 경로 또는 None
        """
        return self.get_document_config().document_directory
    
    def set_document_directory(self, directory_path: str) -> None:
        """문서 디렉토리를 설정합니다.
        
        Args:
            directory_path: 문서 디렉토리 경로
            
        Raises:
            ValueError: 디렉토리가 존재하지 않는 경우
        """
        path = Path(directory_path)
        if not path.exists():
            raise ValueError(f"디렉토리가 존재하지 않습니다: {directory_path}")
        if not path.is_dir():
            raise ValueError(f"경로가 디렉토리가 아닙니다: {directory_path}")
        
        doc_config = self.get_document_config()
        doc_config.document_directory = str(path.absolute())
        self.update_document_config(doc_config)
    
    def set_llm_provider(self, provider: str, api_key: Optional[str] = None, 
                        model_name: Optional[str] = None, **kwargs) -> None:
        """LLM 제공업체를 설정합니다.
        
        Args:
            provider: LLM 제공업체 (openai, gemini, ollama)
            api_key: API 키 (필요한 경우)
            model_name: 모델명 (선택사항)
            **kwargs: 추가 설정 매개변수
        """
        llm_config = LLMConfig(
            provider=provider,
            api_key=api_key,
            model_name=model_name or "",
            **kwargs
        )
        self.update_llm_config(llm_config)
    
    def set_native_language(self, language: str) -> None:
        """사용자의 모국어를 설정합니다.
        
        Args:
            language: 모국어 코드
        """
        user_config = self.get_user_config()
        user_config.native_language = language
        self.update_user_config(user_config)
    
    def reset_config(self) -> None:
        """설정을 초기화합니다."""
        self._config = Configuration()
        self.save_config()
    
    def backup_config(self, backup_path: Optional[str] = None) -> str:
        """현재 설정을 백업합니다.
        
        Args:
            backup_path: 백업 파일 경로. None이면 자동 생성
            
        Returns:
            str: 백업 파일 경로
        """
        if not backup_path:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_path = f"config_backup_{timestamp}.yaml"
        
        backup_path = Path(backup_path)
        
        # 현재 설정을 백업 파일에 저장
        config = self.get_config()
        with open(backup_path, 'w', encoding='utf-8') as file:
            yaml.dump(
                config.to_dict(), 
                file, 
                default_flow_style=False, 
                allow_unicode=True,
                indent=2
            )
        
        return str(backup_path)
    
    def restore_config(self, backup_path: str) -> None:
        """백업에서 설정을 복원합니다.
        
        Args:
            backup_path: 백업 파일 경로
            
        Raises:
            FileNotFoundError: 백업 파일이 존재하지 않는 경우
        """
        backup_path = Path(backup_path)
        if not backup_path.exists():
            raise FileNotFoundError(f"백업 파일이 존재하지 않습니다: {backup_path}")
        
        with open(backup_path, 'r', encoding='utf-8') as file:
            data = yaml.safe_load(file)
            self._config = Configuration.from_dict(data)
            self.save_config()
    
    def get_config_summary(self) -> Dict[str, Any]:
        """설정 요약 정보를 반환합니다.
        
        Returns:
            Dict[str, Any]: 설정 요약 정보
        """
        config = self.get_config()
        setup_status = self.get_setup_status()
        
        return {
            "설정_완료": setup_status.overall_complete,
            "완료_비율": f"{setup_status.get_completion_percentage():.1f}%",
            "LLM_설정": setup_status.llm_configured,
            "문서_디렉토리": setup_status.documents_configured,
            "사용자_설정": setup_status.user_configured,
            "모국어": config.user.native_language,
            "문서_경로": config.document.document_directory,
            "LLM_제공업체": config.llm.provider if config.llm else None,
            "설정_버전": config.version,
            "마지막_업데이트": config.updated_at
        }