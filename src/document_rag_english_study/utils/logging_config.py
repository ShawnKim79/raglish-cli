"""
로깅 설정 및 관리 모듈

이 모듈은 애플리케이션 전체의 로깅 설정을 관리하고,
다양한 로그 레벨과 포맷을 제공합니다.
"""

import logging
import logging.handlers
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any


class LoggingConfig:
    """로깅 설정을 관리하는 클래스"""
    
    def __init__(self, log_dir: str = "logs", app_name: str = "document_rag_english_study"):
        """
        LoggingConfig 초기화
        
        Args:
            log_dir: 로그 파일이 저장될 디렉토리
            app_name: 애플리케이션 이름
        """
        self.log_dir = Path(log_dir)
        self.app_name = app_name
        self.log_dir.mkdir(exist_ok=True)
        
        # 로그 파일 경로들
        self.main_log_file = self.log_dir / f"{app_name}.log"
        self.error_log_file = self.log_dir / f"{app_name}_error.log"
        self.debug_log_file = self.log_dir / f"{app_name}_debug.log"
        
        self._configured = False
    
    def setup_logging(self, 
                     level: str = "INFO",
                     console_output: bool = True,
                     file_output: bool = True,
                     max_file_size: int = 10 * 1024 * 1024,  # 10MB
                     backup_count: int = 5) -> None:
        """
        로깅 시스템을 설정합니다
        
        Args:
            level: 로그 레벨 (DEBUG, INFO, WARNING, ERROR, CRITICAL)
            console_output: 콘솔 출력 여부
            file_output: 파일 출력 여부
            max_file_size: 로그 파일 최대 크기 (바이트)
            backup_count: 백업 파일 개수
        """
        if self._configured:
            return
        
        # 기존 핸들러 제거
        root_logger = logging.getLogger()
        for handler in root_logger.handlers[:]:
            root_logger.removeHandler(handler)
        
        # 로그 레벨 설정
        log_level = getattr(logging, level.upper(), logging.INFO)
        root_logger.setLevel(log_level)
        
        # 포맷터 설정
        detailed_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        
        simple_formatter = logging.Formatter(
            '%(asctime)s - %(levelname)s - %(message)s',
            datefmt='%H:%M:%S'
        )
        
        # 콘솔 핸들러 설정
        if console_output:
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setLevel(log_level)
            console_handler.setFormatter(simple_formatter)
            
            # ERROR 이상은 stderr로 출력
            error_console_handler = logging.StreamHandler(sys.stderr)
            error_console_handler.setLevel(logging.ERROR)
            error_console_handler.setFormatter(simple_formatter)
            
            root_logger.addHandler(console_handler)
            root_logger.addHandler(error_console_handler)
        
        # 파일 핸들러 설정
        if file_output:
            # 메인 로그 파일 (모든 레벨)
            main_handler = logging.handlers.RotatingFileHandler(
                self.main_log_file,
                maxBytes=max_file_size,
                backupCount=backup_count,
                encoding='utf-8'
            )
            main_handler.setLevel(log_level)
            main_handler.setFormatter(detailed_formatter)
            root_logger.addHandler(main_handler)
            
            # 에러 로그 파일 (ERROR 이상만)
            error_handler = logging.handlers.RotatingFileHandler(
                self.error_log_file,
                maxBytes=max_file_size,
                backupCount=backup_count,
                encoding='utf-8'
            )
            error_handler.setLevel(logging.ERROR)
            error_handler.setFormatter(detailed_formatter)
            root_logger.addHandler(error_handler)
            
            # 디버그 로그 파일 (DEBUG 레벨일 때만)
            if log_level <= logging.DEBUG:
                debug_handler = logging.handlers.RotatingFileHandler(
                    self.debug_log_file,
                    maxBytes=max_file_size,
                    backupCount=backup_count,
                    encoding='utf-8'
                )
                debug_handler.setLevel(logging.DEBUG)
                debug_handler.setFormatter(detailed_formatter)
                root_logger.addHandler(debug_handler)
        
        self._configured = True
        
        # 설정 완료 로그
        logger = logging.getLogger(__name__)
        logger.info(f"로깅 시스템이 설정되었습니다. 레벨: {level}, 콘솔: {console_output}, 파일: {file_output}")
    
    def get_logger(self, name: str) -> logging.Logger:
        """
        지정된 이름의 로거를 반환합니다
        
        Args:
            name: 로거 이름
            
        Returns:
            logging.Logger: 설정된 로거
        """
        if not self._configured:
            self.setup_logging()
        
        return logging.getLogger(name)
    
    def log_system_info(self) -> None:
        """시스템 정보를 로그에 기록합니다"""
        logger = self.get_logger(__name__)
        
        logger.info("=" * 50)
        logger.info(f"애플리케이션 시작: {self.app_name}")
        logger.info(f"시작 시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info(f"Python 버전: {sys.version}")
        logger.info(f"작업 디렉토리: {os.getcwd()}")
        logger.info(f"로그 디렉토리: {self.log_dir.absolute()}")
        logger.info("=" * 50)
    
    def cleanup_old_logs(self, days_to_keep: int = 30) -> None:
        """
        오래된 로그 파일들을 정리합니다
        
        Args:
            days_to_keep: 보관할 일수
        """
        logger = self.get_logger(__name__)
        
        try:
            cutoff_time = datetime.now().timestamp() - (days_to_keep * 24 * 60 * 60)
            
            for log_file in self.log_dir.glob("*.log*"):
                if log_file.stat().st_mtime < cutoff_time:
                    log_file.unlink()
                    logger.info(f"오래된 로그 파일 삭제: {log_file}")
                    
        except Exception as e:
            logger.error(f"로그 파일 정리 중 오류 발생: {e}")


# 전역 로깅 설정 인스턴스
_logging_config: Optional[LoggingConfig] = None


def get_logging_config() -> LoggingConfig:
    """전역 로깅 설정 인스턴스를 반환합니다"""
    global _logging_config
    if _logging_config is None:
        _logging_config = LoggingConfig()
    return _logging_config


def setup_logging(level: str = "INFO", 
                 console_output: bool = True,
                 file_output: bool = True) -> None:
    """
    전역 로깅을 설정합니다
    
    Args:
        level: 로그 레벨
        console_output: 콘솔 출력 여부
        file_output: 파일 출력 여부
    """
    config = get_logging_config()
    config.setup_logging(level=level, console_output=console_output, file_output=file_output)
    config.log_system_info()


def get_logger(name: str) -> logging.Logger:
    """
    지정된 이름의 로거를 반환합니다
    
    Args:
        name: 로거 이름
        
    Returns:
        logging.Logger: 설정된 로거
    """
    config = get_logging_config()
    return config.get_logger(name)


def log_context(context: Dict[str, Any], logger: Optional[logging.Logger] = None) -> None:
    """
    컨텍스트 정보를 로그에 기록합니다
    
    Args:
        context: 기록할 컨텍스트 정보
        logger: 사용할 로거 (None이면 기본 로거 사용)
    """
    if logger is None:
        logger = get_logger(__name__)
    
    logger.debug("컨텍스트 정보:")
    for key, value in context.items():
        logger.debug(f"  {key}: {value}")