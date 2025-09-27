#!/usr/bin/env python3
"""
Main CLI entry point for Document RAG English Study application.
"""

import sys
import os
from pathlib import Path

# Add src to path for development
src_path = Path(__file__).parent.parent.parent
sys.path.insert(0, str(src_path))

from document_rag_english_study.cli import cli
from document_rag_english_study.utils import (
    setup_logging, get_logger, handle_error, get_error_handler,
    DocumentRAGError
)


def main() -> None:
    """Main entry point for the CLI application."""
    # 로깅 시스템 초기화
    log_level = os.getenv("LOG_LEVEL", "INFO")
    setup_logging(level=log_level, console_output=True, file_output=True)
    
    logger = get_logger(__name__)
    error_handler = get_error_handler()
    
    try:
        logger.info("Document RAG English Study 애플리케이션 시작")
        
        # Run the CLI directly using Click
        cli()
        
        logger.info("애플리케이션 정상 종료")
        
    except KeyboardInterrupt:
        logger.info("사용자에 의해 애플리케이션이 중단되었습니다")
        print("\n\n애플리케이션이 중단되었습니다. 안녕히 가세요!")
        sys.exit(0)
        
    except DocumentRAGError as e:
        # 우리의 커스텀 예외는 이미 사용자 친화적 메시지를 가지고 있음
        logger.error(f"애플리케이션 오류: {e}")
        print(f"\n오류: {e}")
        sys.exit(1)
        
    except Exception as e:
        # 예상치 못한 오류 처리
        user_message = handle_error(e, {"context": "main_application"})
        
        if error_handler.is_critical_error(e):
            logger.critical(f"치명적 오류 발생: {e}")
            print(f"\n치명적 오류가 발생했습니다: {user_message}")
            sys.exit(2)
        else:
            logger.error(f"예상치 못한 오류: {e}")
            print(f"\n오류: {user_message}")
            print("자세한 정보는 로그 파일을 확인해주세요.")
            sys.exit(1)


if __name__ == "__main__":
    main()