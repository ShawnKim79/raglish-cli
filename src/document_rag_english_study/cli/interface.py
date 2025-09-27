"""CLI Interface for Document RAG English Study application."""

import click
import os
import sys
from pathlib import Path
from typing import Optional

from ..config import ConfigurationManager
from ..models.config import LLMConfig, DocumentConfig, UserConfig
from ..document_manager import DocumentManager
from ..conversation.engine import ConversationEngine
from ..rag.engine import RAGEngine
from ..llm import create_language_model
from ..utils import (
    get_logger, handle_error, error_handler_decorator,
    DocumentRAGError, ConfigurationError, ValidationError
)


@click.group(invoke_without_command=True)
@click.option('--version', is_flag=True, help='애플리케이션 버전 정보 표시')
@click.pass_context
def cli(ctx: click.Context, version: bool) -> None:
    """Document RAG English Study - 문서 기반 RAG 영어 학습 CLI 프로그램"""
    if version:
        click.echo("Document RAG English Study v0.1.0")
        return
    
    if ctx.invoked_subcommand is None:
        show_welcome_message()
        click.echo(ctx.get_help())


def show_welcome_message() -> None:
    """환영 메시지 및 기본 안내 표시"""
    click.echo("\n🎓 Document RAG English Study")
    click.echo("관심사 기반 대화형 영어 학습 프로그램\n")
    
    click.echo("주요 명령어:")
    click.echo("  setup        초기 설정")
    click.echo("  set-docs     문서 디렉토리 설정")
    click.echo("  set-llm      LLM 제공업체 설정")
    click.echo("  set-language 모국어 설정")
    click.echo("  chat         대화형 영어 학습 시작")
    click.echo("  status       현재 설정 상태 확인")
    click.echo("  help         상세 도움말")
    click.echo("\n자세한 사용법은 각 명령어에 --help를 추가하세요.\n")


@cli.command()
@error_handler_decorator(context={"command": "setup"})
def setup() -> None:
    """초기 설정 가이드"""
    logger = get_logger(__name__)
    logger.info("초기 설정 시작")
    
    click.echo("🚀 Document RAG English Study 초기 설정을 시작합니다!\n")
    
    try:
        config_manager = ConfigurationManager()
        
        # 현재 설정 상태 확인
        setup_status = config_manager.get_setup_status()
        
        if setup_status.overall_complete:
            click.echo("✅ 모든 설정이 이미 완료되었습니다!")
            click.echo("현재 설정을 확인하려면 'status' 명령어를 사용하세요.")
            return
        
        click.echo("다음 단계를 순서대로 진행합니다:")
        click.echo("1. 모국어 설정")
        click.echo("2. 문서 디렉토리 설정")
        click.echo("3. LLM 제공업체 설정\n")
        
        # 1. 모국어 설정
        if not setup_status.user_configured or not config_manager.get_native_language():
            click.echo("📍 1단계: 모국어 설정")
            language = click.prompt(
                "모국어를 선택하세요 (ko/en/ja/zh)", 
                type=click.Choice(['ko', 'en', 'ja', 'zh'], case_sensitive=False),
                default='ko'
            )
            config_manager.set_native_language(language.lower())
            language_names = {'ko': '한국어', 'en': 'English', 'ja': '日本語', 'zh': '中文'}
            click.echo(f"✅ 모국어가 {language_names.get(language.lower(), language)}로 설정되었습니다.\n")
        else:
            click.echo("✅ 1단계: 모국어 설정 완료\n")
        
        # 2. 문서 디렉토리 설정
        if not setup_status.documents_configured:
            click.echo("📍 2단계: 문서 디렉토리 설정")
            click.echo("영어 학습에 사용할 문서들이 있는 디렉토리를 지정해주세요.")
            click.echo("지원 형식: PDF, DOCX, TXT, MD")
            
            while True:
                directory = click.prompt("문서 디렉토리 경로")
                directory_path = Path(directory).expanduser().resolve()
                
                if not directory_path.exists():
                    click.echo(f"❌ 디렉토리가 존재하지 않습니다: {directory_path}")
                    continue
                
                if not directory_path.is_dir():
                    click.echo(f"❌ 경로가 디렉토리가 아닙니다: {directory_path}")
                    continue
                
                # 문서 디렉토리 설정 및 인덱싱
                try:
                    click.echo(f"📁 문서 디렉토리를 설정하고 인덱싱을 시작합니다...")
                    config_manager.set_document_directory(str(directory_path))
                    
                    # 문서 인덱싱 수행
                    doc_manager = DocumentManager()
                    result = doc_manager.index_documents(str(directory_path))
                    
                    if result.success:
                        click.echo(f"✅ 문서 인덱싱 완료!")
                        click.echo(f"   - 처리된 문서: {result.documents_processed}개")
                        click.echo(f"   - 처리 시간: {result.processing_time:.2f}초")
                        if result.failed_files:
                            click.echo(f"   - 실패한 파일: {len(result.failed_files)}개")
                        break
                    else:
                        click.echo(f"❌ 문서 인덱싱 실패: {result.errors}")
                        continue
                        
                except Exception as e:
                    click.echo(f"❌ 오류 발생: {str(e)}")
                    continue
            
            click.echo()
        else:
            click.echo("✅ 2단계: 문서 디렉토리 설정 완료\n")
        
        # 3. LLM 설정
        if not setup_status.llm_configured:
            click.echo("📍 3단계: LLM 제공업체 설정")
            click.echo("사용할 LLM 제공업체를 선택해주세요:")
            click.echo("  - openai: OpenAI GPT (API 키 필요)")
            click.echo("  - gemini: Google Gemini (API 키 필요)")
            click.echo("  - ollama: 로컬 Ollama 서버 (무료, 로컬 설치 필요)")
            
            provider = click.prompt(
                "LLM 제공업체", 
                type=click.Choice(['openai', 'gemini', 'ollama'], case_sensitive=False)
            ).lower()
            
            api_key = None
            model_name = None
            
            if provider in ['openai', 'gemini']:
                api_key = click.prompt(f"{provider.upper()} API 키", hide_input=True)
                
                # 기본 모델명 제안
                default_models = {
                    'openai': 'gpt-3.5-turbo',
                    'gemini': 'gemini-pro'
                }
                model_name = click.prompt(
                    f"모델명 (기본값: {default_models[provider]})", 
                    default=default_models[provider]
                )
            
            elif provider == 'ollama':
                model_name = click.prompt("Ollama 모델명 (기본값: llama2)", default="llama2")
                host = click.prompt("Ollama 서버 주소 (기본값: localhost:11434)", default="localhost:11434")
            
            try:
                # LLM 설정 저장
                kwargs = {}
                if provider == 'ollama':
                    kwargs['host'] = host
                
                config_manager.set_llm_provider(provider, api_key, model_name, **kwargs)
                click.echo(f"✅ {provider.upper()} 설정이 완료되었습니다!\n")
                
            except Exception as e:
                click.echo(f"❌ LLM 설정 실패: {str(e)}")
                return
        else:
            click.echo("✅ 3단계: LLM 설정 완료\n")
        
        # 최종 설정 확인
        final_status = config_manager.get_setup_status()
        if final_status.overall_complete:
            click.echo("🎉 모든 설정이 완료되었습니다!")
            click.echo("이제 'chat' 명령어로 영어 학습을 시작할 수 있습니다.")
        else:
            click.echo("⚠️  일부 설정이 완료되지 않았습니다.")
            click.echo("'status' 명령어로 현재 상태를 확인해보세요.")
    
    except Exception as e:
        click.echo(f"❌ 설정 중 오류가 발생했습니다: {str(e)}")
        sys.exit(1)


@cli.command("set-docs")
@click.argument('directory', type=click.Path(exists=True, file_okay=False, dir_okay=True))
@click.option('--no-index', is_flag=True, help='인덱싱을 수행하지 않고 디렉토리만 설정')
@error_handler_decorator(context={"command": "set-docs"})
def set_docs(directory: str, no_index: bool) -> None:
    """문서 디렉토리 설정 및 인덱싱"""
    logger = get_logger(__name__)
    logger.info(f"문서 디렉토리 설정 시작: {directory}")
    try:
        directory_path = Path(directory).resolve()
        click.echo(f"📁 문서 디렉토리 설정: {directory_path}")
        
        config_manager = ConfigurationManager()
        
        # 디렉토리 설정
        config_manager.set_document_directory(str(directory_path))
        click.echo("✅ 문서 디렉토리가 설정되었습니다.")
        
        if not no_index:
            # 문서 인덱싱 수행
            click.echo("📚 문서 인덱싱을 시작합니다...")
            
            doc_manager = DocumentManager()
            
            # 진행률 표시를 위한 콜백 설정
            def progress_callback(status):
                if status.is_indexing:
                    progress = (status.processed_documents / status.total_documents * 100) if status.total_documents > 0 else 0
                    click.echo(f"진행률: {progress:.1f}% ({status.processed_documents}/{status.total_documents})")
                    if status.current_file:
                        click.echo(f"현재 처리 중: {Path(status.current_file).name}")
            
            doc_manager.set_progress_callback(progress_callback)
            result = doc_manager.index_documents(str(directory_path))
            
            if result.success:
                click.echo(f"\n✅ 문서 인덱싱 완료!")
                click.echo(f"   - 처리된 문서: {result.documents_processed}개")
                click.echo(f"   - 처리 시간: {result.processing_time:.2f}초")
                
                if result.failed_files:
                    click.echo(f"   - 실패한 파일: {len(result.failed_files)}개")
                    for failed_file, error in result.failed_files.items():
                        click.echo(f"     * {Path(failed_file).name}: {error}")
            else:
                click.echo(f"❌ 문서 인덱싱 실패:")
                for error in result.errors:
                    click.echo(f"   - {error}")
        else:
            click.echo("ℹ️  인덱싱을 건너뛰었습니다. 나중에 'setup' 명령어로 인덱싱을 수행할 수 있습니다.")
    
    except Exception as e:
        click.echo(f"❌ 오류 발생: {str(e)}")
        sys.exit(1)


@cli.command("set-llm")
@click.argument('provider', type=click.Choice(['openai', 'gemini', 'ollama'], case_sensitive=False))
@click.option('--api-key', help='API 키 (OpenAI, Gemini 필수)')
@click.option('--model', help='모델명')
@click.option('--host', default='localhost:11434', help='Ollama 서버 주소 (기본값: localhost:11434)')
@click.option('--temperature', type=float, default=0.7, help='응답 생성 온도 (0.0-2.0, 기본값: 0.7)')
@click.option('--max-tokens', type=int, default=1000, help='최대 토큰 수 (기본값: 1000)')
@error_handler_decorator(context={"command": "set-llm"})
def set_llm(provider: str, api_key: Optional[str], model: Optional[str], 
           host: str, temperature: float, max_tokens: int) -> None:
    """LLM 제공업체 설정"""
    logger = get_logger(__name__)
    logger.info(f"LLM 설정 시작: {provider}")
    try:
        provider = provider.lower()
        click.echo(f"🤖 LLM 설정: {provider.upper()}")
        
        # API 키 검증
        if provider in ['openai', 'gemini'] and not api_key:
            click.echo(f"❌ {provider.upper()}를 사용하려면 --api-key 옵션이 필요합니다.")
            click.echo(f"예시: set-llm {provider} --api-key YOUR_API_KEY")
            return
        
        # 기본 모델명 설정
        if not model:
            default_models = {
                'openai': 'gpt-3.5-turbo',
                'gemini': 'gemini-pro',
                'ollama': 'llama2'
            }
            model = default_models.get(provider, '')
        
        # 설정 매개변수 준비
        kwargs = {
            'temperature': temperature,
            'max_tokens': max_tokens
        }
        
        if provider == 'ollama':
            kwargs['host'] = host
        
        # 설정 저장
        config_manager = ConfigurationManager()
        config_manager.set_llm_provider(provider, api_key, model, **kwargs)
        
        click.echo(f"✅ {provider.upper()} 설정이 완료되었습니다!")
        click.echo(f"   - 제공업체: {provider.upper()}")
        click.echo(f"   - 모델: {model}")
        if provider == 'ollama':
            click.echo(f"   - 서버: {host}")
        click.echo(f"   - 온도: {temperature}")
        click.echo(f"   - 최대 토큰: {max_tokens}")
        
        # 연결 테스트 제안
        if provider == 'ollama':
            click.echo(f"\nℹ️  Ollama 서버({host})가 실행 중인지 확인하세요.")
            click.echo("   ollama serve 명령어로 서버를 시작할 수 있습니다.")
        
    except ValueError as e:
        click.echo(f"❌ 설정 오류: {str(e)}")
        sys.exit(1)
    except Exception as e:
        click.echo(f"❌ 예상치 못한 오류: {str(e)}")
        sys.exit(1)


@cli.command("set-language")
@click.argument('language', type=click.Choice(['ko', 'en', 'ja', 'zh'], case_sensitive=False))
@click.option('--learning-level', type=click.Choice(['beginner', 'intermediate', 'advanced']), 
              default='intermediate', help='영어 학습 수준 (기본값: intermediate)')
@click.option('--feedback-level', type=click.Choice(['minimal', 'normal', 'detailed']), 
              default='normal', help='피드백 상세도 (기본값: normal)')
@error_handler_decorator(context={"command": "set-language"})
def set_language(language: str, learning_level: str, feedback_level: str) -> None:
    """모국어 및 학습 설정"""
    logger = get_logger(__name__)
    logger.info(f"언어 설정 시작: {language}")
    try:
        language = language.lower()
        language_names = {'ko': '한국어', 'en': 'English', 'ja': '日本語', 'zh': '中文'}
        
        click.echo(f"🌍 언어 및 학습 설정")
        click.echo(f"   - 모국어: {language_names.get(language, language)}")
        click.echo(f"   - 학습 수준: {learning_level}")
        click.echo(f"   - 피드백 수준: {feedback_level}")
        
        # 설정 저장
        config_manager = ConfigurationManager()
        
        # 사용자 설정 업데이트
        user_config = config_manager.get_user_config()
        user_config.native_language = language
        user_config.learning_level = learning_level
        user_config.feedback_level = feedback_level
        
        config_manager.update_user_config(user_config)
        
        click.echo("✅ 언어 및 학습 설정이 완료되었습니다!")
        
        # 학습 수준별 안내 메시지
        level_descriptions = {
            'beginner': '기초 문법과 어휘에 중점을 둔 학습을 제공합니다.',
            'intermediate': '실용적인 표현과 문법 교정에 중점을 둔 학습을 제공합니다.',
            'advanced': '고급 표현과 뉘앙스 차이에 중점을 둔 학습을 제공합니다.'
        }
        
        feedback_descriptions = {
            'minimal': '간단한 교정과 핵심 피드백만 제공합니다.',
            'normal': '적절한 수준의 교정과 설명을 제공합니다.',
            'detailed': '상세한 문법 설명과 다양한 표현 방법을 제공합니다.'
        }
        
        click.echo(f"\nℹ️  학습 수준 ({learning_level}): {level_descriptions.get(learning_level, '')}")
        click.echo(f"ℹ️  피드백 수준 ({feedback_level}): {feedback_descriptions.get(feedback_level, '')}")
        
    except Exception as e:
        click.echo(f"❌ 설정 오류: {str(e)}")
        sys.exit(1)


@cli.command()
@click.option('--session-id', help='재개할 세션 ID')
@click.option('--topic', help='선호하는 대화 주제')
@click.option('--save-session/--no-save-session', default=True, help='세션 저장 여부')
@error_handler_decorator(context={"command": "chat"})
def chat(session_id: Optional[str], topic: Optional[str], save_session: bool) -> None:
    """대화형 영어 학습 시작"""
    logger = get_logger(__name__)
    logger.info("대화형 학습 세션 시작")
    try:
        # 설정 상태 확인
        config_manager = ConfigurationManager()
        setup_status = config_manager.get_setup_status()
        
        if not setup_status.overall_complete:
            click.echo("❌ 설정이 완료되지 않았습니다.")
            click.echo("먼저 'setup' 명령어를 실행하여 초기 설정을 완료해주세요.")
            return
        
        # 대화형 학습 세션 시작
        _start_interactive_chat_session(config_manager, session_id, topic, save_session)
        
    except KeyboardInterrupt:
        click.echo("\n\n👋 대화를 종료합니다. 다음에 또 만나요!")
    except Exception as e:
        click.echo(f"❌ 대화 시작 중 오류 발생: {str(e)}")
        sys.exit(1)


@cli.command()
@click.option('--detailed', is_flag=True, help='상세한 설정 정보 표시')
@click.option('--json', 'output_json', is_flag=True, help='JSON 형식으로 출력')
@error_handler_decorator(context={"command": "status"})
def status(detailed: bool, output_json: bool) -> None:
    """현재 설정 및 인덱싱 상태 확인
    
    시스템의 전체 설정 상태, 문서 인덱싱 상태, LLM 연결 상태 등을 확인합니다.
    
    Examples:
        english-study status                    # 기본 상태 확인
        english-study status --detailed         # 상세 정보 포함
        english-study status --json            # JSON 형식 출력
    """
    try:
        config_manager = ConfigurationManager()
        setup_status = config_manager.get_setup_status()
        config = config_manager.get_config()
        
        # JSON 출력 모드
        if output_json:
            import json
            status_data = {
                "overall_complete": setup_status.overall_complete,
                "completion_percentage": setup_status.get_completion_percentage(),
                "missing_steps": setup_status.get_missing_steps(),
                "user_config": {
                    "configured": setup_status.user_configured,
                    "native_language": config.user.native_language,
                    "learning_level": config.user.learning_level,
                    "feedback_level": config.user.feedback_level,
                    "session_timeout": config.user.session_timeout
                },
                "document_config": {
                    "configured": setup_status.documents_configured,
                    "directory": config.document.document_directory,
                    "supported_formats": config.document.supported_formats,
                    "chunk_size": config.document.chunk_size,
                    "chunk_overlap": config.document.chunk_overlap
                },
                "llm_config": {
                    "configured": setup_status.llm_configured,
                    "provider": config.llm.provider if config.llm else None,
                    "model_name": config.llm.model_name if config.llm else None,
                    "temperature": config.llm.temperature if config.llm else None,
                    "max_tokens": config.llm.max_tokens if config.llm else None
                },
                "system_info": {
                    "config_file": str(config_manager.config_path),
                    "version": config.version,
                    "created_at": config.created_at,
                    "updated_at": config.updated_at
                }
            }
            
            # 문서 인덱싱 정보 추가
            try:
                doc_manager = DocumentManager()
                summary = doc_manager.get_document_summary()
                status_data["indexing_info"] = {
                    "total_documents": summary.total_documents,
                    "total_words": summary.total_words,
                    "file_types": summary.file_types,
                    "last_indexed": summary.last_indexed.isoformat() if hasattr(summary, 'last_indexed') and summary.last_indexed else None
                }
            except Exception as e:
                status_data["indexing_info"] = {"error": str(e)}
            
            click.echo(json.dumps(status_data, indent=2, ensure_ascii=False))
            return
        
        # 일반 텍스트 출력 모드
        click.echo("📊 Document RAG English Study 시스템 상태\n")
        
        # 전체 설정 완료 상태
        if setup_status.overall_complete:
            click.echo("🎉 모든 설정이 완료되었습니다!")
        else:
            completion = setup_status.get_completion_percentage()
            click.echo(f"⚠️  설정 진행률: {completion:.1f}%")
            missing_steps = setup_status.get_missing_steps()
            if missing_steps:
                click.echo(f"   미완료 항목: {', '.join(missing_steps)}")
        
        click.echo()
        
        # 개별 설정 상태
        click.echo("📋 설정 상태:")
        
        # 1. 모국어 설정
        user_config = config.user
        status_icon = "✅" if setup_status.user_configured else "❌"
        language_names = {
            'ko': '한국어', 'korean': '한국어',
            'en': 'English', 'english': 'English',
            'ja': '日本語', 'japanese': '日本語',
            'zh': '中文', 'chinese': '中文'
        }
        language_display = language_names.get(user_config.native_language, user_config.native_language)
        click.echo(f"   {status_icon} 모국어: {language_display}")
        
        if detailed and setup_status.user_configured:
            click.echo(f"      - 학습 수준: {user_config.learning_level}")
            click.echo(f"      - 피드백 수준: {user_config.feedback_level}")
            click.echo(f"      - 세션 타임아웃: {user_config.session_timeout}분")
            if user_config.preferred_topics:
                click.echo(f"      - 선호 주제: {', '.join(user_config.preferred_topics[:3])}")
        
        # 2. 문서 디렉토리 설정
        doc_config = config.document
        status_icon = "✅" if setup_status.documents_configured else "❌"
        doc_path = doc_config.document_directory or "미설정"
        if doc_path != "미설정":
            doc_path = Path(doc_path).name + f" ({Path(doc_path).parent})"
        click.echo(f"   {status_icon} 문서 디렉토리: {doc_path}")
        
        if detailed and setup_status.documents_configured:
            click.echo(f"      - 지원 형식: {', '.join(doc_config.supported_formats)}")
            click.echo(f"      - 청크 크기: {doc_config.chunk_size}")
            click.echo(f"      - 청크 겹침: {doc_config.chunk_overlap}")
            click.echo(f"      - 최대 파일 크기: {doc_config.max_file_size // (1024*1024)}MB")
            
            # 인덱싱된 문서 정보
            try:
                doc_manager = DocumentManager()
                summary = doc_manager.get_document_summary()
                click.echo(f"      - 인덱싱된 문서: {summary.total_documents}개")
                if summary.total_documents > 0:
                    click.echo(f"      - 총 단어 수: {summary.total_words:,}개")
                    click.echo(f"      - 파일 형식별:")
                    for file_type, count in summary.file_types.items():
                        click.echo(f"        * {file_type.upper()}: {count}개")
                    
                    # 인덱싱 상태 확인
                    indexing_status = doc_manager.get_indexing_status()
                    if indexing_status.is_indexing:
                        progress = (indexing_status.processed_documents / indexing_status.total_documents * 100) if indexing_status.total_documents > 0 else 0
                        click.echo(f"      - 인덱싱 진행 중: {progress:.1f}%")
                    else:
                        click.echo(f"      - 인덱싱 상태: 완료")
                else:
                    click.echo("      - 인덱싱된 문서가 없습니다")
            except Exception as e:
                click.echo(f"      - 인덱싱 정보: 확인 불가 ({str(e)})")
        
        # 3. LLM 설정
        llm_config = config.llm
        status_icon = "✅" if setup_status.llm_configured else "❌"
        llm_provider = llm_config.provider.upper() if llm_config else "미설정"
        click.echo(f"   {status_icon} LLM 제공업체: {llm_provider}")
        
        if detailed and setup_status.llm_configured and llm_config:
            click.echo(f"      - 모델: {llm_config.model_name}")
            click.echo(f"      - 온도: {llm_config.temperature}")
            click.echo(f"      - 최대 토큰: {llm_config.max_tokens}")
            if llm_config.provider == 'ollama':
                click.echo(f"      - 서버: {llm_config.host}")
                # Ollama 서버 연결 상태 확인
                try:
                    import requests
                    response = requests.get(f"http://{llm_config.host}", timeout=2)
                    click.echo(f"      - 서버 상태: 🟢 연결됨")
                except:
                    click.echo(f"      - 서버 상태: 🔴 연결 불가")
            elif llm_config.api_key:
                masked_key = llm_config.api_key[:8] + "..." if len(llm_config.api_key) > 8 else "***"
                click.echo(f"      - API 키: {masked_key}")
        
        click.echo()
        
        # 시스템 정보
        if detailed:
            click.echo("🖥️  시스템 정보:")
            click.echo(f"   - 설정 파일: {config_manager.config_path}")
            click.echo(f"   - 설정 버전: {config.version}")
            if config.created_at:
                click.echo(f"   - 생성일: {config.created_at}")
            if config.updated_at:
                click.echo(f"   - 수정일: {config.updated_at}")
            
            # 디스크 사용량 정보
            try:
                import shutil
                if config.document.document_directory:
                    total, used, free = shutil.disk_usage(config.document.document_directory)
                    click.echo(f"   - 문서 디렉토리 디스크 사용량: {used // (1024**3):.1f}GB / {total // (1024**3):.1f}GB")
            except:
                pass
            
            click.echo()
        
        # 다음 단계 안내
        if not setup_status.overall_complete:
            click.echo("🚀 다음 단계:")
            missing_steps = setup_status.get_missing_steps()
            for step in missing_steps:
                if "LLM" in step:
                    click.echo("   • 'set-llm' 명령어로 LLM 제공업체를 설정하세요")
                elif "Document" in step:
                    click.echo("   • 'set-docs' 명령어로 문서 디렉토리를 설정하세요")
                elif "User" in step:
                    click.echo("   • 'set-language' 명령어로 사용자 설정을 완료하세요")
            click.echo("   또는 'setup' 명령어로 통합 설정을 진행하세요")
        else:
            click.echo("💬 사용 가능한 명령어:")
            click.echo("   • 'chat' - 대화형 영어 학습 시작")
            click.echo("   • 'help' - 상세 도움말 보기")
            click.echo("   • 'status --detailed' - 더 자세한 상태 정보")
    
    except Exception as e:
        click.echo(f"❌ 상태 확인 중 오류 발생: {str(e)}")
        if detailed:
            import traceback
            click.echo(f"상세 오류: {traceback.format_exc()}")
        sys.exit(1)


@cli.command()
@click.option('--command', help='특정 명령어에 대한 상세 도움말')
@click.option('--examples', is_flag=True, help='사용 예제 표시')
def help(command: Optional[str], examples: bool) -> None:
    """상세 사용법 안내 및 예제 제공
    
    Document RAG English Study의 모든 기능과 사용법을 상세히 안내합니다.
    
    Examples:
        english-study help                      # 전체 도움말
        english-study help --command setup     # setup 명령어 상세 도움말
        english-study help --examples          # 사용 예제 모음
    """
    if command:
        _show_command_help(command)
        return
    
    if examples:
        _show_usage_examples()
        return
    
    # 전체 도움말 표시
    click.echo("📖 Document RAG English Study 상세 도움말\n")
    
    click.echo("🎯 프로그램 개요:")
    click.echo("   관심사 기반 문서를 활용한 RAG(Retrieval-Augmented Generation) 영어 학습 시스템")
    click.echo("   사용자의 관심 분야 문서들을 인덱싱하여 자연스러운 대화형 영어 학습을 제공합니다.\n")
    
    click.echo("🚀 빠른 시작:")
    click.echo("   1. english-study setup          # 초기 설정 (모국어, 문서, LLM)")
    click.echo("   2. english-study chat           # 대화형 영어 학습 시작")
    click.echo("   3. english-study status         # 설정 상태 확인\n")
    
    click.echo("📋 주요 명령어:")
    
    # 설정 관련 명령어
    click.echo("\n  🔧 설정 명령어:")
    click.echo("     setup                        통합 초기 설정 가이드")
    click.echo("     set-docs <directory>         문서 디렉토리 설정 및 인덱싱")
    click.echo("     set-llm <provider>           LLM 제공업체 설정 (openai/gemini/ollama)")
    click.echo("     set-language <language>      모국어 및 학습 설정")
    
    # 학습 관련 명령어
    click.echo("\n  📚 학습 명령어:")
    click.echo("     chat                         대화형 영어 학습 시작")
    click.echo("     chat --topic <topic>         특정 주제로 대화 시작")
    click.echo("     chat --session-id <id>       기존 세션 재개")
    
    # 정보 확인 명령어
    click.echo("\n  ℹ️  정보 명령어:")
    click.echo("     status                       현재 설정 상태 확인")
    click.echo("     status --detailed            상세 설정 정보 표시")
    click.echo("     status --json               JSON 형식으로 상태 출력")
    click.echo("     help                         이 도움말 표시")
    click.echo("     help --command <cmd>         특정 명령어 상세 도움말")
    click.echo("     help --examples              사용 예제 모음")
    
    click.echo("\n💡 지원 기능:")
    click.echo("   • 다양한 문서 형식 지원 (PDF, DOCX, TXT, MD)")
    click.echo("   • 다중 LLM 제공업체 지원 (OpenAI, Google Gemini, Ollama)")
    click.echo("   • 실시간 문법 교정 및 어휘 제안")
    click.echo("   • 관심사 기반 대화 주제 생성")
    click.echo("   • 학습 진행 상황 추적")
    click.echo("   • 다국어 피드백 지원")
    
    click.echo("\n🔗 추가 정보:")
    click.echo("   • 각 명령어에 --help 옵션을 추가하면 상세 사용법을 확인할 수 있습니다")
    click.echo("   • 예시: english-study setup --help")
    click.echo("   • 문제 발생 시 'status --detailed'로 시스템 상태를 확인하세요")
    
    click.echo("\n📞 문제 해결:")
    click.echo("   • 설정이 완료되지 않은 경우: 'setup' 명령어 실행")
    click.echo("   • 문서 인덱싱 실패: 문서 형식 및 권한 확인")
    click.echo("   • LLM 연결 실패: API 키 또는 서버 상태 확인")
    click.echo("   • 대화 시작 불가: 'status' 명령어로 설정 상태 확인")


def _show_command_help(command: str) -> None:
    """특정 명령어에 대한 상세 도움말을 표시합니다.
    
    Args:
        command: 도움말을 표시할 명령어
    """
    command_help = {
        'setup': {
            'description': '초기 설정을 위한 통합 가이드',
            'usage': 'english-study setup',
            'details': [
                '모국어 설정 (한국어, 영어, 일본어, 중국어)',
                '문서 디렉토리 설정 및 자동 인덱싱',
                'LLM 제공업체 설정 (OpenAI, Gemini, Ollama)',
                '설정 완료 상태 확인'
            ],
            'examples': [
                'english-study setup  # 대화형 설정 시작'
            ]
        },
        'set-docs': {
            'description': '문서 디렉토리 설정 및 인덱싱',
            'usage': 'english-study set-docs <directory> [options]',
            'options': [
                '--no-index: 인덱싱을 수행하지 않고 디렉토리만 설정'
            ],
            'details': [
                '지원 형식: PDF, DOCX, TXT, MD',
                '자동 텍스트 추출 및 청크 분할',
                '벡터 임베딩 생성 및 저장',
                '진행률 표시 및 오류 보고'
            ],
            'examples': [
                'english-study set-docs ./documents',
                'english-study set-docs ~/my-papers --no-index'
            ]
        },
        'set-llm': {
            'description': 'LLM 제공업체 설정',
            'usage': 'english-study set-llm <provider> [options]',
            'options': [
                '--api-key: API 키 (OpenAI, Gemini 필수)',
                '--model: 모델명',
                '--host: Ollama 서버 주소',
                '--temperature: 응답 생성 온도 (0.0-2.0)',
                '--max-tokens: 최대 토큰 수'
            ],
            'details': [
                'OpenAI: GPT-3.5-turbo, GPT-4 등',
                'Gemini: gemini-pro, gemini-pro-vision 등',
                'Ollama: 로컬 모델 (llama2, mistral 등)',
                'API 키는 환경 변수로도 설정 가능'
            ],
            'examples': [
                'english-study set-llm openai --api-key sk-...',
                'english-study set-llm gemini --api-key AIza...',
                'english-study set-llm ollama --model llama2'
            ]
        },
        'set-language': {
            'description': '모국어 및 학습 설정',
            'usage': 'english-study set-language <language> [options]',
            'options': [
                '--learning-level: 학습 수준 (beginner/intermediate/advanced)',
                '--feedback-level: 피드백 상세도 (minimal/normal/detailed)'
            ],
            'details': [
                '지원 언어: ko(한국어), en(영어), ja(일본어), zh(중국어)',
                '학습 수준에 따른 맞춤형 피드백',
                '피드백 상세도 조절 가능'
            ],
            'examples': [
                'english-study set-language ko',
                'english-study set-language ko --learning-level beginner',
                'english-study set-language en --feedback-level detailed'
            ]
        },
        'chat': {
            'description': '대화형 영어 학습 시작',
            'usage': 'english-study chat [options]',
            'options': [
                '--session-id: 재개할 세션 ID',
                '--topic: 선호하는 대화 주제',
                '--save-session/--no-save-session: 세션 저장 여부'
            ],
            'details': [
                'RAG 기반 관심사 대화',
                '실시간 문법 교정',
                '어휘 향상 제안',
                '학습 진행 상황 추적',
                '대화 중 특수 명령어 지원'
            ],
            'examples': [
                'english-study chat',
                'english-study chat --topic "artificial intelligence"',
                'english-study chat --session-id abc123'
            ],
            'chat_commands': [
                '/help: 대화 중 도움말',
                '/topics: 대화 주제 제안',
                '/progress: 학습 진행 상황',
                '/quit: 대화 종료'
            ]
        },
        'status': {
            'description': '현재 설정 및 인덱싱 상태 확인',
            'usage': 'english-study status [options]',
            'options': [
                '--detailed: 상세한 설정 정보 표시',
                '--json: JSON 형식으로 출력'
            ],
            'details': [
                '전체 설정 완료 상태',
                '개별 구성 요소 상태',
                '문서 인덱싱 정보',
                'LLM 연결 상태',
                '시스템 정보'
            ],
            'examples': [
                'english-study status',
                'english-study status --detailed',
                'english-study status --json'
            ]
        }
    }
    
    if command not in command_help:
        click.echo(f"❌ 알 수 없는 명령어: {command}")
        click.echo("사용 가능한 명령어: setup, set-docs, set-llm, set-language, chat, status")
        return
    
    help_info = command_help[command]
    
    click.echo(f"📖 '{command}' 명령어 상세 도움말\n")
    click.echo(f"📝 설명: {help_info['description']}")
    click.echo(f"💻 사용법: {help_info['usage']}")
    
    if 'options' in help_info:
        click.echo("\n⚙️  옵션:")
        for option in help_info['options']:
            click.echo(f"   {option}")
    
    if 'details' in help_info:
        click.echo("\n🔍 상세 기능:")
        for detail in help_info['details']:
            click.echo(f"   • {detail}")
    
    if 'examples' in help_info:
        click.echo("\n💡 사용 예제:")
        for example in help_info['examples']:
            click.echo(f"   {example}")
    
    if 'chat_commands' in help_info:
        click.echo("\n🗨️  대화 중 명령어:")
        for cmd in help_info['chat_commands']:
            click.echo(f"   {cmd}")


def _show_usage_examples() -> None:
    """다양한 사용 예제를 표시합니다."""
    click.echo("💡 Document RAG English Study 사용 예제 모음\n")
    
    click.echo("🚀 1. 처음 사용하는 경우:")
    click.echo("   # 통합 설정으로 시작")
    click.echo("   english-study setup")
    click.echo("   ")
    click.echo("   # 또는 단계별 설정")
    click.echo("   english-study set-language ko")
    click.echo("   english-study set-docs ./my-documents")
    click.echo("   english-study set-llm openai --api-key sk-your-key")
    click.echo("   english-study chat")
    
    click.echo("\n📚 2. 다양한 문서 형식 활용:")
    click.echo("   # PDF 논문 모음으로 학습")
    click.echo("   english-study set-docs ~/research-papers")
    click.echo("   ")
    click.echo("   # 기술 문서로 학습")
    click.echo("   english-study set-docs ./tech-docs")
    click.echo("   ")
    click.echo("   # 소설이나 에세이로 학습")
    click.echo("   english-study set-docs ~/books")
    
    click.echo("\n🤖 3. 다양한 LLM 제공업체 사용:")
    click.echo("   # OpenAI GPT 사용")
    click.echo("   english-study set-llm openai --api-key sk-... --model gpt-4")
    click.echo("   ")
    click.echo("   # Google Gemini 사용")
    click.echo("   english-study set-llm gemini --api-key AIza...")
    click.echo("   ")
    click.echo("   # 로컬 Ollama 사용 (무료)")
    click.echo("   english-study set-llm ollama --model llama2")
    
    click.echo("\n🎯 4. 맞춤형 학습 설정:")
    click.echo("   # 초보자 설정")
    click.echo("   english-study set-language ko --learning-level beginner --feedback-level detailed")
    click.echo("   ")
    click.echo("   # 고급자 설정")
    click.echo("   english-study set-language en --learning-level advanced --feedback-level minimal")
    
    click.echo("\n💬 5. 대화형 학습 활용:")
    click.echo("   # 기본 대화 시작")
    click.echo("   english-study chat")
    click.echo("   ")
    click.echo("   # 특정 주제로 대화")
    click.echo("   english-study chat --topic \"machine learning\"")
    click.echo("   ")
    click.echo("   # 이전 세션 재개")
    click.echo("   english-study chat --session-id session_20240101_001")
    
    click.echo("\n🔍 6. 상태 확인 및 문제 해결:")
    click.echo("   # 기본 상태 확인")
    click.echo("   english-study status")
    click.echo("   ")
    click.echo("   # 상세 정보 확인")
    click.echo("   english-study status --detailed")
    click.echo("   ")
    click.echo("   # JSON 형식으로 출력 (스크립트 활용)")
    click.echo("   english-study status --json")
    
    click.echo("\n🔧 7. 고급 사용법:")
    click.echo("   # 문서만 설정하고 나중에 인덱싱")
    click.echo("   english-study set-docs ./docs --no-index")
    click.echo("   ")
    click.echo("   # 세션 저장 없이 대화")
    click.echo("   english-study chat --no-save-session")
    click.echo("   ")
    click.echo("   # 특정 명령어 도움말")
    click.echo("   english-study help --command chat")
    
    click.echo("\n🌟 8. 실제 학습 시나리오:")
    click.echo("   # 시나리오 1: 논문 읽기 학습")
    click.echo("   english-study set-docs ~/research-papers")
    click.echo("   english-study chat --topic \"research methodology\"")
    click.echo("   ")
    click.echo("   # 시나리오 2: 기술 블로그 학습")
    click.echo("   english-study set-docs ~/tech-articles")
    click.echo("   english-study chat --topic \"software development\"")
    click.echo("   ")
    click.echo("   # 시나리오 3: 비즈니스 문서 학습")
    click.echo("   english-study set-docs ~/business-docs")
    click.echo("   english-study chat --topic \"business strategy\"")
    
    click.echo("\n💡 팁:")
    click.echo("   • 문서는 관심 있는 주제로 구성하면 학습 효과가 높아집니다")
    click.echo("   • 정기적으로 새로운 문서를 추가하여 학습 내용을 확장하세요")
    click.echo("   • 대화 중 모르는 표현이 나오면 자연스럽게 질문하세요")
    click.echo("   • 학습 수준과 피드백 레벨을 조정하여 최적의 학습 경험을 찾으세요")


def _start_interactive_chat_session(
    config_manager: ConfigurationManager,
    session_id: Optional[str] = None,
    preferred_topic: Optional[str] = None,
    save_session: bool = True
) -> None:
    """대화형 학습 세션을 시작합니다.
    
    Args:
        config_manager: 설정 관리자
        session_id: 재개할 세션 ID (선택사항)
        preferred_topic: 선호하는 대화 주제 (선택사항)
        save_session: 세션 저장 여부
    """
    try:
        config = config_manager.get_config()
        
        # 대화 엔진 초기화
        click.echo("🚀 대화형 영어 학습을 준비하고 있습니다...")
        
        # RAG 엔진 초기화
        rag_engine = RAGEngine()
        
        # LLM 초기화
        llm = create_language_model(config.llm)
        
        # 대화 엔진 생성
        conversation_engine = ConversationEngine(
            rag_engine=rag_engine,
            llm=llm,
            user_language=config.user.native_language
        )
        
        # 대화 세션 시작
        if session_id:
            click.echo(f"📂 기존 세션을 재개합니다: {session_id}")
        else:
            click.echo("✨ 새로운 대화 세션을 시작합니다!")
        
        session = conversation_engine.start_conversation(
            preferred_topic=preferred_topic,
            session_id=session_id
        )
        
        # 환영 메시지 및 사용법 안내
        _show_chat_welcome_message(config.user.native_language)
        
        # 첫 번째 메시지 표시 (대화 시작 메시지)
        if session.messages:
            last_message = session.messages[-1]
            if last_message.role == "assistant":
                _display_assistant_message(last_message.content)
        
        # 대화 루프 시작
        _run_conversation_loop(conversation_engine, session, save_session)
        
    except Exception as e:
        click.echo(f"❌ 대화 엔진 초기화 실패: {str(e)}")
        raise


def _show_chat_welcome_message(user_language: str) -> None:
    """대화 시작 환영 메시지를 표시합니다.
    
    Args:
        user_language: 사용자 모국어
    """
    if user_language == "korean":
        click.echo("\n" + "="*60)
        click.echo("🎓 Document RAG English Study - 대화형 학습")
        click.echo("="*60)
        click.echo("\n💡 사용법:")
        click.echo("  • 영어로 자유롭게 대화해보세요")
        click.echo("  • 문법 교정과 어휘 제안을 받을 수 있습니다")
        click.echo("  • '/help' - 도움말 보기")
        click.echo("  • '/topics' - 대화 주제 제안 받기")
        click.echo("  • '/progress' - 학습 진행 상황 확인")
        click.echo("  • '/quit' 또는 Ctrl+C - 대화 종료")
        click.echo("\n" + "="*60 + "\n")
    else:
        click.echo("\n" + "="*60)
        click.echo("🎓 Document RAG English Study - Interactive Learning")
        click.echo("="*60)
        click.echo("\n💡 How to use:")
        click.echo("  • Chat freely in English")
        click.echo("  • Get grammar corrections and vocabulary suggestions")
        click.echo("  • '/help' - Show help")
        click.echo("  • '/topics' - Get conversation topic suggestions")
        click.echo("  • '/progress' - Check learning progress")
        click.echo("  • '/quit' or Ctrl+C - End conversation")
        click.echo("\n" + "="*60 + "\n")


def _run_conversation_loop(
    conversation_engine: ConversationEngine,
    session,
    save_session: bool
) -> None:
    """대화 루프를 실행합니다.
    
    Args:
        conversation_engine: 대화 엔진
        session: 현재 세션
        save_session: 세션 저장 여부
    """
    try:
        while True:
            # 사용자 입력 받기
            user_input = click.prompt("You", type=str, prompt_suffix=": ").strip()
            
            if not user_input:
                continue
            
            # 특수 명령어 처리
            if user_input.startswith('/'):
                if _handle_special_command(user_input, conversation_engine, session):
                    break  # /quit 명령어인 경우
                continue
            
            # 사용자 입력 처리
            try:
                click.echo("🤔 생각 중...")
                response = conversation_engine.process_user_input(user_input, session)
                
                # 응답 표시
                _display_conversation_response(response)
                
            except Exception as e:
                click.echo(f"❌ 응답 생성 중 오류 발생: {str(e)}")
                click.echo("다시 시도해주세요.")
    
    except KeyboardInterrupt:
        click.echo("\n")
        pass  # 정상적인 종료 처리
    
    finally:
        # 세션 종료 처리
        _end_chat_session(conversation_engine, session, save_session)


def _handle_special_command(
    command: str,
    conversation_engine: ConversationEngine,
    session
) -> bool:
    """특수 명령어를 처리합니다.
    
    Args:
        command: 입력된 명령어
        conversation_engine: 대화 엔진
        session: 현재 세션
        
    Returns:
        bool: 대화를 종료해야 하는 경우 True
    """
    command = command.lower().strip()
    
    if command in ['/quit', '/exit', '/q']:
        return True
    
    elif command == '/help':
        _show_chat_help()
    
    elif command == '/topics':
        _show_topic_suggestions(conversation_engine)
    
    elif command == '/progress':
        _show_learning_progress(conversation_engine, session)
    
    elif command == '/session':
        _show_session_info(session)
    
    else:
        click.echo(f"❓ 알 수 없는 명령어: {command}")
        click.echo("'/help'를 입력하여 사용 가능한 명령어를 확인하세요.")
    
    return False


def _show_chat_help() -> None:
    """대화 중 도움말을 표시합니다."""
    click.echo("\n📖 대화 중 사용 가능한 명령어:")
    click.echo("  /help     - 이 도움말 표시")
    click.echo("  /topics   - 대화 주제 제안")
    click.echo("  /progress - 학습 진행 상황")
    click.echo("  /session  - 현재 세션 정보")
    click.echo("  /quit     - 대화 종료")
    click.echo()


def _show_topic_suggestions(conversation_engine: ConversationEngine) -> None:
    """대화 주제 제안을 표시합니다.
    
    Args:
        conversation_engine: 대화 엔진
    """
    try:
        topics = conversation_engine.suggest_conversation_topics(count=5)
        
        if topics:
            click.echo("\n💡 추천 대화 주제:")
            for i, topic in enumerate(topics, 1):
                click.echo(f"  {i}. {topic}")
            click.echo("\n이 중 하나를 선택해서 대화해보세요!")
        else:
            click.echo("\n😅 현재 추천할 수 있는 주제가 없습니다.")
            click.echo("문서가 인덱싱되어 있는지 확인해보세요.")
        
        click.echo()
        
    except Exception as e:
        click.echo(f"❌ 주제 제안 중 오류 발생: {str(e)}")


def _show_learning_progress(conversation_engine: ConversationEngine, session) -> None:
    """학습 진행 상황을 표시합니다.
    
    Args:
        conversation_engine: 대화 엔진
        session: 현재 세션
    """
    try:
        # 현재 세션 통계
        click.echo(f"\n📊 현재 세션 통계:")
        click.echo(f"  • 대화 시간: {len(session.messages)}개 메시지")
        click.echo(f"  • 다룬 주제: {len(session.topics_covered)}개")
        click.echo(f"  • 학습 포인트: {len(session.learning_points)}개")
        
        if session.topics_covered:
            click.echo(f"  • 주요 주제: {', '.join(session.topics_covered[:3])}")
        
        # 전체 학습 진행 상황
        progress = conversation_engine.get_learning_progress()
        if progress:
            click.echo(f"\n📈 전체 학습 진행 상황:")
            if 'total_sessions' in progress:
                click.echo(f"  • 총 세션 수: {progress['total_sessions']}개")
            if 'total_messages' in progress:
                click.echo(f"  • 총 메시지 수: {progress['total_messages']}개")
            if 'average_session_length' in progress:
                click.echo(f"  • 평균 세션 길이: {progress['average_session_length']:.1f}분")
        
        click.echo()
        
    except Exception as e:
        click.echo(f"❌ 진행 상황 조회 중 오류 발생: {str(e)}")


def _show_session_info(session) -> None:
    """현재 세션 정보를 표시합니다.
    
    Args:
        session: 현재 세션
    """
    click.echo(f"\n📋 현재 세션 정보:")
    click.echo(f"  • 세션 ID: {session.session_id}")
    click.echo(f"  • 시작 시간: {session.start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    click.echo(f"  • 사용자 언어: {session.user_language}")
    click.echo(f"  • 활성 상태: {'예' if session.is_active() else '아니오'}")
    click.echo()


def _display_assistant_message(content: str) -> None:
    """어시스턴트 메시지를 표시합니다.
    
    Args:
        content: 메시지 내용
    """
    click.echo(f"\n🤖 Assistant: {content}\n")


def _display_conversation_response(response) -> None:
    """대화 응답을 표시합니다.
    
    Args:
        response: ConversationResponse 객체
    """
    # 메인 응답 표시
    click.echo(f"\n🤖 Assistant: {response.response_text}")
    
    # 학습 피드백 표시
    if response.learning_feedback and response.learning_feedback.has_feedback():
        click.echo("\n" + "─" * 50)
        click.echo("📚 학습 피드백:")
        
        # 문법 교정
        if response.learning_feedback.corrections:
            click.echo("\n📝 문법 교정:")
            for correction in response.learning_feedback.corrections:
                click.echo(f"  • {correction.original_text} → {correction.corrected_text}")
                click.echo(f"    💡 {correction.explanation}")
        
        # 어휘 제안
        if response.learning_feedback.vocabulary_suggestions:
            click.echo("\n📖 어휘 제안:")
            for vocab in response.learning_feedback.vocabulary_suggestions:
                click.echo(f"  • {vocab.word}: {vocab.definition}")
                if vocab.usage_example:
                    click.echo(f"    예시: {vocab.usage_example}")
        
        # 격려 메시지
        if response.learning_feedback.encouragement:
            click.echo(f"\n💪 {response.learning_feedback.encouragement}")
        
        click.echo("─" * 50)
    
    # 제안 주제 표시 (간단히)
    if response.suggested_topics:
        topics_text = ", ".join(response.suggested_topics[:3])
        click.echo(f"\n💡 다음 주제도 이야기해보세요: {topics_text}")
    
    click.echo()


def _end_chat_session(
    conversation_engine: ConversationEngine,
    session,
    save_session: bool
) -> None:
    """대화 세션을 종료합니다.
    
    Args:
        conversation_engine: 대화 엔진
        session: 현재 세션
        save_session: 세션 저장 여부
    """
    try:
        click.echo("\n🔄 세션을 종료하고 있습니다...")
        
        if save_session:
            # 세션 종료 및 요약 생성
            summary = conversation_engine.end_conversation(session)
            
            click.echo("✅ 세션이 저장되었습니다!")
            click.echo(f"\n📊 세션 요약:")
            click.echo(f"  • 세션 ID: {summary['session_id']}")
            click.echo(f"  • 대화 시간: {summary['duration_seconds']:.0f}초")
            click.echo(f"  • 총 메시지: {summary['total_messages']}개")
            click.echo(f"  • 학습 포인트: {summary['learning_points_count']}개")
            
            if summary['topics_covered']:
                topics_text = ", ".join(summary['topics_covered'][:3])
                click.echo(f"  • 주요 주제: {topics_text}")
            
            if summary['recommendations']:
                click.echo(f"\n💡 학습 권장사항:")
                for rec in summary['recommendations'][:2]:
                    click.echo(f"  • {rec}")
        else:
            click.echo("ℹ️  세션이 저장되지 않았습니다.")
        
        click.echo("\n👋 대화를 종료합니다. 좋은 하루 되세요!")
        
    except Exception as e:
        click.echo(f"❌ 세션 종료 중 오류 발생: {str(e)}")
        click.echo("세션 데이터가 손실될 수 있습니다.")


if __name__ == "__main__":
    cli()