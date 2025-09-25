"""CLI Interface for Document RAG English Study application."""

import click
import os
import sys
from pathlib import Path
from typing import Optional

from ..config import ConfigurationManager
from ..models.config import LLMConfig, DocumentConfig, UserConfig
from ..document_manager import DocumentManager


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
def setup() -> None:
    """초기 설정 가이드"""
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
def set_docs(directory: str, no_index: bool) -> None:
    """문서 디렉토리 설정 및 인덱싱"""
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
def set_llm(provider: str, api_key: Optional[str], model: Optional[str], 
           host: str, temperature: float, max_tokens: int) -> None:
    """LLM 제공업체 설정"""
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
def set_language(language: str, learning_level: str, feedback_level: str) -> None:
    """모국어 및 학습 설정"""
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
def chat() -> None:
    """대화형 영어 학습 시작"""
    click.echo("💬 대화형 영어 학습을 시작합니다!")
    click.echo("이 명령어는 아직 구현되지 않았습니다.")


@cli.command()
@click.option('--detailed', is_flag=True, help='상세한 설정 정보 표시')
def status(detailed: bool) -> None:
    """현재 설정 상태 확인"""
    try:
        config_manager = ConfigurationManager()
        setup_status = config_manager.get_setup_status()
        config = config_manager.get_config()
        
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
        click.echo(f"   {status_icon} 모국어: {user_config.native_language}")
        if detailed and setup_status.user_configured:
            click.echo(f"      - 학습 수준: {user_config.learning_level}")
            click.echo(f"      - 피드백 수준: {user_config.feedback_level}")
            click.echo(f"      - 세션 타임아웃: {user_config.session_timeout}분")
        
        # 2. 문서 디렉토리 설정
        doc_config = config.document
        status_icon = "✅" if setup_status.documents_configured else "❌"
        doc_path = doc_config.document_directory or "미설정"
        click.echo(f"   {status_icon} 문서 디렉토리: {doc_path}")
        
        if detailed and setup_status.documents_configured:
            click.echo(f"      - 지원 형식: {', '.join(doc_config.supported_formats)}")
            click.echo(f"      - 청크 크기: {doc_config.chunk_size}")
            click.echo(f"      - 청크 겹침: {doc_config.chunk_overlap}")
            
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
            except Exception:
                click.echo("      - 인덱싱 정보: 확인 불가")
        
        # 3. LLM 설정
        llm_config = config.llm
        status_icon = "✅" if setup_status.llm_configured else "❌"
        llm_provider = llm_config.provider if llm_config else "미설정"
        click.echo(f"   {status_icon} LLM 제공업체: {llm_provider}")
        
        if detailed and setup_status.llm_configured and llm_config:
            click.echo(f"      - 모델: {llm_config.model_name}")
            click.echo(f"      - 온도: {llm_config.temperature}")
            click.echo(f"      - 최대 토큰: {llm_config.max_tokens}")
            if llm_config.provider == 'ollama':
                click.echo(f"      - 서버: {llm_config.host}")
            elif llm_config.api_key:
                masked_key = llm_config.api_key[:8] + "..." if len(llm_config.api_key) > 8 else "***"
                click.echo(f"      - API 키: {masked_key}")
        
        click.echo()
        
        # 설정 파일 정보
        if detailed:
            click.echo("📁 설정 파일 정보:")
            click.echo(f"   - 설정 파일: {config_manager.config_path}")
            click.echo(f"   - 설정 버전: {config.version}")
            if config.created_at:
                click.echo(f"   - 생성일: {config.created_at}")
            if config.updated_at:
                click.echo(f"   - 수정일: {config.updated_at}")
            click.echo()
        
        # 다음 단계 안내
        if not setup_status.overall_complete:
            click.echo("🚀 다음 단계:")
            click.echo("   'setup' 명령어를 실행하여 초기 설정을 완료하세요.")
        else:
            click.echo("💬 사용 가능한 명령어:")
            click.echo("   'chat' - 대화형 영어 학습 시작")
            click.echo("   'help' - 상세 도움말 보기")
    
    except Exception as e:
        click.echo(f"❌ 상태 확인 중 오류 발생: {str(e)}")
        sys.exit(1)


@cli.command()
def help() -> None:
    """상세 도움말"""
    click.echo("📖 Document RAG English Study 도움말")
    click.echo("\n1. 초기 설정:")
    click.echo("   english-study setup")
    click.echo("\n2. 영어 학습 시작:")
    click.echo("   english-study chat")
    click.echo("\n3. 상태 확인:")
    click.echo("   english-study status")


if __name__ == "__main__":
    cli()