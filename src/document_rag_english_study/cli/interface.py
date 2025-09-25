"""CLI Interface for Document RAG English Study application."""

import click
from typing import Optional


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
    click.echo("🚀 초기 설정을 시작합니다!")
    click.echo("이 명령어는 아직 구현되지 않았습니다.")


@cli.command("set-docs")
@click.argument('directory', type=click.Path(exists=True, file_okay=False, dir_okay=True))
def set_docs(directory: str) -> None:
    """문서 디렉토리 설정"""
    click.echo(f"📁 문서 디렉토리 설정: {directory}")
    click.echo("이 명령어는 아직 구현되지 않았습니다.")


@cli.command("set-llm")
@click.argument('provider', type=click.Choice(['openai', 'gemini', 'ollama'], case_sensitive=False))
@click.option('--api-key', help='API 키')
@click.option('--model', help='모델명')
def set_llm(provider: str, api_key: Optional[str], model: Optional[str]) -> None:
    """LLM 제공업체 설정"""
    click.echo(f"🤖 LLM 설정: {provider.upper()}")
    if provider in ['openai', 'gemini'] and not api_key:
        click.echo(f"❌ {provider.upper()}를 사용하려면 --api-key 옵션이 필요합니다.")
        return
    click.echo("이 명령어는 아직 구현되지 않았습니다.")


@cli.command("set-language")
@click.argument('language', type=click.Choice(['ko', 'en', 'ja', 'zh'], case_sensitive=False))
def set_language(language: str) -> None:
    """모국어 설정"""
    language_names = {'ko': '한국어', 'en': 'English', 'ja': '日本語', 'zh': '中文'}
    click.echo(f"🌍 모국어 설정: {language_names.get(language, language)}")
    click.echo("이 명령어는 아직 구현되지 않았습니다.")


@cli.command()
def chat() -> None:
    """대화형 영어 학습 시작"""
    click.echo("💬 대화형 영어 학습을 시작합니다!")
    click.echo("이 명령어는 아직 구현되지 않았습니다.")


@cli.command()
def status() -> None:
    """현재 설정 상태 확인"""
    click.echo("📊 시스템 상태")
    click.echo("문서 디렉토리: ❌ 미설정")
    click.echo("LLM 연결: ❌ 미설정")
    click.echo("모국어: ❌ 미설정")
    click.echo("이 명령어는 아직 구현되지 않았습니다.")


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