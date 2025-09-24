# Document RAG English Study

사용자의 관심사 문서를 기반으로 RAG 시스템을 구축하고, 자연스러운 대화를 통해 영어 학습을 지원하는 CLI 프로그램입니다.

## Project Overview

이 프로젝트는 사용자가 자신이 관심 있어 하는 문서들을 RAG(Retrieval-Augmented Generation) 시스템에 인덱싱하고, 이를 기반으로 관심사 기반의 대화를 통해 영어 학습을 할 수 있는 환경을 제공합니다.

### 핵심 기능

1. **문서 디렉토리 설정**: 관심사 문서들이 저장된 디렉토리 지정
2. **LLM 모델 연결**: OpenAI GPT, Google Gemini, Ollama 지원
3. **모국어 설정**: 이해하기 쉬운 학습 지원을 위한 언어 설정
4. **대화형 영어 학습**: 문서 내용을 활용한 자연스러운 영어 대화

## Project Structure

```
document-rag-english-study/
├── src/
│   ├── core/           # 핵심 애플리케이션 로직
│   ├── models/         # 데이터 모델
│   ├── services/       # 비즈니스 로직 서비스
│   ├── cli/           # CLI 인터페이스
│   ├── utils/         # 유틸리티 함수
│   └── config/        # 설정 관리
├── tests/             # 테스트 코드
├── docs/              # 문서
├── pyproject.toml     # 프로젝트 설정 (uv 사용)
└── README.md          # 프로젝트 설명
```

## Development Standards

### Code Style
- **Language**: Python 3.8+
- **Code Formatting**: Black formatter 사용
- **Import Sorting**: isort 사용
- **Linting**: flake8 또는 pylint 사용
- **Type Hints**: 모든 함수와 메서드에 타입 힌트 필수

### Naming Conventions
- **Classes**: PascalCase (예: `DocumentManager`, `RAGEngine`)
- **Functions/Methods**: snake_case (예: `process_document`, `generate_response`)
- **Variables**: snake_case (예: `user_input`, `conversation_session`)
- **Constants**: UPPER_SNAKE_CASE (예: `DEFAULT_LANGUAGE`, `MAX_CHUNK_SIZE`)
- **Files/Modules**: snake_case (예: `document_manager.py`, `rag_engine.py`)

### Testing Requirements
- **Unit Tests**: 각 클래스와 함수에 대한 단위 테스트 필수
- **Integration Tests**: 주요 워크플로우에 대한 통합 테스트
- **Test Coverage**: 최소 90% 코드 커버리지 목표
- **Test Framework**: pytest 사용

### Documentation
- **Docstrings**: 모든 클래스, 함수, 메서드에 Google 스타일 docstring 작성
- **API Documentation**: 주요 클래스와 메서드에 대한 API 문서
- **Examples**: 실제 사용 예제 코드 제공

## CLI Commands

- `setup`: 초기 설정 (문서 디렉토리, LLM, 모국어)
- `set-docs <directory>`: 문서 디렉토리 설정 및 인덱싱
- `set-llm <provider>`: LLM 제공업체 설정 (openai, gemini, ollama)
- `set-language <language>`: 모국어 설정
- `chat`: 관심사 기반 영어 학습 대화 시작
- `status`: 현재 설정 및 인덱싱 상태 확인
- `help`: 도움말

## Technology Stack

- **CLI Framework**: Click
- **Vector Database**: ChromaDB
- **Embeddings**: sentence-transformers
- **Language Models**: OpenAI GPT API, Google Gemini API, Ollama
- **Document Processing**: 
  - PDF: PyPDF2
  - DOCX: python-docx
  - Text/Markdown: 내장 라이브러리
- **Configuration**: YAML
- **Package Management**: uv

## Installation & Setup

```bash
# 프로젝트 클론
git clone <repository-url>
cd document-rag-english-study

# uv를 사용한 가상환경 생성 및 의존성 설치
uv venv
uv pip install -e .

# 초기 설정
python -m src.cli setup
```

## Usage Example

```bash
# 문서 디렉토리 설정
python -m src.cli set-docs /path/to/your/documents

# LLM 설정
python -m src.cli set-llm openai

# 모국어 설정
python -m src.cli set-language korean

# 영어 학습 대화 시작
python -m src.cli chat
```

## Architecture

### High-Level Components

- **DocumentManager**: 문서 인덱싱 및 관리
- **RAGEngine**: 벡터 검색 및 컨텍스트 제공
- **ConversationEngine**: 대화형 학습 엔진
- **LearningAssistant**: 영어 교정 및 피드백
- **ConfigurationManager**: 설정 관리

### RAG Implementation
- **Chunk Size**: 500-1000 토큰 단위로 문서 분할
- **Embedding Model**: 다국어 지원 임베딩 모델 사용
- **Vector Search**: ChromaDB를 통한 효율적인 유사도 검색
- **Context Management**: 관련성 높은 컨텍스트 선별

### Learning Engine
- **Natural Conversation**: 자연스러운 대화 흐름 유지
- **Adaptive Feedback**: 사용자 수준에 맞는 피드백 제공
- **Progress Tracking**: 학습 진행 상황 추적 및 분석
- **Multilingual Support**: 다양한 모국어 지원

## Contributing

1. 각 작업 단계 완료 후 반드시 commit
2. uv를 사용한 패키지 관리
3. 새로운 패키지 필요 시 개발자에게 요청
4. 코드 품질 가이드라인 준수
5. 테스트 코드 작성 필수

## License

MIT License