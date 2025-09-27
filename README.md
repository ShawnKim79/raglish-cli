# Document RAG English Study

🎓 **관심사 기반 대화형 영어 학습 CLI 프로그램**

사용자의 관심사 문서를 RAG(Retrieval-Augmented Generation) 시스템에 인덱싱하여, 자연스럽고 흥미로운 대화를 통해 영어 학습을 지원하는 CLI 애플리케이션입니다.

## ✨ 주요 특징

- 📚 **다양한 문서 형식 지원**: PDF, DOCX, TXT, MD 파일 자동 처리
- 🤖 **다중 LLM 지원**: OpenAI GPT, Google Gemini, Ollama 로컬 모델
- 🌍 **다국어 피드백**: 한국어, 영어, 일본어, 중국어 지원
- 💬 **실시간 학습 피드백**: 문법 교정, 어휘 제안, 발음 가이드
- 🎯 **관심사 기반 학습**: 사용자의 관심 분야로 자연스러운 대화 유도
- 📊 **학습 진행 추적**: 세션별 학습 포인트 및 진행 상황 기록

## 🚀 빠른 시작

### 1. 설치

```bash
# 저장소 클론
git clone https://github.com/your-username/document-rag-english-study.git
cd document-rag-english-study

# uv를 사용한 가상환경 생성 및 의존성 설치
uv venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
uv pip install -e .
```

### 2. 초기 설정

```bash
# 통합 설정 가이드 실행
english-study setup
```

### 3. 영어 학습 시작

```bash
# 대화형 영어 학습 시작
english-study chat
```

## 📋 상세 설치 가이드

### 사전 요구사항

- **Python 3.9+** (권장: Python 3.11+)
- **uv** (Python 패키지 관리자)
- **Git**

### uv 설치

```bash
# macOS/Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# Windows (PowerShell)
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"

# 또는 pip를 통한 설치
pip install uv
```

### 프로젝트 설치

```bash
# 1. 저장소 클론
git clone https://github.com/your-username/document-rag-english-study.git
cd document-rag-english-study

# 2. 가상환경 생성 및 활성화
uv venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# 3. 의존성 설치
uv pip install -e .

# 4. 개발 의존성 설치 (선택사항)
uv pip install -e ".[dev]"
```

### LLM 제공업체별 추가 설정

#### OpenAI GPT
```bash
# OpenAI API 키 발급: https://platform.openai.com/api-keys
export OPENAI_API_KEY="your-api-key-here"
```

#### Google Gemini
```bash
# Google AI Studio에서 API 키 발급: https://makersuite.google.com/app/apikey
export GOOGLE_API_KEY="your-api-key-here"
```

#### Ollama (로컬 모델)
```bash
# Ollama 설치: https://ollama.ai/download
# 모델 다운로드 및 서버 실행
ollama pull llama2
ollama serve
```

## 🎯 사용법

### 기본 명령어

```bash
# 도움말 보기
english-study --help

# 시스템 상태 확인
english-study status

# 상세 상태 정보
english-study status --detailed
```

### 초기 설정

```bash
# 통합 설정 가이드 (권장)
english-study setup

# 또는 개별 설정
english-study set-language ko                    # 모국어 설정
english-study set-docs ./my-documents           # 문서 디렉토리 설정
english-study set-llm openai --api-key YOUR_KEY # LLM 설정
```

### 대화형 학습

```bash
# 기본 대화 시작
english-study chat

# 특정 주제로 대화 시작
english-study chat --topic "artificial intelligence"

# 기존 세션 재개
english-study chat --session-id abc123
```

### 고급 설정

```bash
# 학습 수준 및 피드백 설정
english-study set-language ko --learning-level advanced --feedback-level detailed

# LLM 매개변수 조정
english-study set-llm openai --model gpt-4 --temperature 0.8 --max-tokens 1500

# 문서 디렉토리 설정 (인덱싱 제외)
english-study set-docs ./documents --no-index
```

## 📖 사용 예제

### 시나리오 1: 기술 문서로 영어 학습

```bash
# 1. 기술 문서 디렉토리 설정
english-study set-docs ~/Documents/tech-articles

# 2. OpenAI GPT 설정
english-study set-llm openai --api-key sk-...

# 3. 한국어 피드백으로 고급 학습 설정
english-study set-language ko --learning-level advanced

# 4. AI 주제로 대화 시작
english-study chat --topic "machine learning"
```

### 시나리오 2: 로컬 모델로 개인정보 보호

```bash
# 1. Ollama 서버 실행 (별도 터미널)
ollama serve

# 2. 로컬 모델 설정
english-study set-llm ollama --model llama2

# 3. 개인 문서로 학습
english-study set-docs ~/private-docs

# 4. 대화 시작
english-study chat
```

### 시나리오 3: 학술 논문으로 전문 영어 학습

```bash
# 1. 논문 디렉토리 설정
english-study set-docs ~/research-papers

# 2. Gemini 모델 설정
english-study set-llm gemini --api-key AIza...

# 3. 상세 피드백 설정
english-study set-language en --feedback-level detailed

# 4. 특정 세션 재개
english-study chat --session-id research-session-001
```

## 🏗️ 프로젝트 구조

```
document-rag-english-study/
├── src/document_rag_english_study/
│   ├── cli/                    # CLI 인터페이스
│   │   ├── interface.py        # 주요 명령어 구현
│   │   └── main.py            # CLI 진입점
│   ├── config/                 # 설정 관리
│   │   ├── manager.py         # 설정 매니저
│   │   └── utils.py           # 설정 유틸리티
│   ├── conversation/           # 대화형 학습 엔진
│   │   ├── engine.py          # 대화 엔진
│   │   ├── dialog_manager.py  # 대화 관리
│   │   ├── learning_assistant.py # 학습 도우미
│   │   └── session_tracker.py # 세션 추적
│   ├── document_manager/       # 문서 처리
│   │   ├── manager.py         # 문서 관리자
│   │   └── parser.py          # 문서 파서
│   ├── llm/                   # 언어 모델 추상화
│   │   ├── base.py            # 기본 인터페이스
│   │   ├── openai_model.py    # OpenAI 구현
│   │   ├── gemini_model.py    # Gemini 구현
│   │   └── ollama_model.py    # Ollama 구현
│   ├── models/                # 데이터 모델
│   │   ├── config.py          # 설정 모델
│   │   ├── conversation.py    # 대화 모델
│   │   ├── document.py        # 문서 모델
│   │   └── llm.py            # LLM 모델
│   ├── rag/                   # RAG 엔진
│   │   ├── engine.py          # RAG 엔진
│   │   ├── embedding_generator.py # 임베딩 생성
│   │   └── vector_database.py # 벡터 데이터베이스
│   └── utils/                 # 유틸리티
│       ├── error_handler.py   # 오류 처리
│       ├── exceptions.py      # 커스텀 예외
│       └── logging_config.py  # 로깅 설정
├── tests/                     # 테스트 코드
├── test_docs/                 # 샘플 문서
├── config/                    # 설정 파일
├── logs/                      # 로그 파일
└── pyproject.toml            # 프로젝트 설정
```

## 🛠️ 기술 스택

| 구성 요소 | 기술 |
|-----------|------|
| **CLI Framework** | Click 8.0+ |
| **Vector Database** | ChromaDB 1.1+ |
| **Embeddings** | sentence-transformers 5.1+ |
| **Language Models** | OpenAI GPT, Google Gemini, Ollama |
| **Document Processing** | PyPDF (PDF), python-docx (DOCX) |
| **Configuration** | YAML, Pydantic |
| **Package Management** | uv |
| **Testing** | pytest, pytest-cov |
| **Code Quality** | Black, isort, mypy |

## 🔧 개발자 가이드

### 개발 환경 설정

```bash
# 개발 의존성 설치
uv pip install -e ".[dev]"

# 코드 품질 도구 설정
pre-commit install

# 테스트 실행
pytest

# 코드 커버리지 확인
pytest --cov=src/document_rag_english_study --cov-report=html
```

### 코딩 표준

- **Python 3.9+** 지원
- **Type Hints** 필수
- **Google Style Docstrings**
- **Black** 코드 포맷팅
- **isort** import 정렬
- **mypy** 타입 체크
- **pytest** 테스트 프레임워크

### 테스트

```bash
# 전체 테스트 실행
pytest

# 특정 모듈 테스트
pytest tests/test_conversation_engine.py

# 커버리지 리포트 생성
pytest --cov=src --cov-report=html
open htmlcov/index.html
```

## 🚨 문제 해결

### 일반적인 문제

#### 설정 관련
```bash
# 설정 상태 확인
english-study status --detailed

# 설정 초기화
rm -rf ~/.config/document-rag-english-study
english-study setup
```

#### 문서 인덱싱 실패
```bash
# 지원되는 파일 형식 확인: PDF, DOCX, TXT, MD
# 파일 권한 확인
ls -la /path/to/documents

# 개별 파일 테스트
english-study set-docs /path/to/single-file-directory
```

#### LLM 연결 문제
```bash
# OpenAI API 키 확인
echo $OPENAI_API_KEY

# Ollama 서버 상태 확인
curl http://localhost:11434/api/tags

# 네트워크 연결 테스트
ping api.openai.com
```

### 로그 확인

```bash
# 로그 파일 위치
tail -f logs/document_rag_english_study.log
tail -f logs/document_rag_english_study_error.log
```

### 성능 최적화

- **문서 크기**: 10MB 이하 권장
- **청크 크기**: 기본값 1000 토큰
- **임베딩 모델**: 다국어 모델 사용 시 성능 고려
- **벡터 DB**: 정기적인 인덱스 최적화

## 🤝 기여하기

### 기여 방법

1. **Fork** 저장소
2. **Feature branch** 생성 (`git checkout -b feature/amazing-feature`)
3. **변경사항 커밋** (`git commit -m 'Add amazing feature'`)
4. **Branch에 Push** (`git push origin feature/amazing-feature`)
5. **Pull Request** 생성

### 개발 규칙

- 각 작업 단계 완료 후 반드시 commit
- uv를 사용한 패키지 관리
- 코드 품질 가이드라인 준수
- 테스트 코드 작성 필수 (90% 커버리지 목표)
- 모든 주석과 문서는 한글로 작성

### 버그 리포트

버그를 발견하셨나요? [Issues](https://github.com/your-username/document-rag-english-study/issues)에서 다음 정보와 함께 리포트해주세요:

- 운영체제 및 Python 버전
- 오류 메시지 및 로그
- 재현 단계
- 예상 동작 vs 실제 동작

## 📄 라이선스

이 프로젝트는 [MIT License](LICENSE)를 따릅니다.

## 🙏 감사의 말

- [ChromaDB](https://www.trychroma.com/) - 벡터 데이터베이스
- [sentence-transformers](https://www.sbert.net/) - 임베딩 모델
- [Click](https://click.palletsprojects.com/) - CLI 프레임워크
- [OpenAI](https://openai.com/), [Google](https://ai.google.dev/), [Ollama](https://ollama.ai/) - LLM 제공업체

---

📧 **문의사항**: [Issues](https://github.com/your-username/document-rag-english-study/issues)  
🌟 **도움이 되셨다면 Star를 눌러주세요!**