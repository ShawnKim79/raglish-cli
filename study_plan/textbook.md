# RAGlish 학습을 위한 기술 교과서

`study.md`의 학습 계획에 맞춰 각 단계별로 필요한 핵심 개념과 참고 자료를 정리한 문서입니다.

---

### 1단계: 프로젝트 큰 그림 이해

#### 핵심 개념

- **`pyproject.toml`**: 최신 Python 프로젝트의 표준 설정 파일입니다. PEP 518에 정의되었으며, 프로젝트의 의존성, 빌드 시스템, 메타데이터(이름, 버전 등)를 한 곳에서 관리합니다. 이 파일을 통해 `uv`나 `pip` 같은 도구가 어떤 라이브러리를 설치해야 하는지 알 수 있습니다.
- **`uv`**: Rust로 작성된 매우 빠른 Python 패키지 설치 및 관리 도구입니다. `pip`와 `venv`의 기능을 합친 것으로, 기존 도구들보다 월등히 빠른 속도로 의존성을 해결하고 가상환경을 관리해줍니다.
- **`YAML (YAML Ain't Markup Language)`**: 사람이 읽기 쉬운 데이터 직렬화 형식입니다. 주로 설정 파일에 사용되며, 들여쓰기를 통해 데이터의 계층 구조를 표현합니다. JSON과 유사하지만 주석을 사용할 수 있고 문법이 더 간결하여 가독성이 높습니다.

#### 추천 자료

- **Python Packaging User Guide**: `pyproject.toml`의 역할과 구조에 대한 공식 설명
  - [https://packaging.python.org/en/latest/guides/writing-pyproject-toml/](https://packaging.python.org/en/latest/guides/writing-pyproject-toml/)
- **`uv` 공식 문서**: `uv`의 설치 및 기본 사용법
  - [https://github.com/astral-sh/uv](https://github.com/astral-sh/uv)
- **YAML 공식 사이트**: YAML의 기본 문법과 예제
  - [https://yaml.org/](https://yaml.org/)

---

### 2단계: 데이터 모델 분석

#### 핵심 개념

- **`dataclasses`**: Python 3.7부터 도입된 기능으로, 데이터 저장을 주 목적으로 하는 클래스를 쉽게 만들 수 있도록 도와주는 데코레이터입니다. `@dataclass`를 클래스 위에 붙여주면 `__init__`, `__repr__`, `__eq__`와 같은 특별 메서드들을 자동으로 생성해주어, 보일러플레이트 코드를 획기적으로 줄여줍니다. 타입 힌트와 함께 사용되어 코드의 명확성을 높입니다.

#### 추천 자료

- **Python 공식 문서: `dataclasses`**: 데이터클래스의 모든 기능과 옵션에 대한 상세한 설명
  - [https://docs.python.org/3/library/dataclasses.html](https://docs.python.org/3/library/dataclasses.html)
- **Real Python: The Ultimate Guide to Data Classes in Python 3.7**: 친절한 설명과 다양한 예제
  - [https://realpython.com/python-data-classes/](https://realpython.com/python-data-classes/)

---

### 3단계: 데이터 수집 및 전처리 (Ingestion)

#### 핵심 개념

- **파싱 (Parsing)**: PDF, DOCX 등 다양한 형식의 파일에서 순수한 텍스트 정보만 추출하는 과정입니다. 각 파일 형식에 맞는 라이브러리(e.g., `pypdf` for PDF)를 사용하여 내용을 읽어 들입니다.
- **청킹 (Chunking)**: LLM이 한 번에 처리할 수 있는 컨텍스트 길이의 한계가 있기 때문에, 긴 문서를 의미 있는 작은 조각(Chunk)으로 나누는 과정입니다. 어떻게 나누느냐(예: 문단, 문장, 고정 크기)가 RAG의 성능에 매우 큰 영향을 미칩니다. 이 프로젝트에서는 문단과 문장 기반의 분할 전략을 사용합니다.

#### 추천 자료

- **Pinecone - Chunking Strategies**: 다양한 청킹 전략과 장단점에 대한 기술 블로그
  - [https://www.pinecone.io/learn/chunking-strategies/](https://www.pinecone.io/learn/chunking-strategies/)
- **LangChain - Text Splitters**: LangChain에서 제공하는 다양한 텍스트 분할기 종류와 예제
  - [https://python.langchain.com/v0.1/docs/modules/data_connection/document_transformers/](https://python.langchain.com/v0.1/docs/modules/data_connection/document_transformers/)

---

### 4단계: 임베딩 및 벡터 저장 (Embedding & Storage)

#### 핵심 개념

- **임베딩 (Embedding)**: 텍스트의 의미를 다차원 공간의 벡터(숫자 배열)로 변환하는 과정입니다. 의미적으로 유사한 텍스트는 벡터 공간에서 가까운 위치에 존재하게 됩니다.
- **Sentence-Transformers**: 문장이나 텍스트를 고품질의 임베딩 벡터로 변환하는 데 널리 사용되는 Python 라이브러리입니다.
- **벡터 데이터베이스 (Vector Database)**: 임베딩 벡터를 효율적으로 저장하고, 특정 벡터와 유사한 벡터를 빠르게 검색(유사도 검색)하는 데 특화된 데이터베이스입니다.
- **ChromaDB**: 사용하기 쉬운 오픈소스 벡터 데이터베이스로, 로컬 환경에서도 쉽게 RAG 시스템을 구축할 수 있도록 도와줍니다.
- **코사인 유사도 (Cosine Similarity)**: 두 벡터 사이의 각도의 코사인 값을 이용하여 유사도를 측정하는 방법입니다. 벡터의 크기가 아닌 방향에 초점을 맞추기 때문에, 문장의 길이에 상관없이 의미적 유사도를 잘 측정할 수 있습니다. RAG의 검색 단계에서 가장 널리 사용되는 유사도 척도입니다.

#### 추천 자료

- **Sentence-Transformers 공식 문서**: 라이브러리 사용법 및 사전 학습된 모델 목록
  - [https://www.sbert.net/](https://www.sbert.net/)
- **ChromaDB 공식 문서**: ChromaDB 시작 가이드
  - [https://docs.trychroma.com/getting-started](https://docs.trychroma.com/getting-started)
- **Towards Data Science - Cosine Similarity, Explained**: 코사인 유사도에 대한 시각적이고 직관적인 설명
  - [https://towardsdatascience.com/cosine-similarity-explained-and-visualized-d6afd0899f83](https://towardsdatascience.com/cosine-similarity-explained-and-visualized-d6afd0899f83)

---

### 5단계: 검색 및 생성 (Retrieval & Generation)

#### 핵심 개념

- **RAG 파이프라인**: "검색(Retrieve) -> 증강(Augment) -> 생성(Generate)"의 흐름을 의미합니다. 사용자 질문이 들어오면, 1) 벡터 DB에서 관련 문서를 **검색**하고, 2) 검색된 문서를 원본 질문에 컨텍스트로 추가하여 프롬프트를 **증강**시킨 후, 3) 증강된 프롬프트를 LLM에 보내 최종 답변을 **생성**합니다.
- **프롬프트 엔지니어링 (Prompt Engineering)**: 검색된 컨텍스트와 사용자 질문을 효과적으로 조합하여 LLM이 원하는 답변을 생성하도록 유도하는 프롬프트를 설계하는 기술입니다.
- **추상화 (Abstraction)**: `llm/base.py`의 `LanguageModel`처럼, 구체적인 구현(OpenAI, Gemini 등)의 공통적인 기능을 인터페이스로 정의하는 객체 지향 설계 원칙입니다. 이를 통해 코드의 다른 부분은 어떤 LLM이 사용되는지 신경 쓸 필요 없이 일관된 방식으로 모델을 사용할 수 있어, 새로운 LLM을 추가하기가 매우 쉬워집니다.

#### 추천 자료

- **Facebook AI - Retrieval Augmented Generation (RAG) 소개**: RAG의 개념을 처음 제안한 논문 요약 및 설명
  - [https://ai.meta.com/blog/retrieval-augmented-generation-streamlining-the-creation-of-intelligent-natural-language-processing-models/](https://ai.meta.com/blog/retrieval-augmented-generation-streamlining-the-creation-of-intelligent-natural-language-processing-models/)
- **OpenAI - Prompt engineering Best Practices**: 프롬프트 엔지니어링에 대한 공식 가이드
  - [https://platform.openai.com/docs/guides/prompt-engineering/strategy-write-clear-instructions](https://platform.openai.com/docs/guides/prompt-engineering/strategy-write-clear-instructions)

---

### 6단계: 애플리케이션 통합

#### 핵심 개념

- **CLI (Command-Line Interface)**: 사용자가 텍스트 기반의 명령어를 통해 소프트웨어와 상호작용하는 인터페이스입니다.
- **Click**: 복잡한 CLI 애플리케이션을 쉽고 보기 좋게 만들 수 있도록 도와주는 Python 라이브러리입니다. 데코레이터를 사용하여 명령어, 옵션, 인자 등을 직관적으로 정의할 수 있습니다.

#### 추천 자료

- **Click 공식 문서**: Click 라이브러리의 튜토리얼 및 전체 기능 설명
  - [https://click.palletsprojects.com/en/8.1.x/](https://click.palletsprojects.com/en/8.1.x/)
