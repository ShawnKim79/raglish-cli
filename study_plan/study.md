# "RAGlish" 소스 코드를 활용한 RAG 구현 학습 계획

## 1. 학습 목표

이 프로젝트의 소스 코드를 분석하여, 현대적인 RAG(Retrieval-Augmented Generation) 애플리케이션의 전체 구조와 각 구성 요소의 역할, 그리고 데이터 흐름을 이해합니다. 최종적으로는 RAG 시스템을 독립적으로 설계하고 구현할 수 있는 역량을 갖추는 것을 목표로 합니다.

## 2. 사전 요구사항

- Python 기본 문법 및 객체 지향 프로그래밍(OOP)에 대한 이해
- 가상환경 및 패키지 관리(`uv`, `pip`) 사용 경험
- LLM, 임베딩, 벡터 데이터베이스에 대한 기본적인 개념 이해

## 3. 단계별 학습 계획

RAG의 핵심 파이프라인과 프로젝트의 모듈 구조에 따라 아래 순서로 학습하는 것을 권장합니다.

### 1단계: 프로젝트 큰 그림 이해 (High-Level Overview)

- **목표**: 프로젝트의 전체 구조, 설정, 의존성을 파악합니다.
- **학습 자료**:
  - `README.md`: 프로젝트의 목적과 기본 사용법을 숙지합니다.
  - `pyproject.toml`: 어떤 라이브러리(`click`, `chromadb`, `sentence-transformers` 등)가 사용되었는지 확인합니다.
  - `config/default.yaml`: 시스템의 설정 가능한 항목들을 통해 전체 기능을 유추해봅니다.
  - `dev_log/`: 모든 개발 로그를 가볍게 훑어보며 프로젝트의 발전 과정을 이해합니다.

### 2단계: 데이터 모델 분석 (Data Modeling)

- **목표**: 시스템 내에서 정보가 어떤 형태로 흐르는지 이해합니다.
- **학습 자료**: `src/document_rag_english_study/models/`
  - `document.py`: `Document`, `IndexingResult` 등 문서 처리 관련 데이터 구조를 분석합니다.
  - `response.py`: `SearchResult`, `ConversationResponse` 등 RAG 결과물 데이터 구조를 분석합니다.
  - `llm.py`: `LLMResponse` 등 LLM과의 인터페이스 데이터 구조를 분석합니다.
- **학습 포인트**: 각 데이터 클래스가 시스템의 어느 부분에서 생성되고 소비되는지 생각해보세요.

### 3단계: 데이터 수집 및 전처리 (Ingestion Pipeline)

- **목표**: 문서가 어떻게 시스템에 입력되고 처리되는지 학습합니다.
- **학습 자료**:
  - `src/document_rag_english_study/document_manager/parser.py`: 다양한 파일(`PDF`, `MD` 등)에서 텍스트를 추출하는 방법을 분석합니다.
  - `src/document_rag_english_study/rag/engine.py`의 `_split_document_into_chunks`: 추출된 텍스트를 의미 있는 단위(Chunk)로 분할하는 전략을 학습합니다. (이 프로젝트의 핵심 중 하나)

### 4단계: 임베딩 및 벡터 저장 (Embedding & Storage)

- **목표**: 처리된 텍스트 청크가 어떻게 벡터로 변환되고 데이터베이스에 저장되는지 이해합니다.
- **학습 자료**:
  - `src/document_rag_english_study/rag/embeddings.py`: `EmbeddingGenerator`가 `sentence-transformers` 라이브러리를 사용해 텍스트를 벡터로 변환하는 과정을 분석합니다.
  - `src/document_rag_english_study/rag/vector_database.py`: `VectorDatabase`가 `ChromaDB`를 활용하여 벡터와 메타데이터를 저장하는 방법을 학습합니다.

### 5단계: 검색 및 생성 (Retrieval & Generation)

- **목표**: RAG의 핵심, 즉 사용자 질문에 가장 관련성 높은 정보를 찾아 LLM에게 전달하고 답변을 생성하는 과정을 학습합니다.
- **학습 자료**:
  - `src/document_rag_english_study/rag/engine.py`:
    - `search_similar_content`: 쿼리(질문)가 들어왔을 때 `VectorDatabase`에서 유사한 청크를 검색(Retrieve)하는 과정을 분석합니다.
    - `generate_answer`: 검색된 청크를 컨텍스트로 삼아, LLM 프롬프트를 구성하고 최종 답변을 생성(Generate)하는 과정을 분석합니다.
  - `src/document_rag_english_study/llm/base.py`: `LanguageModel` 추상화를 통해 어떻게 다양한 LLM을 지원하는지 구조를 파악합니다.

### 6단계: 애플리케이션 통합 (Application Layer)

- **목표**: 완성된 RAG 엔진이 실제 애플리케이션에서 어떻게 활용되는지 학습합니다.
- **학습 자료**:
  - `src/document_rag_english_study/conversation/engine.py`: RAG 엔진을 활용하여 대화의 흐름을 관리하고 학습 피드백을 제공하는 방법을 분석합니다.
  - `src/document_rag_english_study/cli/interface.py`: `click` 라이브러리를 통해 최종 사용자에게 RAG 기능을 제공하는 CLI 인터페이스 구현을 학습합니다.

## 4. 학습 방법 제안

- **테스트 코드 활용**: `tests/` 디렉토리의 테스트 코드는 각 모듈의 기능과 사용법을 보여주는 최고의 예제입니다. 소스 코드 분석과 함께 반드시 해당 테스트 코드를 읽고 실행해보세요.
- **개발 로그 참고**: 코드만으로 이해하기 어려운 설계 의도나 결정 배경은 `dev_log/` 디렉토리의 해당 기능 개발 로그를 참고하면 큰 도움이 됩니다.
- **수정 및 실험**: 코드 분석 후, 직접 기능을 수정하거나 새로운 기능을 추가해보세요. 예를 들어, `.txt` 파일 외에 새로운 파일 형식을 지원하도록 `parser.py`를 수정해보는 것은 좋은 실습이 될 것입니다.

이 계획을 따라 꾸준히 학습하신다면, RAG 시스템의 이론과 실제 구현 모두에 대한 깊은 이해를 얻으실 수 있을 것입니다.