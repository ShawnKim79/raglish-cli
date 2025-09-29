# RAGlish 프로젝트 학습 목표 체크리스트

`study.md`의 학습 계획에 따른 각 단계별 구체적인 학습 목표입니다.
각 목표를 달성한 후 체크하여 진행 상황을 관리하세요.

---

### 1단계: 프로젝트 큰 그림 이해

- [ ] 이 프로젝트의 목적과 핵심 기능을 다른 사람에게 1분 내로 설명할 수 있다. (관련 파일: `README.md`)
- [ ] `pyproject.toml`을 보고 이 프로젝트의 주요 의존성 3가지(웹 프레임워크 제외)와 그 역할을 말할 수 있다. (관련 파일: `pyproject.toml`)
- [ ] `config.yaml` 파일을 수정하여 LLM 모델을 `gpt-3.5-turbo`에서 `gpt-4`로 변경할 수 있다. (관련 파일: `config.yaml`)
- [ ] `dev_log`에서 가장 흥미로웠던 문제 해결 과정 하나를 선택하고, 그 원인과 해결책을 요약할 수 있다. (관련 디렉토리: `dev_log/`)

### 2단계: 데이터 모델 분석

- [ ] `Document`와 `SearchResult` 데이터 클래스의 차이점을 설명할 수 있다. (관련 파일: `src/document_rag_english_study/models/document.py`, `src/document_rag_english_study/models/response.py`)
- [ ] `ConversationSession`이 `Message`와 `LearningPoint`를 어떻게 포함하여 대화의 전체 맥락을 저장하는지 설명할 수 있다. (관련 파일: `src/document_rag_english_study/models/conversation.py`)
- [ ] `LLMResponse` 모델이 토큰 사용량(`usage`)과 같은 메타데이터를 왜 별도로 관리하는지 설명할 수 있다. (관련 파일: `src/document_rag_english_study/models/llm.py`)

### 3단계: 데이터 수집 및 전처리 (Ingestion)

- [ ] `DocumentParser`가 새로운 파일 형식(예: `.html`)을 지원하도록 코드를 수정할 수 있다. (관련 파일: `src/document_rag_english_study/document_manager/parser.py`)
- [ ] `RAGEngine`의 `_split_document_into_chunks` 메서드가 텍스트를 문단과 문장 단위로 분할하는 이유를 설명할 수 있다. (관련 파일: `src/document_rag_english_study/rag/engine.py`)
- [ ] 텍스트 청크(Chunk)의 크기와 겹침(Overlap) 설정이 RAG 성능에 어떤 영향을 미치는지 설명할 수 있다. (관련 파일: `src/document_rag_english_study/rag/engine.py`의 생성자)

### 4단계: 임베딩 및 벡터 저장 (Embedding & Storage)

- [ ] `EmbeddingGenerator`의 역할이 무엇이며, `sentence-transformers` 라이브러리가 왜 사용되었는지 설명할 수 있다. (관련 파일: `src/document_rag_english_study/rag/embeddings.py`)
- [ ] `VectorDatabase` 클래스가 텍스트가 아닌 벡터(Embedding)를 저장하는 이유를 설명할 수 있다. (관련 파일: `src/document_rag_english_study/rag/vector_database.py`)
- [ ] 코사인 유사도(Cosine Similarity)가 벡터 검색에서 어떤 의미를 갖는지 설명할 수 있다. (관련 파일: `src/document_rag_english_study/rag/vector_database.py`)

### 5단계: 검색 및 생성 (Retrieval & Generation)

- [ ] `RAGEngine`의 `search_similar_content` 메서드가 어떻게 사용자 질문과 가장 관련 높은 문서를 찾아내는지 그 과정을 설명할 수 있다. (관련 파일: `src/document_rag_english_study/rag/engine.py`)
- [ ] `RAGEngine`의 `generate_answer` 메서드에서, 검색된 문서(Context)와 사용자 질문(Query)을 어떻게 조합하여 LLM에게 프롬프트를 전달하는지 설명할 수 있다. (관련 파일: `src/document_rag_english_study/rag/engine.py`)
- [ ] `llm/base.py`의 `LanguageModel` 추상 클래스가 왜 필요한지, 이것이 시스템의 확장성에 어떤 이점을 주는지 설명할 수 있다. (관련 파일: `src/document_rag_english_study/llm/base.py`)

### 6단계: 애플리케이션 통합

- [ ] `cli/interface.py`의 `chat` 명령어가 실행되었을 때, `ConversationEngine`을 거쳐 `RAGEngine`의 `generate_answer`가 호출되기까지의 전체적인 함수 호출 흐름을 그릴 수 있다. (관련 파일: `src/document_rag_english_study/cli/interface.py`, `src/document_rag_english_study/conversation/engine.py`)
- [ ] `ConversationEngine`이 `DialogManager`와 `LearningAssistant`를 어떻게 사용하여 단순한 RAG 답변을 넘어 '영어 학습 대화'를 만들어내는지 설명할 수 있다. (관련 파일: `src/document_rag_english_study/conversation/engine.py`)
- [ ] CLI에 새로운 명령어(예: 인덱싱된 문서의 총 개수를 보여주는 `count-docs`)를 추가할 수 있다. (관련 파일: `src/document_rag_english_study/cli/interface.py`)
