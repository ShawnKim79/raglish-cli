# Document RAG English Study - Project Rules

## Project Overview
이 프로젝트는 사용자의 관심사 문서를 기반으로 RAG 시스템을 구축하고, 자연스러운 대화를 통해 영어 학습을 지원하는 CLI 프로그램입니다.

## Development Standards

### Code Style
- **Language**: Python 3.8+
- **Code Formatting**: Black formatter 사용
- **Import Sorting**: isort 사용
- **Linting**: flake8 또는 pylint 사용
- **Type Hints**: 모든 함수와 메서드에 타입 힌트 필수

### Project Structure
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
├── requirements.txt   # 의존성
└── setup.py          # 패키지 설정
```

### Naming Conventions
- **Classes**: PascalCase (예: `DocumentManager`, `RAGEngine`)
- **Functions/Methods**: snake_case (예: `process_document`, `generate_response`)
- **Variables**: snake_case (예: `user_input`, `conversation_session`)
- **Constants**: UPPER_SNAKE_CASE (예: `DEFAULT_LANGUAGE`, `MAX_CHUNK_SIZE`)
- **Files/Modules**: snake_case (예: `document_manager.py`, `rag_engine.py`)

### Error Handling
- 모든 외부 API 호출에 대해 적절한 예외 처리 구현
- 사용자 친화적인 오류 메시지 제공
- 로깅을 통한 디버깅 정보 기록
- 복구 가능한 오류에 대한 재시도 로직 구현

### Testing Requirements
- **Unit Tests**: 각 클래스와 함수에 대한 단위 테스트 필수
- **Integration Tests**: 주요 워크플로우에 대한 통합 테스트
- **Test Coverage**: 최소 90% 코드 커버리지 목표
- **Test Framework**: pytest 사용
- **Mocking**: unittest.mock 또는 pytest-mock 사용

### Documentation
- **Docstrings**: 모든 클래스, 함수, 메서드에 Google 스타일 docstring 작성
- **README**: 설치, 설정, 사용법에 대한 명확한 가이드
- **API Documentation**: 주요 클래스와 메서드에 대한 API 문서
- **Examples**: 실제 사용 예제 코드 제공

### Dependencies Management
- **Core Dependencies**: 최소한의 필수 라이브러리만 사용
- **Version Pinning**: requirements.txt에서 버전 고정
- **Virtual Environment**: 개발 시 가상환경 사용 필수
- **Security**: 알려진 보안 취약점이 있는 패키지 사용 금지

### Configuration Management
- **Environment Variables**: 민감한 정보는 환경 변수로 관리
- **Config Files**: YAML 형식의 설정 파일 사용
- **Default Values**: 모든 설정에 대한 합리적인 기본값 제공
- **Validation**: 설정 값에 대한 검증 로직 구현

### Performance Guidelines
- **Lazy Loading**: 필요할 때만 리소스 로드
- **Caching**: 반복적인 계산 결과 캐싱
- **Batch Processing**: 대량 데이터 처리 시 배치 단위 처리
- **Memory Management**: 대용량 파일 처리 시 스트리밍 방식 사용

### Security Best Practices
- **Input Validation**: 모든 사용자 입력에 대한 검증
- **API Key Security**: API 키는 환경 변수로만 관리
- **File Access**: 사용자 지정 경로 접근 시 보안 검증
- **Data Sanitization**: 외부 데이터 처리 시 새니타이징

### Git Workflow
- **Branch Naming**: feature/task-name, bugfix/issue-name 형식
- **Commit Messages**: 명확하고 설명적인 커밋 메시지
- **Pull Requests**: 코드 리뷰를 위한 PR 필수
- **Pre-commit Hooks**: 코드 품질 검사 자동화

### CLI Design Principles
- **User-Friendly**: 직관적이고 이해하기 쉬운 명령어
- **Consistent**: 일관된 명령어 구조와 옵션
- **Helpful**: 상세한 도움말 및 오류 메시지
- **Progressive**: 단계별 설정 가이드 제공

### RAG Implementation Guidelines
- **Chunk Size**: 문서 청킹 시 적절한 크기 유지 (500-1000 토큰)
- **Embedding Model**: 다국어 지원 임베딩 모델 사용
- **Vector Search**: 효율적인 유사도 검색 알고리즘
- **Context Management**: 관련성 높은 컨텍스트 선별

### Learning Engine Guidelines
- **Natural Conversation**: 자연스러운 대화 흐름 유지
- **Adaptive Feedback**: 사용자 수준에 맞는 피드백 제공
- **Progress Tracking**: 학습 진행 상황 추적 및 분석
- **Multilingual Support**: 다양한 모국어 지원