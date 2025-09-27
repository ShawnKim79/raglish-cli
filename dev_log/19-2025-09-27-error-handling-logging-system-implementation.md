# 오류 처리 및 로깅 시스템 구현

**날짜**: 2025-09-27  
**작업자**: Kiro AI Assistant  
**태스크**: 8. 오류 처리 및 로깅 시스템 구현  
**상태**: ✅ 완료

## 개요

Document RAG English Study 프로젝트에 포괄적인 오류 처리 및 로깅 시스템을 구현했습니다. 이 시스템은 사용자 친화적인 오류 메시지 제공, 상세한 로깅, 자동 재시도 기능 등을 포함합니다.

## 구현된 주요 컴포넌트

### 1. 사용자 정의 예외 클래스 (`exceptions.py`)

```python
# 기본 예외 클래스
class DocumentRAGError(Exception)

# 특화된 예외 클래스들
class DocumentError(DocumentRAGError)      # 문서 처리 관련
class RAGError(DocumentRAGError)           # RAG 엔진 관련
class LearningError(DocumentRAGError)      # 학습 모듈 관련
class ConfigurationError(DocumentRAGError) # 설정 관련
class ValidationError(DocumentRAGError)    # 입력 검증 관련
class LLMError(DocumentRAGError)          # LLM 관련
class VectorDatabaseError(RAGError)       # 벡터 DB 관련
class EmbeddingError(RAGError)            # 임베딩 관련
```

**특징**:
- 각 예외는 컨텍스트 정보와 오류 코드를 포함
- 계층적 구조로 설계되어 유연한 예외 처리 가능
- 추가 메타데이터 (파일 경로, 작업명 등) 지원

### 2. 전역 오류 처리기 (`error_handler.py`)

```python
class ErrorHandler:
    def handle_error(self, error: Exception, context: Dict[str, Any]) -> None
    def get_user_friendly_message(self, error: Exception) -> str
    def should_retry(self, error: Exception) -> bool
    def is_critical_error(self, error: Exception) -> bool
```

**주요 기능**:
- **사용자 친화적 메시지 생성**: 기술적 오류를 이해하기 쉬운 한국어 메시지로 변환
- **재시도 전략**: 오류 타입별 자동 재시도 횟수 설정
- **치명적 오류 판단**: SystemExit, KeyboardInterrupt 등 시스템 레벨 오류 구분
- **상세 로깅**: 개발자를 위한 디버깅 정보 기록

### 3. 로깅 설정 시스템 (`logging_config.py`)

```python
class LoggingConfig:
    def setup_logging(self, level: str, console_output: bool, file_output: bool)
    def get_logger(self, name: str) -> logging.Logger
    def log_system_info(self) -> None
    def cleanup_old_logs(self, days_to_keep: int) -> None
```

**로깅 구조**:
- **메인 로그**: `document_rag_english_study.log` (모든 레벨)
- **에러 로그**: `document_rag_english_study_error.log` (ERROR 이상)
- **디버그 로그**: `document_rag_english_study_debug.log` (DEBUG 레벨일 때)

**기능**:
- 로그 파일 회전 (최대 10MB, 5개 백업)
- 콘솔 및 파일 출력 동시 지원
- 시스템 정보 자동 로깅
- 오래된 로그 파일 자동 정리

### 4. 데코레이터 기반 오류 처리

#### `@error_handler_decorator`
```python
@error_handler_decorator(context={"operation": "document_processing"})
def process_document(file_path: str):
    # 함수 실행 중 오류 발생 시 자동 처리
    pass
```

#### `@retry_on_error`
```python
@retry_on_error(max_retries=3, delay=1.0)
def connect_to_llm():
    # 연결 실패 시 자동 재시도
    pass
```

## CLI 통합

### 메인 애플리케이션 (`cli/main.py`)
- 애플리케이션 시작 시 로깅 시스템 자동 초기화
- 환경 변수 `LOG_LEVEL`로 로그 레벨 제어
- 예외 타입별 차별화된 종료 코드 (0: 정상, 1: 일반 오류, 2: 치명적 오류)

### CLI 명령어 (`cli/interface.py`)
모든 주요 CLI 명령어에 오류 처리 데코레이터 적용:
- `setup`: 초기 설정
- `set-docs`: 문서 디렉토리 설정
- `set-llm`: LLM 설정
- `set-language`: 언어 설정
- `chat`: 대화형 학습
- `status`: 상태 확인

## 기존 모듈 통합

### DocumentManager 업데이트
- `DocumentManagerError` → `DocumentError`로 통합
- 주요 메서드에 오류 처리 데코레이터 적용
- 재시도 기능 추가 (`@retry_on_error`)

## 테스트 구현

### 테스트 커버리지
총 **36개 테스트 케이스** 작성 및 모두 통과:

1. **예외 클래스 테스트** (10개)
   - 각 예외 클래스의 초기화 및 컨텍스트 정보 테스트

2. **ErrorHandler 테스트** (9개)
   - 사용자 친화적 메시지 생성
   - 재시도 가능 여부 판단
   - 치명적 오류 판단
   - 오류 로깅 기능

3. **LoggingConfig 테스트** (5개)
   - 로깅 시스템 초기화
   - 로그 파일 생성
   - 시스템 정보 로깅
   - 오래된 로그 정리

4. **데코레이터 테스트** (7개)
   - 오류 처리 데코레이터 동작
   - 재시도 데코레이터 동작
   - 치명적 오류 처리

5. **전역 함수 테스트** (5개)
   - 싱글톤 패턴 확인
   - 전역 함수 동작 검증

## 사용자 경험 개선

### 오류 메시지 예시

**기술적 오류**:
```
FileNotFoundError: [Errno 2] No such file or directory: '/path/to/file.txt'
```

**사용자 친화적 메시지**:
```
파일을 찾을 수 없습니다. 파일 경로를 확인해주세요. (파일: /path/to/file.txt)
```

### 로그 출력 예시

**콘솔 출력** (간단한 형식):
```
19:28:17 - INFO - 애플리케이션 시작
19:28:17 - WARNING - 문서 처리 중 오류 발생
```

**파일 로그** (상세한 형식):
```
2025-09-27 19:28:17 - src.document_rag_english_study.cli.main - INFO - main.py:25 - Document RAG English Study 애플리케이션 시작
2025-09-27 19:28:17 - src.document_rag_english_study.utils.error_handler - WARNING - error_handler.py:105 - 오류 발생: DocumentError: 문서 처리 실패
```

## 요구사항 충족

### Requirements 1.4
✅ **디렉토리가 존재하지 않거나 지원되는 파일이 없으면 적절한 안내 메시지 표시**
- `DocumentError`를 통한 명확한 오류 메시지
- 파일 경로 정보 포함

### Requirements 2.4
✅ **LLM 연결 실패 시 오류 원인과 해결 방법 안내**
- `LLMError`를 통한 구체적인 오류 정보
- API 키, 네트워크 연결 상태 안내

### Requirements 5.3
✅ **잘못된 명령어 입력 시 오류 메시지와 올바른 사용법 안내**
- CLI 레벨에서 자동 처리
- 사용자 친화적 메시지 제공

## 기술적 특징

### 1. 계층적 예외 구조
- 기본 `DocumentRAGError`에서 파생된 특화 예외들
- 상속 관계를 활용한 유연한 예외 처리

### 2. 컨텍스트 기반 오류 처리
- 오류 발생 시점의 상황 정보 자동 수집
- 디버깅을 위한 풍부한 메타데이터 제공

### 3. 재시도 전략
- 오류 타입별 차별화된 재시도 정책
- 네트워크 오류, LLM 연결 오류 등에 대한 자동 재시도

### 4. 로그 레벨 분리
- 사용자용 콘솔 출력 (간단)
- 개발자용 파일 로그 (상세)
- 에러 전용 로그 파일

## 성능 및 안정성

### 로그 파일 관리
- 파일 크기 제한 (10MB)
- 자동 회전 (5개 백업 파일)
- 오래된 로그 자동 정리 (30일)

### 메모리 효율성
- 싱글톤 패턴으로 인스턴스 관리
- 지연 초기화 (lazy initialization)

### 스레드 안전성
- 로깅 시스템의 스레드 안전 보장
- 동시 접근 시 안전한 로그 기록

## 향후 확장 가능성

### 1. 알림 시스템
- 치명적 오류 발생 시 이메일/슬랙 알림
- 오류 발생 빈도 모니터링

### 2. 오류 분석
- 오류 패턴 분석 및 리포팅
- 사용자 행동 기반 오류 예측

### 3. 다국어 지원
- 오류 메시지의 다국어 번역
- 사용자 언어 설정에 따른 메시지 제공

## 커밋 히스토리

1. **feat: 오류 처리 및 로깅 시스템 구현** (457ee16)
   - 사용자 정의 예외 클래스들 추가
   - 전역 오류 처리기 구현
   - 포괄적인 로깅 설정 시스템 구현
   - 36개 테스트 케이스 작성

2. **fix: DocumentManagerError를 DocumentError로 교체** (411855e)
   - 일관된 예외 처리 시스템 적용
   - 기존 테스트 코드 업데이트

## 검증 결과

### 테스트 실행 결과
```bash
uv run pytest tests/test_error_handling.py -v
======================================================================= test session starts ========================================================================
collected 36 items

tests/test_error_handling.py::TestExceptions::test_document_rag_error_basic PASSED                    [  2%]
tests/test_error_handling.py::TestExceptions::test_document_rag_error_with_context PASSED             [  5%]
# ... (중략) ...
tests/test_error_handling.py::TestGlobalFunctions::test_get_logger_function PASSED                   [100%]

======================================================================== 36 passed in 0.40s ========================================================================
```

### CLI 동작 확인
```bash
uv run python -m src.document_rag_english_study.cli.main --help
19:28:17 - INFO - 로깅 시스템이 설정되었습니다. 레벨: INFO, 콘솔: True, 파일: True
19:28:17 - INFO - Document RAG English Study 애플리케이션 시작
Usage: python -m src.document_rag_english_study.cli.main [OPTIONS] COMMAND [ARGS]...
```

### 로그 파일 생성 확인
- `logs/document_rag_english_study.log` ✅
- `logs/document_rag_english_study_error.log` ✅

## 결론

오류 처리 및 로깅 시스템 구현을 통해 다음과 같은 개선사항을 달성했습니다:

1. **사용자 경험 향상**: 기술적 오류를 이해하기 쉬운 메시지로 변환
2. **개발자 생산성 향상**: 상세한 로깅을 통한 효율적인 디버깅
3. **시스템 안정성 향상**: 자동 재시도 및 치명적 오류 처리
4. **유지보수성 향상**: 일관된 예외 처리 구조 및 포괄적인 테스트

이제 Document RAG English Study 시스템은 견고한 오류 처리 기반 위에서 안정적으로 동작할 수 있습니다.