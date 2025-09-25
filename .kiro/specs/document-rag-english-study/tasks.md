# Implementation Plan

- [x] 1. 프로젝트 구조 및 기본 설정 구성





  - 프로젝트 디렉토리 구조 생성 및 기본 패키지 초기화
  - requirements.txt 파일 작성 및 필요한 의존성 정의
  - 기본 설정 파일 구조 및 환경 변수 관리 설정
  - _Requirements: 5.1, 5.2_

- [x] 2. 핵심 데이터 모델 및 설정 관리 구현





  - [x] 2.1 기본 데이터 모델 클래스 구현



    - ConversationSession, Message, ConversationResponse 등 핵심 데이터 클래스 작성
    - 데이터 검증 및 직렬화 메서드 구현
    - _Requirements: 4.1, 4.2_

  - [x] 2.2 설정 관리 시스템 구현


    - ConfigurationManager 클래스 구현
    - YAML 기반 설정 파일 로드/저장 기능
    - 모국어, LLM 제공업체, 문서 디렉토리 설정 관리
    - 설정 완료 상태 확인 기능
    - _Requirements: 2.1, 2.3, 3.1, 3.3_

- [x] 3. 문서 처리 및 인덱싱 시스템 구현





  - [x] 3.1 문서 파서 구현


    - PDF, DOCX, TXT, MD 파일 파싱 기능 구현
    - 각 파일 형식별 텍스트 추출 메서드 작성
    - 파일 형식 검증 및 오류 처리
    - _Requirements: 1.2, 1.4_

  - [x] 3.2 문서 관리자 구현


    - DocumentManager 클래스 구현
    - 디렉토리 스캔 및 문서 일괄 인덱싱 기능
    - 인덱싱 상태 추적 및 진행률 표시
    - 문서 요약 정보 제공
    - _Requirements: 1.1, 1.3, 1.4_

- [x] 4. LLM 추상화 레이어 구현






  - [x] 4.1 언어 모델 추상 클래스 구현


    - LanguageModel 추상 베이스 클래스 정의
    - 공통 인터페이스 메서드 정의 (generate_response, translate_text, analyze_grammar)
    - _Requirements: 2.2, 4.3_

  - [x] 4.2 OpenAI GPT 구현체 작성


    - OpenAILanguageModel 클래스 구현
    - OpenAI API 연동 및 프롬프트 엔지니어링
    - API 키 관리 및 오류 처리
    - _Requirements: 2.2, 2.3, 2.4_

  - [x] 4.3 Google Gemini 구현체 작성




    - GeminiLanguageModel 클래스 구현
    - Google Gemini API 연동
    - API 응답 파싱 및 오류 처리
    - _Requirements: 2.2, 2.3, 2.4_

  - [x] 4.4 Ollama 로컬 모델 구현체 작성





    - OllamaLanguageModel 클래스 구현
    - 로컬 Ollama 서버 연동
    - 연결 상태 확인 및 모델 가용성 검증
    - _Requirements: 2.2, 2.3, 2.4_

- [x] 5. RAG 엔진 구현
  - [x] 5.1 임베딩 생성기 구현






    - EmbeddingGenerator 클래스 구현
    - sentence-transformers를 활용한 텍스트 임베딩 생성
    - 배치 처리 및 캐싱 기능
    - _Requirements: 1.2, 4.2_

  - [x] 5.2 벡터 데이터베이스 구현






    - VectorDatabase 클래스 구현
    - ChromaDB 연동 및 컬렉션 관리
    - 문서 청크 저장 및 유사도 검색 기능
    - _Requirements: 1.2, 4.2_

  - [x] 5.3 RAG 엔진 코어 구현





    - RAGEngine 클래스 구현
    - 문서 인덱싱 및 검색 기능
    - 컨텍스트 기반 답변 생성을 위한 관련 문서 검색
    - _Requirements: 1.2, 4.2_

- [x] 6. 대화형 학습 엔진 구현






  - [x] 6.1 학습 어시스턴트 구현





    - LearningAssistant 클래스 구현
    - 사용자 영어 분석 및 오류 식별 기능
    - 문법 교정 및 개선 제안 기능
    - 어휘 향상 제안 기능
    - _Requirements: 4.3, 4.4_

  - [x] 6.2 대화 관리자 구현






    - DialogManager 클래스 구현
    - 문서 주제 기반 대화 시작 기능
    - 대화 흐름 유지 및 자연스러운 전환
    - 후속 질문 제안 기능
    - _Requirements: 4.1, 4.2, 4.5_

  - [x] 6.3 세션 추적기 구현






    - SessionTracker 클래스 구현
    - 대화 세션 생성, 업데이트, 저장 기능
    - 학습 포인트 및 진행 상황 추적
    - 세션 요약 생성 기능
    - _Requirements: 4.1, 4.5_

  - [x] 6.4 대화 엔진 통합






    - ConversationEngine 클래스 구현
    - RAG 엔진과 LLM을 활용한 대화 처리
    - 사용자 입력 분석 및 학습 피드백 제공
    - 관심사 기반 대화 유도 및 유지
    - _Requirements: 4.1, 4.2, 4.3, 4.4, 4.5_

- [ ] 7. CLI 인터페이스 구현
  - [x] 7.1 기본 CLI 프레임워크 구성






    - Click 기반 CLI 애플리케이션 구조 설정
    - 명령어 그룹 및 옵션 정의
    - 기본 도움말 및 버전 정보 표시
    - _Requirements: 5.1, 5.2_

  - [-] 7.2 초기 설정 명령어 구현



    - setup 명령어 구현 (통합 설정 가이드)
    - set-docs, set-llm, set-language 개별 설정 명령어
    - 설정 상태 확인 및 검증 기능
    - _Requirements: 1.1, 2.1, 3.1_

  - [ ] 7.3 대화형 학습 명령어 구현
    - chat 명령어 구현
    - 실시간 대화 인터페이스 구현
    - 학습 피드백 표시 및 상호작용
    - 대화 세션 저장 및 종료 처리
    - _Requirements: 4.1, 4.2, 4.3, 4.4, 4.5_

  - [ ] 7.4 상태 확인 및 도움말 명령어 구현
    - status 명령어 구현 (설정 및 인덱싱 상태)
    - help 명령어 구현
    - 상세 사용법 안내 및 예제 제공
    - _Requirements: 5.2, 5.3_

- [ ] 8. 오류 처리 및 로깅 시스템 구현
  - 전역 오류 처리기 구현
  - 사용자 친화적 오류 메시지 생성
  - 로깅 시스템 설정 및 오류 추적
  - _Requirements: 1.4, 2.4, 5.3_

- [ ] 9. 테스트 코드 작성
  - [ ] 9.1 단위 테스트 작성
    - 각 모듈별 단위 테스트 구현
    - Mock 객체를 활용한 의존성 격리 테스트
    - 테스트 커버리지 90% 이상 달성
    - _Requirements: 모든 요구사항_

  - [ ] 9.2 통합 테스트 작성
    - RAG 엔진 통합 테스트
    - 대화형 학습 파이프라인 테스트
    - CLI 명령어 통합 테스트
    - _Requirements: 모든 요구사항_

- [ ] 10. 문서화 및 배포 준비
  - README.md 작성 및 사용법 안내
  - 설치 가이드 및 설정 방법 문서화
  - 예제 시나리오 및 샘플 문서 제공
  - _Requirements: 5.2_

- [ ] 11. 성능 최적화 및 마무리
  - 벡터 검색 성능 최적화
  - 실시간 대화 응답 시간 개선
  - 대용량 문서 인덱싱 성능 개선
  - 최종 통합 테스트 및 버그 수정
  - _Requirements: 모든 요구사항_