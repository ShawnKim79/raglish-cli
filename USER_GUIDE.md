# 사용자 가이드

Document RAG English Study의 상세 사용법과 활용 방법을 안내합니다.

## 📚 목차

1. [빠른 시작](#빠른-시작)
2. [기본 사용법](#기본-사용법)
3. [고급 기능](#고급-기능)
4. [학습 전략](#학습-전략)
5. [문제 해결](#문제-해결)
6. [팁과 요령](#팁과-요령)

## 🚀 빠른 시작

### 첫 번째 대화 시작하기

```bash
# 1. 초기 설정 (최초 1회)
english-study setup

# 2. 대화 시작
english-study chat
```

### 5분 만에 시작하기

1. **문서 준비**: 관심 있는 영어 문서들을 한 폴더에 모으기
2. **설정 실행**: `english-study setup` 명령어로 간단 설정
3. **대화 시작**: `english-study chat`으로 학습 시작

## 📖 기본 사용법

### 명령어 구조

```bash
english-study [COMMAND] [OPTIONS] [ARGUMENTS]
```

### 주요 명령어

#### 1. 설정 관련

```bash
# 통합 설정 가이드
english-study setup

# 개별 설정
english-study set-language ko                    # 모국어 설정
english-study set-docs ./documents              # 문서 디렉토리
english-study set-llm openai --api-key KEY     # LLM 설정

# 설정 확인
english-study status                            # 기본 상태
english-study status --detailed                # 상세 정보
english-study status --json                    # JSON 형식
```

#### 2. 학습 관련

```bash
# 기본 대화
english-study chat

# 주제 지정 대화
english-study chat --topic "artificial intelligence"

# 세션 재개
english-study chat --session-id abc123

# 세션 저장 안 함
english-study chat --no-save-session
```

#### 3. 도움말

```bash
# 전체 도움말
english-study help

# 특정 명령어 도움말
english-study help --command setup

# 사용 예제
english-study help --examples
```

### 설정 옵션 상세

#### 모국어 설정

```bash
# 기본 언어 설정
english-study set-language ko

# 학습 수준 포함 설정
english-study set-language ko \
  --learning-level advanced \
  --feedback-level detailed
```

**학습 수준 옵션:**
- `beginner`: 기초 문법과 어휘 중심
- `intermediate`: 실용적 표현과 교정 중심 (기본값)
- `advanced`: 고급 표현과 뉘앙스 중심

**피드백 수준 옵션:**
- `minimal`: 간단한 교정만
- `normal`: 적절한 교정과 설명 (기본값)
- `detailed`: 상세한 문법 설명과 다양한 표현

#### LLM 설정

```bash
# OpenAI GPT
english-study set-llm openai \
  --api-key sk-your-key \
  --model gpt-4 \
  --temperature 0.7 \
  --max-tokens 1500

# Google Gemini
english-study set-llm gemini \
  --api-key AIza-your-key \
  --model gemini-pro \
  --temperature 0.8

# Ollama (로컬)
english-study set-llm ollama \
  --model llama2 \
  --host localhost:11434 \
  --temperature 0.6
```

**매개변수 설명:**
- `temperature`: 응답의 창의성 (0.0-2.0, 낮을수록 일관성 높음)
- `max-tokens`: 최대 응답 길이
- `model`: 사용할 모델명
- `host`: Ollama 서버 주소

#### 문서 설정

```bash
# 기본 인덱싱
english-study set-docs ./my-documents

# 인덱싱 없이 디렉토리만 설정
english-study set-docs ./my-documents --no-index
```

## 🎯 고급 기능

### 세션 관리

#### 세션 ID 사용

```bash
# 특정 ID로 세션 시작
english-study chat --session-id "ai-study-session"

# 세션 재개
english-study chat --session-id "ai-study-session"
```

#### 주제별 대화

```bash
# 기술 주제
english-study chat --topic "machine learning"

# 비즈니스 주제
english-study chat --topic "business strategy"

# 일상 주제
english-study chat --topic "daily conversation"
```

### 문서 관리

#### 대용량 문서 처리

```bash
# 진행률 표시와 함께 인덱싱
english-study set-docs ./large-document-collection

# 인덱싱 상태 확인
english-study status --detailed
```

#### 문서 형식별 최적화

**PDF 문서:**
- 텍스트 기반 PDF 권장
- 스캔된 이미지 PDF는 OCR 필요
- 10MB 이하 권장

**DOCX 문서:**
- 표와 이미지는 텍스트로 변환
- 복잡한 서식은 단순화됨

**Markdown/텍스트:**
- 가장 빠른 처리 속도
- 구조화된 내용 권장

### 성능 최적화

#### 임베딩 캐시 활용

```bash
# 캐시 상태 확인
english-study status --detailed

# 캐시 클리어 (필요시)
rm -rf ~/.cache/document-rag-english-study
```

#### 메모리 사용량 관리

```bash
# 청크 크기 조정 (config/default.yaml)
document:
  chunk_size: 800        # 기본값: 1000
  chunk_overlap: 150     # 기본값: 200
```

## 📈 학습 전략

### 효과적인 문서 선택

#### 1. 관심사 기반 선택
```
documents/
├── technology/          # 기술 관심사
│   ├── ai-papers/
│   ├── programming/
│   └── tech-news/
├── business/           # 비즈니스 관심사
│   ├── strategy/
│   ├── marketing/
│   └── finance/
└── hobbies/           # 취미 관심사
    ├── photography/
    ├── cooking/
    └── travel/
```

#### 2. 난이도별 구성
- **초급**: 뉴스 기사, 블로그 포스트
- **중급**: 기술 문서, 비즈니스 리포트
- **고급**: 학술 논문, 전문 서적

#### 3. 분량 조절
- **시작**: 10-20개 문서 (총 100-200페이지)
- **확장**: 점진적으로 문서 추가
- **유지**: 정기적인 문서 업데이트

### 대화 학습 전략

#### 1. 단계별 학습

**1단계: 기본 이해**
```
User: What is machine learning?
Assistant: Machine learning is a subset of artificial intelligence...
[문법 교정 및 어휘 설명 제공]
```

**2단계: 심화 토론**
```
User: How does deep learning differ from traditional ML?
Assistant: Great question! Based on your documents...
[관련 문서 내용 활용한 상세 설명]
```

**3단계: 실용적 적용**
```
User: Can you help me explain this concept to my colleagues?
Assistant: Here's how you could present it professionally...
[비즈니스 영어 표현 학습]
```

#### 2. 주제별 학습 패턴

**기술 주제:**
- 정의 → 원리 → 응용 → 비교 → 전망
- 전문 용어 학습 중심

**비즈니스 주제:**
- 현황 → 분석 → 전략 → 실행 → 평가
- 프레젠테이션 영어 중심

**일상 주제:**
- 경험 → 의견 → 토론 → 결론
- 자연스러운 표현 중심

### 피드백 활용법

#### 1. 문법 교정
```
❌ "I am very interesting in AI"
✅ "I am very interested in AI"
💡 "interested" (형용사) vs "interesting" (형용사) 구분
```

#### 2. 어휘 향상
```
기본: "AI is good"
개선: "AI is beneficial/advantageous/promising"
고급: "AI demonstrates significant potential"
```

#### 3. 표현 다양화
```
단조로운 표현: "I think..."
다양한 표현: "In my opinion...", "From my perspective...", "I believe..."
```

## 🛠️ 문제 해결

### 일반적인 사용 문제

#### 1. 대화가 시작되지 않음

**증상**: `english-study chat` 실행 시 오류 발생

**해결책**:
```bash
# 설정 상태 확인
english-study status --detailed

# 누락된 설정 완료
english-study setup

# LLM 연결 테스트
english-study set-llm openai --api-key YOUR_KEY
```

#### 2. 문서 인덱싱 실패

**증상**: "문서 인덱싱 실패" 메시지

**해결책**:
```bash
# 파일 권한 확인
ls -la /path/to/documents

# 지원 형식 확인 (PDF, DOCX, TXT, MD)
file /path/to/documents/*

# 개별 파일 테스트
english-study set-docs /path/to/single-file
```

#### 3. 응답 속도 느림

**증상**: LLM 응답이 매우 느림

**해결책**:
```bash
# 로컬 모델 사용 (Ollama)
english-study set-llm ollama --model llama2

# 토큰 수 제한
english-study set-llm openai --max-tokens 500

# 문서 수 줄이기
english-study set-docs ./smaller-document-set
```

#### 4. 메모리 부족

**증상**: 시스템 메모리 부족으로 인한 오류

**해결책**:
```bash
# 청크 크기 줄이기 (config/default.yaml 수정)
document:
  chunk_size: 500
  chunk_overlap: 100

# 문서 수 제한
# 큰 문서들을 별도 폴더로 분리
```

### 설정 관련 문제

#### 1. API 키 오류

```bash
# OpenAI API 키 테스트
curl -H "Authorization: Bearer $OPENAI_API_KEY" \
     https://api.openai.com/v1/models

# 환경 변수 확인
echo $OPENAI_API_KEY

# 새 API 키로 재설정
english-study set-llm openai --api-key NEW_KEY
```

#### 2. Ollama 연결 실패

```bash
# 서버 상태 확인
curl http://localhost:11434/api/tags

# 서버 재시작
pkill ollama
ollama serve

# 모델 재다운로드
ollama pull llama2
```

### 성능 관련 문제

#### 1. 검색 정확도 낮음

**해결책**:
- 더 구체적인 질문하기
- 관련 키워드 포함하기
- 문서 품질 개선하기

#### 2. 응답 품질 낮음

**해결책**:
- 더 나은 LLM 모델 사용
- temperature 값 조정
- 더 많은 관련 문서 추가

## 💡 팁과 요령

### 효과적인 질문 방법

#### 1. 구체적인 질문
```
❌ "Tell me about AI"
✅ "What are the main differences between supervised and unsupervised learning?"
```

#### 2. 맥락 제공
```
❌ "How to improve?"
✅ "How can I improve my presentation skills for technical topics?"
```

#### 3. 단계적 접근
```
1. "What is neural network?"
2. "How does backpropagation work?"
3. "Can you give me an example of training a neural network?"
```

### 학습 효율 극대화

#### 1. 정기적인 학습
- 매일 15-30분 대화
- 주 3-4회 새로운 주제
- 월 1회 문서 업데이트

#### 2. 다양한 주제 활용
- 업무 관련 문서
- 취미 관련 자료
- 시사 이슈 기사

#### 3. 피드백 적극 활용
- 교정 내용 노트 정리
- 새로운 표현 연습
- 반복 학습으로 체화

### 고급 활용법

#### 1. 전문 분야 학습
```bash
# 의학 영어
english-study set-docs ./medical-papers
english-study chat --topic "medical terminology"

# 법률 영어
english-study set-docs ./legal-documents
english-study chat --topic "contract law"

# 기술 영어
english-study set-docs ./tech-specs
english-study chat --topic "software architecture"
```

#### 2. 프레젠테이션 연습
```bash
# 발표 준비
english-study chat --topic "presentation skills"

# 질의응답 연습
english-study chat --topic "Q&A session"
```

#### 3. 비즈니스 영어
```bash
# 회의 영어
english-study chat --topic "business meeting"

# 이메일 작성
english-study chat --topic "professional email"

# 협상 영어
english-study chat --topic "business negotiation"
```

### 문서 관리 팁

#### 1. 폴더 구조 최적화
```
documents/
├── current/           # 현재 학습 중인 문서
├── archive/          # 완료된 문서
├── reference/        # 참고 자료
└── new/             # 새로 추가할 문서
```

#### 2. 파일명 규칙
```
2024-01-15_AI_Research_Paper.pdf
2024-01-16_Business_Strategy_Guide.docx
2024-01-17_Programming_Tutorial.md
```

#### 3. 정기적인 정리
- 주 1회: 새 문서 추가
- 월 1회: 오래된 문서 정리
- 분기 1회: 전체 재인덱싱

## 📞 추가 지원

### 커뮤니티 리소스
- **GitHub Issues**: 버그 리포트 및 기능 요청
- **Discussions**: 사용법 질문 및 팁 공유
- **Wiki**: 추가 문서 및 가이드

### 학습 리소스
- **영어 학습 사이트**: 기본 문법 및 어휘 학습
- **전문 용어 사전**: 분야별 용어 정리
- **발음 가이드**: 정확한 발음 학습

### 정기 업데이트
- **기능 개선**: 새로운 LLM 모델 지원
- **성능 최적화**: 더 빠른 검색 및 응답
- **사용성 개선**: 더 직관적인 인터페이스

---

이 가이드가 도움이 되셨나요? 추가 질문이나 제안사항이 있으시면 [Issues](https://github.com/your-username/document-rag-english-study/issues)에 남겨주세요!