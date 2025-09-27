# 사용 예제 및 시나리오

Document RAG English Study의 다양한 활용 시나리오와 실제 사용 예제를 소개합니다.

## 📚 목차

1. [기본 사용 예제](#기본-사용-예제)
2. [학습 시나리오](#학습-시나리오)
3. [전문 분야별 활용](#전문-분야별-활용)
4. [대화 예제](#대화-예제)
5. [설정 예제](#설정-예제)

## 🚀 기본 사용 예제

### 예제 1: 첫 번째 설정 및 학습

```bash
# 1. 프로그램 설치 확인
english-study --version

# 2. 초기 설정
english-study setup
# → 모국어: 한국어 선택
# → 문서 디렉토리: ~/Documents/english-learning 설정
# → LLM: OpenAI GPT 선택 및 API 키 입력

# 3. 상태 확인
english-study status
# ✅ 모든 설정이 완료되었습니다!

# 4. 첫 번째 대화
english-study chat
```

**대화 예제:**
```
🎓 Document RAG English Study에 오신 것을 환영합니다!
문서를 기반으로 흥미로운 주제에 대해 영어로 대화해보세요.

💬 Assistant: Hello! I see you have some interesting documents about technology and AI. 
What aspect of artificial intelligence interests you the most?

👤 User: I'm very interesting in machine learning algorithms.

💡 Assistant: Great topic! I noticed a small correction: you should say "I'm very *interested* in" 
rather than "interesting in." 

"Interested" describes your feeling, while "interesting" describes the topic itself.
- ✅ "I'm very interested in machine learning" (you feel interest)
- ✅ "Machine learning is very interesting" (the topic creates interest)

Now, based on your documents, I can see you have materials about neural networks and 
deep learning. What specific algorithms would you like to discuss?

👤 User: Can you explain how neural networks work in simple terms?

💬 Assistant: Absolutely! Think of a neural network like a simplified version of how 
our brain processes information...
[상세한 설명과 함께 문서 내용 활용]
```

### 예제 2: 주제별 대화 시작

```bash
# 특정 주제로 대화 시작
english-study chat --topic "business strategy"

# 세션 ID를 지정하여 체계적 학습
english-study chat --session-id "business-english-week1"
```

## 🎯 학습 시나리오

### 시나리오 1: IT 전문가의 영어 실력 향상

**배경**: 소프트웨어 개발자가 해외 컨퍼런스 발표를 준비

**문서 준비:**
```
tech-documents/
├── programming/
│   ├── clean-code-principles.pdf
│   ├── software-architecture-patterns.md
│   └── agile-development-guide.docx
├── conferences/
│   ├── tech-talk-examples.txt
│   └── presentation-tips.md
└── industry-trends/
    ├── ai-in-software-development.pdf
    └── future-of-programming.md
```

**학습 과정:**

1. **초기 설정**
```bash
english-study set-docs ./tech-documents
english-study set-language ko --learning-level intermediate
english-study set-llm openai --model gpt-4
```

2. **1주차: 기술 용어 학습**
```bash
english-study chat --topic "software architecture" --session-id "week1-architecture"
```

**대화 예제:**
```
👤 User: I want to talk about microservices architecture.

💬 Assistant: Excellent choice! Microservices is a hot topic in software development. 
Based on your architecture documents, I can see you have good materials on this.

Let me ask: How would you explain the main benefits of microservices to a non-technical stakeholder?

👤 User: Microservices make system more flexible and easy to maintain.

💡 Assistant: Good start! Let me help you make this more professional:

✅ "Microservices make *the* system more flexible and *easier* to maintain."
✅ "Microservices architecture provides greater flexibility and maintainability."

For business stakeholders, you might say:
"Microservices architecture enables faster development cycles, better scalability, 
and reduced risk when making changes to the system."

Can you try explaining how microservices differ from monolithic architecture?
```

3. **2주차: 프레젠테이션 연습**
```bash
english-study chat --topic "technical presentation" --session-id "week2-presentation"
```

4. **3주차: Q&A 세션 연습**
```bash
english-study chat --topic "conference Q&A" --session-id "week3-qa"
```

### 시나리오 2: 비즈니스 전문가의 국제 회의 준비

**배경**: 마케팅 매니저가 글로벌 팀과의 협업을 위한 영어 실력 향상

**문서 준비:**
```
business-documents/
├── marketing/
│   ├── digital-marketing-trends.pdf
│   ├── customer-segmentation-guide.docx
│   └── brand-strategy-framework.md
├── management/
│   ├── team-leadership-principles.pdf
│   └── cross-cultural-communication.md
└── industry/
    ├── market-analysis-reports.pdf
    └── competitive-intelligence.docx
```

**학습 과정:**

1. **비즈니스 영어 기초**
```bash
english-study chat --topic "business communication" --session-id "biz-basics"
```

**대화 예제:**
```
👤 User: I need to present our marketing strategy to international team.

💬 Assistant: That's a great opportunity! Let me help you prepare. 
Based on your marketing documents, I can see you have comprehensive strategy materials.

First, let's work on your opening. Instead of "I need to present," try:
✅ "I'll be presenting our marketing strategy to the international team."
✅ "I'm going to walk you through our marketing strategy."

What's the main objective of your marketing strategy that you want to highlight?

👤 User: We want to increase brand awareness in Asian markets.

💬 Assistant: Perfect! That's clear and specific. For a more professional presentation style:

✅ "Our primary objective is to enhance brand awareness across Asian markets."
✅ "We're focusing on expanding our brand presence in the Asian market segment."

Now, can you tell me about your target audience in these markets?
```

2. **회의 진행 영어**
```bash
english-study chat --topic "meeting facilitation" --session-id "meeting-skills"
```

3. **협상 및 의사결정**
```bash
english-study chat --topic "business negotiation" --session-id "negotiation-skills"
```

### 시나리오 3: 학술 연구자의 논문 발표 준비

**배경**: 대학원생이 국제 학회에서 연구 결과 발표

**문서 준비:**
```
research-documents/
├── papers/
│   ├── my-research-paper.pdf
│   ├── related-work-survey.pdf
│   └── methodology-references.pdf
├── presentations/
│   ├── academic-presentation-guide.md
│   └── conference-talk-examples.txt
└── writing/
    ├── academic-writing-style.pdf
    └── research-communication.docx
```

**학습 과정:**

1. **학술 영어 표현**
```bash
english-study chat --topic "academic presentation" --session-id "academic-english"
```

**대화 예제:**
```
👤 User: I research about deep learning for image recognition.

💬 Assistant: Fascinating research area! Let me help you express this more academically:

✅ "I conduct research on deep learning for image recognition."
✅ "My research focuses on deep learning applications in image recognition."
✅ "I'm investigating deep learning approaches to image recognition."

Academic presentations often start with context. How would you introduce the problem 
your research addresses?

👤 User: Image recognition is very difficult problem in computer vision.

💡 Assistant: Good foundation! Let's make it more academic:

✅ "Image recognition presents significant challenges in computer vision."
✅ "Image recognition remains a fundamental challenge in the field of computer vision."
✅ "Despite recent advances, image recognition continues to pose substantial challenges..."

Can you explain what specific aspect of image recognition your research improves?
```

## 🔬 전문 분야별 활용

### 의료 분야

**문서 예제:**
- 의학 논문 및 케이스 스터디
- 의료 기기 매뉴얼
- 환자 커뮤니케이션 가이드

**활용 명령어:**
```bash
english-study set-docs ./medical-documents
english-study chat --topic "medical terminology"
english-study chat --topic "patient communication"
```

**대화 예제:**
```
👤 User: How can I explain diagnosis to patient in simple way?

💬 Assistant: Great question! Clear patient communication is crucial. 
Based on your medical communication guides, here are some strategies:

Instead of: "You have hypertension with comorbid diabetes mellitus."
Try: "You have high blood pressure, and you also have diabetes."

Key principles:
1. Use everyday language instead of medical terms
2. Check understanding: "Does this make sense to you?"
3. Use analogies: "Think of your blood vessels like garden hoses..."

Can you practice explaining a common condition using simple terms?
```

### 법률 분야

**문서 예제:**
- 법률 문서 및 계약서
- 판례 분석
- 법정 절차 가이드

**활용 명령어:**
```bash
english-study set-docs ./legal-documents
english-study chat --topic "contract law"
english-study chat --topic "legal writing"
```

### 금융 분야

**문서 예제:**
- 금융 리포트
- 투자 분석 자료
- 경제 동향 보고서

**활용 명령어:**
```bash
english-study set-docs ./finance-documents
english-study chat --topic "financial analysis"
english-study chat --topic "investment strategy"
```

## 💬 대화 예제 모음

### 기술 토론

```
👤 User: What do you think about artificial intelligence future?

💬 Assistant: That's a thought-provoking question! Based on your AI research documents, 
there are several fascinating perspectives to consider.

First, let me help with your question structure:
✅ "What do you think about the future of artificial intelligence?"
✅ "What are your thoughts on AI's future prospects?"

From your documents, I can see discussions about both opportunities and challenges. 
What aspect interests you most - the technological possibilities or the societal implications?

👤 User: I'm worry about job displacement because of AI.

💡 Assistant: That's a valid concern shared by many. Small correction:
✅ "I'm worried about job displacement due to AI."
✅ "I have concerns about AI-driven job displacement."

Your documents mention this issue. Some experts argue that while AI may automate certain tasks, 
it also creates new types of jobs. What's your perspective on reskilling and adaptation?
```

### 비즈니스 전략

```
👤 User: Our company need to digital transformation.

💬 Assistant: Digital transformation is indeed crucial for modern businesses! 
Let me help you express this more professionally:

✅ "Our company needs digital transformation."
✅ "Our company requires a digital transformation strategy."
✅ "We need to undergo digital transformation."

Based on your business strategy documents, what's driving this need? 
Is it customer expectations, competitive pressure, or operational efficiency?

👤 User: Customers expect more personalized experience now.

💬 Assistant: Exactly! Customer expectations have evolved significantly. 
Your phrasing is good, but here's a more business-formal version:

✅ "Customers now expect more personalized experiences."
✅ "There's a growing demand for personalized customer experiences."
✅ "Customer expectations have shifted toward personalization."

What specific areas of personalization is your company considering?
```

### 학술 토론

```
👤 User: My research show that deep learning can improve accuracy.

💬 Assistant: Interesting findings! Let me help you express this in academic style:

✅ "My research shows that deep learning can improve accuracy."
✅ "Our research demonstrates that deep learning enhances accuracy."
✅ "The findings indicate that deep learning approaches yield improved accuracy."

For academic writing, you might also specify:
- Accuracy of what? (image classification, prediction, etc.)
- By how much? (quantitative improvement)
- Compared to what baseline?

Can you elaborate on your specific results?
```

## ⚙️ 설정 예제

### 다양한 LLM 설정

```bash
# OpenAI GPT-4 (최고 품질)
english-study set-llm openai \
  --api-key sk-your-key \
  --model gpt-4 \
  --temperature 0.7 \
  --max-tokens 1500

# Google Gemini (균형잡힌 성능)
english-study set-llm gemini \
  --api-key AIza-your-key \
  --model gemini-pro \
  --temperature 0.8 \
  --max-tokens 1200

# Ollama 로컬 모델 (개인정보 보호)
english-study set-llm ollama \
  --model llama2 \
  --host localhost:11434 \
  --temperature 0.6 \
  --max-tokens 1000

# 코딩 특화 모델
english-study set-llm ollama \
  --model codellama \
  --temperature 0.3 \
  --max-tokens 2000
```

### 학습 수준별 설정

```bash
# 초급자 설정
english-study set-language ko \
  --learning-level beginner \
  --feedback-level detailed

# 중급자 설정 (기본값)
english-study set-language ko \
  --learning-level intermediate \
  --feedback-level normal

# 고급자 설정
english-study set-language en \
  --learning-level advanced \
  --feedback-level minimal
```

### 전문 분야별 설정

```bash
# 기술 분야
english-study set-docs ./tech-docs
english-study set-llm openai --model gpt-4 --temperature 0.5

# 비즈니스 분야
english-study set-docs ./business-docs
english-study set-llm gemini --temperature 0.7

# 학술 분야
english-study set-docs ./academic-papers
english-study set-llm openai --model gpt-4 --temperature 0.3
```

## 🎯 학습 목표별 활용

### 목표 1: 프레젠테이션 스킬 향상

**4주 학습 계획:**

```bash
# 1주차: 기본 프레젠테이션 구조
english-study chat --topic "presentation structure" --session-id "week1-structure"

# 2주차: 시각 자료 설명
english-study chat --topic "describing charts" --session-id "week2-visuals"

# 3주차: 질의응답 처리
english-study chat --topic "handling questions" --session-id "week3-qa"

# 4주차: 종합 연습
english-study chat --topic "full presentation" --session-id "week4-practice"
```

### 목표 2: 비즈니스 이메일 작성

**학습 시퀀스:**

```bash
# 기본 이메일 구조
english-study chat --topic "email structure"

# 정중한 요청 표현
english-study chat --topic "polite requests"

# 회의 일정 조율
english-study chat --topic "scheduling meetings"

# 프로젝트 업데이트 보고
english-study chat --topic "project updates"
```

### 목표 3: 기술 문서 작성

**단계별 학습:**

```bash
# 기술 명세서 작성
english-study chat --topic "technical specifications"

# API 문서 작성
english-study chat --topic "API documentation"

# 사용자 가이드 작성
english-study chat --topic "user guide writing"

# 코드 리뷰 코멘트
english-study chat --topic "code review comments"
```

## 📊 성과 측정 예제

### 학습 진행 추적

```bash
# 세션 기록 확인
english-study status --detailed

# 특정 세션 재개하여 진행 상황 확인
english-study chat --session-id "business-week1"
```

### 개선 영역 식별

**정기적인 자가 평가:**
1. 문법 정확도 향상
2. 어휘 다양성 증가
3. 자연스러운 표현 사용
4. 전문 용어 활용 능력

## 🚀 고급 활용 팁

### 1. 멀티 세션 관리

```bash
# 주제별 세션 분리
english-study chat --session-id "tech-discussions"
english-study chat --session-id "business-meetings"
english-study chat --session-id "academic-writing"
```

### 2. 문서 세트 전환

```bash
# 작업 환경별 문서 세트
english-study set-docs ./work-documents    # 업무용
english-study set-docs ./study-materials   # 학습용
english-study set-docs ./hobby-content     # 취미용
```

### 3. 성능 최적화

```bash
# 빠른 응답을 위한 설정
english-study set-llm openai --max-tokens 500 --temperature 0.5

# 창의적 대화를 위한 설정
english-study set-llm openai --max-tokens 1500 --temperature 0.9
```

---

이 예제들이 Document RAG English Study를 효과적으로 활용하는 데 도움이 되기를 바랍니다. 
더 많은 예제나 특정 상황에 대한 가이드가 필요하시면 [Issues](https://github.com/your-username/document-rag-english-study/issues)에 요청해주세요!