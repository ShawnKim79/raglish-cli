# 설치 가이드

Document RAG English Study 설치 및 설정에 대한 상세 가이드입니다.

## 📋 시스템 요구사항

### 최소 요구사항
- **Python**: 3.9 이상
- **메모리**: 4GB RAM
- **저장공간**: 2GB 여유 공간
- **네트워크**: 인터넷 연결 (API 사용 시)

### 권장 요구사항
- **Python**: 3.11 이상
- **메모리**: 8GB RAM
- **저장공간**: 5GB 여유 공간
- **GPU**: CUDA 지원 GPU (로컬 모델 사용 시)

## 🛠️ 설치 단계

### 1단계: Python 환경 확인

```bash
# Python 버전 확인
python --version
# 또는
python3 --version

# Python 3.9 이상이어야 합니다
```

Python이 설치되어 있지 않다면:
- **Windows**: [python.org](https://www.python.org/downloads/)에서 다운로드
- **macOS**: `brew install python` 또는 python.org에서 다운로드
- **Ubuntu/Debian**: `sudo apt update && sudo apt install python3 python3-pip`
- **CentOS/RHEL**: `sudo yum install python3 python3-pip`

### 2단계: uv 패키지 매니저 설치

```bash
# macOS/Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# Windows (PowerShell)
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"

# 또는 pip를 통한 설치
pip install uv
```

설치 확인:
```bash
uv --version
```

### 3단계: 프로젝트 다운로드

```bash
# Git을 사용한 클론
git clone https://github.com/your-username/document-rag-english-study.git
cd document-rag-english-study

# 또는 ZIP 파일 다운로드 후 압축 해제
```

### 4단계: 가상환경 생성 및 의존성 설치

```bash
# 가상환경 생성
uv venv

# 가상환경 활성화
# Linux/macOS:
source .venv/bin/activate

# Windows:
.venv\Scripts\activate

# 의존성 설치
uv pip install -e .

# 개발 의존성 설치 (선택사항)
uv pip install -e ".[dev]"
```

### 5단계: 설치 확인

```bash
# 프로그램 실행 확인
english-study --version

# 도움말 확인
english-study --help
```

## 🤖 LLM 제공업체별 설정

### OpenAI GPT

1. **API 키 발급**
   - [OpenAI Platform](https://platform.openai.com/api-keys) 방문
   - 계정 생성 및 로그인
   - "Create new secret key" 클릭
   - API 키 복사 및 안전한 곳에 저장

2. **환경 변수 설정**
   ```bash
   # Linux/macOS
   export OPENAI_API_KEY="sk-your-api-key-here"
   echo 'export OPENAI_API_KEY="sk-your-api-key-here"' >> ~/.bashrc
   
   # Windows
   set OPENAI_API_KEY=sk-your-api-key-here
   # 또는 시스템 환경 변수에서 설정
   ```

3. **프로그램에서 설정**
   ```bash
   english-study set-llm openai --api-key sk-your-api-key-here
   ```

### Google Gemini

1. **API 키 발급**
   - [Google AI Studio](https://makersuite.google.com/app/apikey) 방문
   - Google 계정으로 로그인
   - "Create API Key" 클릭
   - API 키 복사

2. **환경 변수 설정**
   ```bash
   # Linux/macOS
   export GOOGLE_API_KEY="AIza-your-api-key-here"
   
   # Windows
   set GOOGLE_API_KEY=AIza-your-api-key-here
   ```

3. **프로그램에서 설정**
   ```bash
   english-study set-llm gemini --api-key AIza-your-api-key-here
   ```

### Ollama (로컬 모델)

1. **Ollama 설치**
   - [Ollama 공식 사이트](https://ollama.ai/download) 방문
   - 운영체제에 맞는 설치 파일 다운로드 및 설치

2. **모델 다운로드**
   ```bash
   # 기본 모델 다운로드 (약 4GB)
   ollama pull llama2
   
   # 또는 다른 모델
   ollama pull codellama
   ollama pull mistral
   ```

3. **서버 실행**
   ```bash
   # 백그라운드에서 서버 실행
   ollama serve
   ```

4. **프로그램에서 설정**
   ```bash
   english-study set-llm ollama --model llama2
   ```

## 📁 초기 설정

### 통합 설정 (권장)

```bash
english-study setup
```

이 명령어는 다음을 순서대로 안내합니다:
1. 모국어 선택
2. 문서 디렉토리 설정
3. LLM 제공업체 설정

### 개별 설정

```bash
# 1. 모국어 설정
english-study set-language ko

# 2. 문서 디렉토리 설정
english-study set-docs /path/to/your/documents

# 3. LLM 설정
english-study set-llm openai --api-key your-key
```

## 📚 샘플 문서 준비

학습을 위한 문서를 준비하세요:

### 지원 형식
- **PDF**: 논문, 기술 문서, 책
- **DOCX**: Word 문서
- **TXT**: 텍스트 파일
- **MD**: Markdown 파일

### 권장 문서 구성
```
my-documents/
├── technology/
│   ├── ai-research-paper.pdf
│   ├── programming-guide.md
│   └── tech-articles.txt
├── business/
│   ├── market-analysis.docx
│   └── business-strategy.pdf
└── personal-interests/
    ├── hobby-articles.txt
    └── travel-guides.md
```

### 문서 준비 팁
- **파일 크기**: 10MB 이하 권장
- **언어**: 영어 또는 관심 있는 주제의 다국어 문서
- **내용**: 관심 있고 학습하고 싶은 주제
- **구조**: 폴더별로 주제 분류 권장

## ✅ 설치 확인

### 기본 기능 테스트

```bash
# 1. 상태 확인
english-study status

# 2. 도움말 확인
english-study help

# 3. 설정 상태 확인
english-study status --detailed
```

### 문서 인덱싱 테스트

```bash
# 테스트 문서로 인덱싱
english-study set-docs ./test_docs

# 인덱싱 결과 확인
english-study status
```

### 대화 기능 테스트

```bash
# 간단한 대화 테스트
english-study chat

# 프롬프트에서 "Hello, how are you?" 입력하여 테스트
```

## 🚨 문제 해결

### 일반적인 설치 문제

#### Python 버전 문제
```bash
# Python 버전이 3.9 미만인 경우
python --version

# pyenv를 사용한 Python 버전 관리
curl https://pyenv.run | bash
pyenv install 3.11.0
pyenv global 3.11.0
```

#### uv 설치 실패
```bash
# pip를 통한 대체 설치
pip install --user uv

# 또는 pipx 사용
pip install --user pipx
pipx install uv
```

#### 의존성 설치 실패
```bash
# 캐시 클리어 후 재시도
uv cache clean
uv pip install -e . --no-cache

# 또는 pip 사용
pip install -e .
```

#### 권한 문제 (Linux/macOS)
```bash
# 사용자 권한으로 설치
uv pip install -e . --user

# 또는 sudo 사용 (권장하지 않음)
sudo uv pip install -e .
```

### LLM 연결 문제

#### OpenAI API 오류
```bash
# API 키 확인
echo $OPENAI_API_KEY

# 네트워크 연결 테스트
curl -H "Authorization: Bearer $OPENAI_API_KEY" \
     https://api.openai.com/v1/models
```

#### Ollama 연결 실패
```bash
# 서버 상태 확인
curl http://localhost:11434/api/tags

# 서버 재시작
pkill ollama
ollama serve
```

### 문서 인덱싱 문제

#### 파일 권한 오류
```bash
# 파일 권한 확인
ls -la /path/to/documents

# 권한 수정
chmod -R 755 /path/to/documents
```

#### 지원되지 않는 파일 형식
```bash
# 지원 형식 확인: PDF, DOCX, TXT, MD
file /path/to/document.ext

# 파일 변환 (예: HTML to MD)
pandoc input.html -o output.md
```

## 🔄 업데이트

### 프로그램 업데이트

```bash
# Git을 통한 업데이트
git pull origin main

# 의존성 업데이트
uv pip install -e . --upgrade

# 설정 마이그레이션 (필요시)
english-study setup --migrate
```

### 설정 백업 및 복원

```bash
# 설정 백업
cp -r ~/.config/document-rag-english-study ~/backup/

# 설정 복원
cp -r ~/backup/document-rag-english-study ~/.config/
```

## 📞 지원

설치 중 문제가 발생하면:

1. **로그 확인**: `logs/` 디렉토리의 로그 파일 확인
2. **상태 확인**: `english-study status --detailed` 실행
3. **이슈 리포트**: [GitHub Issues](https://github.com/your-username/document-rag-english-study/issues)에 문제 보고
4. **문서 확인**: README.md 및 관련 문서 재확인

문제 보고 시 다음 정보를 포함해주세요:
- 운영체제 및 버전
- Python 버전
- 오류 메시지
- 실행한 명령어
- 로그 파일 내용