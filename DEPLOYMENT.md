# 배포 가이드

Document RAG English Study의 다양한 환경에서의 배포 방법을 안내합니다.

## 📋 목차

1. [배포 준비](#배포-준비)
2. [로컬 배포](#로컬-배포)
3. [서버 배포](#서버-배포)
4. [Docker 배포](#docker-배포)
5. [클라우드 배포](#클라우드-배포)
6. [모니터링 및 유지보수](#모니터링-및-유지보수)

## 🚀 배포 준비

### 시스템 요구사항

#### 최소 요구사항
- **CPU**: 2 코어
- **메모리**: 4GB RAM
- **저장공간**: 10GB 여유 공간
- **Python**: 3.9 이상
- **네트워크**: 인터넷 연결 (API 사용 시)

#### 권장 요구사항
- **CPU**: 4 코어 이상
- **메모리**: 8GB RAM 이상
- **저장공간**: 50GB 여유 공간
- **Python**: 3.11 이상
- **GPU**: CUDA 지원 GPU (로컬 모델 사용 시)

### 환경 변수 설정

배포 전 필요한 환경 변수를 설정하세요:

```bash
# .env 파일 생성
cat > .env << EOF
# LLM API 키
OPENAI_API_KEY=sk-your-openai-api-key
GOOGLE_API_KEY=AIza-your-google-api-key

# 로그 레벨
LOG_LEVEL=INFO

# 데이터 디렉토리
DATA_DIR=/opt/document-rag-english-study/data

# 설정 디렉토리
CONFIG_DIR=/opt/document-rag-english-study/config

# Ollama 서버 (로컬 모델 사용 시)
OLLAMA_HOST=localhost:11434
EOF
```

## 💻 로컬 배포

### 개발 환경 배포

```bash
# 1. 저장소 클론
git clone https://github.com/your-username/document-rag-english-study.git
cd document-rag-english-study

# 2. 가상환경 생성 및 활성화
uv venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# 3. 의존성 설치
uv pip install -e ".[dev]"

# 4. 환경 변수 로드
source .env

# 5. 초기 설정
english-study setup

# 6. 테스트 실행
pytest

# 7. 애플리케이션 실행
english-study chat
```

### 프로덕션 환경 배포

```bash
# 1. 프로덕션 의존성만 설치
uv pip install -e .

# 2. 설정 파일 복사
cp config/default.yaml config/production.yaml

# 3. 프로덕션 설정 수정
vim config/production.yaml

# 4. 로그 디렉토리 생성
mkdir -p /var/log/document-rag-english-study

# 5. 데이터 디렉토리 생성
mkdir -p /opt/document-rag-english-study/data

# 6. 권한 설정
chown -R $USER:$USER /opt/document-rag-english-study
chmod -R 755 /opt/document-rag-english-study
```

## 🖥️ 서버 배포

### Ubuntu/Debian 서버

```bash
# 1. 시스템 업데이트
sudo apt update && sudo apt upgrade -y

# 2. Python 및 필수 패키지 설치
sudo apt install -y python3.11 python3.11-venv python3-pip git curl

# 3. uv 설치
curl -LsSf https://astral.sh/uv/install.sh | sh
source ~/.bashrc

# 4. 사용자 및 디렉토리 생성
sudo useradd -m -s /bin/bash english-study
sudo mkdir -p /opt/document-rag-english-study
sudo chown english-study:english-study /opt/document-rag-english-study

# 5. 애플리케이션 배포
sudo -u english-study bash << 'EOF'
cd /opt/document-rag-english-study
git clone https://github.com/your-username/document-rag-english-study.git .
uv venv
source .venv/bin/activate
uv pip install -e .
EOF

# 6. 환경 변수 설정
sudo -u english-study tee /opt/document-rag-english-study/.env << 'EOF'
OPENAI_API_KEY=your-api-key
LOG_LEVEL=INFO
DATA_DIR=/opt/document-rag-english-study/data
CONFIG_DIR=/opt/document-rag-english-study/config
EOF

# 7. systemd 서비스 생성 (선택사항)
sudo tee /etc/systemd/system/english-study.service << 'EOF'
[Unit]
Description=Document RAG English Study
After=network.target

[Service]
Type=simple
User=english-study
WorkingDirectory=/opt/document-rag-english-study
Environment=PATH=/opt/document-rag-english-study/.venv/bin
ExecStart=/opt/document-rag-english-study/.venv/bin/python -m document_rag_english_study.cli.main
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
EOF

# 8. 서비스 활성화
sudo systemctl daemon-reload
sudo systemctl enable english-study
sudo systemctl start english-study
```

### CentOS/RHEL 서버

```bash
# 1. 시스템 업데이트
sudo yum update -y

# 2. Python 및 필수 패키지 설치
sudo yum install -y python3.11 python3-pip git curl

# 3. 나머지 단계는 Ubuntu와 동일
# (위의 Ubuntu 가이드 4번부터 따라하기)
```

## 🐳 Docker 배포

### Dockerfile 생성

```dockerfile
# Dockerfile
FROM python:3.11-slim

# 작업 디렉토리 설정
WORKDIR /app

# 시스템 의존성 설치
RUN apt-get update && apt-get install -y \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# uv 설치
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.cargo/bin:$PATH"

# 애플리케이션 코드 복사
COPY . .

# Python 의존성 설치
RUN uv venv && \
    . .venv/bin/activate && \
    uv pip install -e .

# 환경 변수 설정
ENV PATH="/app/.venv/bin:$PATH"
ENV PYTHONPATH="/app"

# 데이터 및 설정 디렉토리 생성
RUN mkdir -p /app/data /app/config /app/logs

# 포트 노출 (필요시)
EXPOSE 8000

# 헬스체크
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import document_rag_english_study; print('OK')" || exit 1

# 실행 명령
CMD ["python", "-m", "document_rag_english_study.cli.main", "chat"]
```

### Docker Compose 설정

```yaml
# docker-compose.yml
version: '3.8'

services:
  english-study:
    build: .
    container_name: document-rag-english-study
    environment:
      - OPENAI_API_KEY=${OPENAI_API_KEY}
      - GOOGLE_API_KEY=${GOOGLE_API_KEY}
      - LOG_LEVEL=INFO
      - DATA_DIR=/app/data
      - CONFIG_DIR=/app/config
    volumes:
      - ./data:/app/data
      - ./config:/app/config
      - ./logs:/app/logs
      - ./documents:/app/documents
    restart: unless-stopped
    stdin_open: true
    tty: true

  # Ollama 서비스 (로컬 모델 사용 시)
  ollama:
    image: ollama/ollama:latest
    container_name: ollama-server
    ports:
      - "11434:11434"
    volumes:
      - ollama_data:/root/.ollama
    restart: unless-stopped

volumes:
  ollama_data:
```

### Docker 배포 실행

```bash
# 1. Docker 이미지 빌드
docker build -t document-rag-english-study .

# 2. 환경 변수 파일 생성
echo "OPENAI_API_KEY=your-api-key" > .env
echo "GOOGLE_API_KEY=your-api-key" >> .env

# 3. Docker Compose로 실행
docker-compose up -d

# 4. 컨테이너 상태 확인
docker-compose ps

# 5. 로그 확인
docker-compose logs -f english-study

# 6. 컨테이너 내부 접근
docker-compose exec english-study bash

# 7. 애플리케이션 실행
docker-compose exec english-study english-study setup
```

## ☁️ 클라우드 배포

### AWS EC2 배포

```bash
# 1. EC2 인스턴스 생성 (Ubuntu 22.04 LTS)
# - 인스턴스 타입: t3.medium 이상
# - 보안 그룹: SSH (22), HTTP (80), HTTPS (443)
# - 키 페어: 생성 또는 기존 사용

# 2. 인스턴스 접속
ssh -i your-key.pem ubuntu@your-ec2-ip

# 3. 서버 배포 가이드 따라하기
# (위의 Ubuntu 서버 배포 가이드 참조)

# 4. 로드 밸런서 설정 (선택사항)
# AWS Application Load Balancer 생성

# 5. 도메인 설정 (선택사항)
# Route 53에서 도메인 연결
```

### Google Cloud Platform 배포

```bash
# 1. GCP 프로젝트 생성 및 설정
gcloud config set project your-project-id

# 2. Compute Engine 인스턴스 생성
gcloud compute instances create english-study-vm \
    --zone=us-central1-a \
    --machine-type=e2-medium \
    --image-family=ubuntu-2204-lts \
    --image-project=ubuntu-os-cloud \
    --boot-disk-size=50GB

# 3. 인스턴스 접속
gcloud compute ssh english-study-vm --zone=us-central1-a

# 4. 서버 배포 가이드 따라하기
```

### Azure 배포

```bash
# 1. 리소스 그룹 생성
az group create --name english-study-rg --location eastus

# 2. 가상 머신 생성
az vm create \
    --resource-group english-study-rg \
    --name english-study-vm \
    --image Ubuntu2204 \
    --size Standard_B2s \
    --admin-username azureuser \
    --generate-ssh-keys

# 3. SSH 접속
az vm show --resource-group english-study-rg --name english-study-vm -d --query publicIps -o tsv
ssh azureuser@your-vm-ip

# 4. 서버 배포 가이드 따라하기
```

### Kubernetes 배포

```yaml
# k8s-deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: english-study-deployment
spec:
  replicas: 2
  selector:
    matchLabels:
      app: english-study
  template:
    metadata:
      labels:
        app: english-study
    spec:
      containers:
      - name: english-study
        image: document-rag-english-study:latest
        ports:
        - containerPort: 8000
        env:
        - name: OPENAI_API_KEY
          valueFrom:
            secretKeyRef:
              name: api-keys
              key: openai-api-key
        - name: LOG_LEVEL
          value: "INFO"
        volumeMounts:
        - name: data-volume
          mountPath: /app/data
        - name: config-volume
          mountPath: /app/config
      volumes:
      - name: data-volume
        persistentVolumeClaim:
          claimName: english-study-data-pvc
      - name: config-volume
        configMap:
          name: english-study-config

---
apiVersion: v1
kind: Service
metadata:
  name: english-study-service
spec:
  selector:
    app: english-study
  ports:
  - port: 80
    targetPort: 8000
  type: LoadBalancer

---
apiVersion: v1
kind: Secret
metadata:
  name: api-keys
type: Opaque
data:
  openai-api-key: <base64-encoded-api-key>
```

```bash
# Kubernetes 배포 실행
kubectl apply -f k8s-deployment.yaml

# 배포 상태 확인
kubectl get deployments
kubectl get pods
kubectl get services

# 로그 확인
kubectl logs -f deployment/english-study-deployment
```

## 📊 모니터링 및 유지보수

### 로그 모니터링

```bash
# 로그 파일 위치
/var/log/document-rag-english-study/
├── application.log
├── error.log
└── access.log

# 실시간 로그 모니터링
tail -f /var/log/document-rag-english-study/application.log

# 로그 로테이션 설정
sudo tee /etc/logrotate.d/english-study << 'EOF'
/var/log/document-rag-english-study/*.log {
    daily
    rotate 30
    compress
    delaycompress
    missingok
    notifempty
    create 644 english-study english-study
}
EOF
```

### 시스템 모니터링

```bash
# 시스템 리소스 모니터링
htop
iostat -x 1
free -h
df -h

# 프로세스 모니터링
ps aux | grep english-study
systemctl status english-study

# 네트워크 모니터링
netstat -tlnp | grep python
ss -tlnp | grep python
```

### 성능 모니터링

```bash
# Python 애플리케이션 프로파일링
python -m cProfile -o profile.stats -m document_rag_english_study.cli.main

# 메모리 사용량 모니터링
python -m memory_profiler your_script.py

# 데이터베이스 성능 모니터링 (ChromaDB)
# 쿼리 응답 시간 및 인덱스 성능 확인
```

### 백업 및 복구

```bash
# 데이터 백업 스크립트
#!/bin/bash
BACKUP_DIR="/backup/english-study"
DATA_DIR="/opt/document-rag-english-study/data"
CONFIG_DIR="/opt/document-rag-english-study/config"
DATE=$(date +%Y%m%d_%H%M%S)

# 백업 디렉토리 생성
mkdir -p $BACKUP_DIR

# 데이터 백업
tar -czf $BACKUP_DIR/data_backup_$DATE.tar.gz -C $DATA_DIR .
tar -czf $BACKUP_DIR/config_backup_$DATE.tar.gz -C $CONFIG_DIR .

# 오래된 백업 파일 삭제 (30일 이상)
find $BACKUP_DIR -name "*.tar.gz" -mtime +30 -delete

echo "Backup completed: $DATE"
```

```bash
# 복구 스크립트
#!/bin/bash
BACKUP_FILE=$1
RESTORE_DIR="/opt/document-rag-english-study"

if [ -z "$BACKUP_FILE" ]; then
    echo "Usage: $0 <backup_file.tar.gz>"
    exit 1
fi

# 서비스 중지
sudo systemctl stop english-study

# 백업 복구
tar -xzf $BACKUP_FILE -C $RESTORE_DIR

# 권한 설정
chown -R english-study:english-study $RESTORE_DIR

# 서비스 시작
sudo systemctl start english-study

echo "Restore completed from: $BACKUP_FILE"
```

### 자동화된 배포 (CI/CD)

```yaml
# .github/workflows/deploy.yml
name: Deploy to Production

on:
  push:
    branches: [ main ]

jobs:
  deploy:
    runs-on: ubuntu-latest
    
    steps:
    - uses: actions/checkout@v3
    
    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: '3.11'
    
    - name: Install uv
      run: curl -LsSf https://astral.sh/uv/install.sh | sh
    
    - name: Install dependencies
      run: |
        source ~/.cargo/env
        uv venv
        source .venv/bin/activate
        uv pip install -e .
    
    - name: Run tests
      run: |
        source .venv/bin/activate
        pytest
    
    - name: Build Docker image
      run: docker build -t document-rag-english-study:${{ github.sha }} .
    
    - name: Deploy to server
      run: |
        # SSH를 통한 서버 배포 스크립트 실행
        ssh -o StrictHostKeyChecking=no ${{ secrets.SERVER_USER }}@${{ secrets.SERVER_HOST }} \
          "cd /opt/document-rag-english-study && \
           git pull origin main && \
           source .venv/bin/activate && \
           uv pip install -e . && \
           sudo systemctl restart english-study"
```

### 보안 설정

```bash
# 방화벽 설정
sudo ufw enable
sudo ufw allow ssh
sudo ufw allow 80
sudo ufw allow 443

# SSL 인증서 설정 (Let's Encrypt)
sudo apt install certbot
sudo certbot --nginx -d your-domain.com

# 정기적인 보안 업데이트
sudo apt update && sudo apt upgrade -y

# 로그 모니터링 및 침입 탐지
sudo apt install fail2ban
sudo systemctl enable fail2ban
sudo systemctl start fail2ban
```

### 문제 해결

```bash
# 일반적인 문제 진단
# 1. 서비스 상태 확인
systemctl status english-study

# 2. 로그 확인
journalctl -u english-study -f

# 3. 포트 사용 확인
netstat -tlnp | grep :8000

# 4. 디스크 공간 확인
df -h

# 5. 메모리 사용량 확인
free -h

# 6. 프로세스 확인
ps aux | grep python

# 7. 네트워크 연결 확인
ping google.com
curl -I https://api.openai.com
```

이 배포 가이드를 통해 다양한 환경에서 Document RAG English Study를 성공적으로 배포하고 운영할 수 있습니다. 각 환경의 특성에 맞게 설정을 조정하고, 정기적인 모니터링과 유지보수를 통해 안정적인 서비스를 제공하세요.