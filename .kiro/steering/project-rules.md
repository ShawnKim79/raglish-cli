# Project Development Rules

## Core Development Rules

### Git Workflow
- **각 작업 단계가 끝나면 반드시 commit을 한다**
- 작업 완료 시점에 의미 있는 커밋 메시지와 함께 변경사항을 커밋
- 커밋 메시지는 "feat: 기능명" 또는 "fix: 수정내용" 형식으로 작성

### Package Management
- **파이썬 패키지는 uv를 이용한다**
- 모든 의존성 설치 및 관리는 uv 명령어 사용
- pyproject.toml 파일을 통한 프로젝트 설정 관리
- 가상환경 생성 및 활성화도 uv 사용

### Dependency Installation
- **필요한 패키지의 설치는 개발자에게 요청한다**
- 새로운 패키지가 필요할 때는 개발자에게 설치 요청
- 패키지명과 용도를 명확히 설명하여 요청
- 설치 완료 후 작업 진행

## Implementation Guidelines

### Code Quality
- 모든 함수와 클래스에 타입 힌트 필수
- Google 스타일 docstring 작성
- 적절한 오류 처리 및 로깅 구현

### Testing
- 각 기능 구현 후 단위 테스트 작성
- pytest 프레임워크 사용
- 테스트 커버리지 90% 이상 목표

### Project Structure
- src/ 디렉토리 기반 구조 사용
- 모듈별 명확한 책임 분리
- 설정 파일은 YAML 형식 사용