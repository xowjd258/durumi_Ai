# 리뷰 분석 도구

이 프로젝트는 ChatGPT API를 활용하여 리뷰를 분석하는 코드를 포함하고 있습니다. 사용자가 제공한 리뷰 데이터를 분석하여 인사이트를 제공하는 것을 목표로 합니다.

## 소개

이 도구는 데이터셋을 분석하여 중요한 메트릭을 추출하고, 결과를 요약하여 사용자가 쉽게 이해할 수 있도록 돕습니다.

- **입력 양식**: `dataset/input.csv` 파일을 참조해 주세요.
- **결과 양식**: 분석 결과는 `output/output.csv`에 저장됩니다.
- **환경 설정**: `.env` 파일을 생성하고, 여기에 OpenAI 키를 `OPENAI_API_KEY=여기에키입력` 형식으로 입력해야 합니다.

## 설치

이 코드를 실행하기 전에 필요한 라이브러리를 설치해야 합니다. 다음 명령어를 사용하여 의존성을 설치하세요:

```bash
pip install -r requirements.txt
```

## 실행 방법

기본 실행 명령어는 아래와 같습니다:

```bash
python durumi_single.py
```

이 명령어는 단일 스레드로 프로그램을 실행합니다.

## 멀티 스레드 실행

멀티 스레드를 사용하여 분석을 더 빠르게 수행하려면 다음 명령어를 사용하세요:

```bash
python durumi_thread.py
```

멀티 스레드 실행을 통해 데이터 처리 속도를 향상시킬 수 있습니다.

## 피드백

프로젝트에 대한 피드백은 언제든지 환영합니다. 문제점이나 개선사항에 대한 의견이 있다면, 이슈를 등록하거나 풀 리퀘스트를 보내주세요.
