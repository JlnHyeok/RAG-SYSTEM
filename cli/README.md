# RAG Agent CLI

간편한 명령줄 인터페이스로 RAG Agent와 상호작용할 수 있습니다.

## 설치 및 실행

```bash
cd cli
python main.py
```

## 사용법

### 대화형 모드

```bash
python main.py
```

### 단일 질문 모드

```bash
python main.py "질문 내용"
```

### 스트리밍 모드

```bash
python main.py -s "질문 내용"
```

## 명령어

- `/upload <파일경로>` - 파일 업로드
- `/delete <문서ID|파일명>` - 문서 삭제
- `/clear` - 모든 문서 삭제
- `/list` - 업로드된 파일 목록
- `/status` - 벡터 DB 상태 확인
- `/search <검색어>` - 벡터 검색 테스트
- `/stream on/off` - 스트리밍 모드 전환
- `/help` - 도움말
- `/exit` - 종료

## 프로젝트 구조

```
cli/
├── main.py              # 메인 진입점
├── config.py           # 설정 파일
├── commands/           # 명령어 모듈
│   ├── ask.py         # 질문 관련
│   ├── upload.py      # 파일 업로드
│   ├── documents.py   # 문서 관리
│   ├── status.py      # 상태 확인
│   └── interactive.py # 대화형 모드
└── utils/             # 유틸리티
    ├── progress.py        # 진행률 추적
    ├── websocket_monitor.py # WebSocket 모니터링
    └── file_utils.py      # 파일 유틸리티
```

## 의존성

- requests
- websocket-client

## 설정

`config.py`에서 서버 URL 및 기타 설정을 변경할 수 있습니다.
