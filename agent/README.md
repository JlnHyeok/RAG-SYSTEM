# RAG Agent Service

이 프로젝트는 멀티모달 문서 처리를 위한 RAG (Retrieval Augmented Generation) 에이전트 서비스입니다.

## 주요 기능

- **멀티모달 문서 처리**: PDF, 이미지 (PNG, JPG, JPEG), 텍스트 파일 지원
- **OCR 통합**: Tesseract, PaddleOCR, EasyOCR를 통한 이미지 텍스트 추출
- **벡터 검색**: Qdrant를 이용한 고성능 벡터 유사도 검색
- **Gemini API**: Google Gemini 모델을 이용한 고품질 응답 생성
- **건축/기계 도면 처리**: 저품질 스캔 이미지 대응 이미지 품질 향상

## 기술 스택

- **Backend**: FastAPI, Python 3.8+
- **Vector DB**: Qdrant
- **LLM**: Google Gemini API
- **Embedding**: sentence-transformers (한국어 지원)
- **OCR**: Tesseract, PaddleOCR, EasyOCR
- **Image Processing**: OpenCV, Pillow

## 설치 및 실행

### 1. 의존성 설치

```bash
pip install -r requirements.txt
```

### 2. 환경 변수 설정

```bash
cp .env.example .env
# .env 파일을 열어 필요한 API 키와 설정을 입력하세요
```

필수 환경변수:

- `GEMINI_API_KEY`: Google Gemini API 키
- `QDRANT_HOST`: Qdrant 서버 호스트
- `QDRANT_PORT`: Qdrant 서버 포트

### 3. Qdrant 실행

Docker를 사용하여 Qdrant 실행:

```bash
docker run -p 6333:6333 -v $(pwd)/qdrant_storage:/qdrant/storage:z qdrant/qdrant
```

### 4. 서버 실행

```bash
python -m uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

서버가 시작되면 다음 URL에서 접근할 수 있습니다:

- API 문서: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc
- Health Check: http://localhost:8000/health

## API 사용법

### 문서 업로드

```bash
curl -X POST "http://localhost:8000/api/v1/documents/upload" \
     -H "Content-Type: multipart/form-data" \
     -F "file=@your_document.pdf" \
     -F "metadata={\"title\":\"문서 제목\",\"tags\":[\"태그1\",\"태그2\"]}"
```

### 질의 검색

```bash
curl -X POST "http://localhost:8000/api/v1/query/search" \
     -H "Content-Type: application/json" \
     -d '{"query":"검색할 질문","limit":5,"use_gemini":true}'
```

### 벡터 검색

```bash
curl -X POST "http://localhost:8000/api/v1/query/vector-search" \
     -H "Content-Type: application/json" \
     -d '{"query":"검색 쿼리","limit":5,"score_threshold":0.7}'
```

## 프로젝트 구조

```
agent/
├── app/
│   ├── api/                # API 엔드포인트
│   │   └── v1/
│   │       ├── documents.py    # 문서 업로드/관리
│   │       ├── query.py        # 쿼리 검색
│   │       └── health.py       # 헬스체크
│   ├── core/               # 핵심 모듈
│   │   ├── embedding_manager.py    # 임베딩 관리
│   │   ├── vector_store.py         # 벡터 저장소
│   │   ├── gemini_service.py       # Gemini API 서비스
│   │   └── rag_engine.py          # RAG 엔진
│   ├── models/             # 데이터 모델
│   │   ├── schemas.py          # Pydantic 스키마
│   │   └── enums.py           # 열거형 정의
│   ├── services/           # 서비스 계층
│   │   └── document_processor.py   # 문서 처리 서비스
│   ├── utils/              # 유틸리티
│   │   ├── logger.py           # 로깅 설정
│   │   └── file_handler.py     # 파일 처리 유틸
│   └── main.py             # FastAPI 애플리케이션
├── tests/                  # 테스트
│   ├── test_embedding.py       # 임베딩 테스트
│   ├── test_vector_store.py    # 벡터 저장소 테스트
│   └── test_api.py            # API 테스트
├── config.py               # 설정 관리
├── requirements.txt        # Python 의존성
├── .env.example           # 환경변수 예제
└── README.md              # 프로젝트 문서
```

## 지원 파일 형식

- **텍스트**: `.txt`, `.md`
- **PDF**: `.pdf` (이미지 포함 PDF 지원)
- **이미지**: `.png`, `.jpg`, `.jpeg`, `.tiff`, `.bmp`
- **문서**: `.docx` (향후 지원 예정)

## 개발

### 테스트 실행

```bash
pytest tests/
```

### 코드 스타일 검사

```bash
black app/
flake8 app/
```

### 개발 서버 실행

```bash
python -m uvicorn app.main:app --reload --log-level debug
```

## 문제 해결

### 일반적인 문제

1. **Qdrant 연결 실패**

   - Qdrant 서버가 실행 중인지 확인
   - 포트가 올바른지 확인 (기본: 6333)

2. **Gemini API 오류**

   - API 키가 올바른지 확인
   - API 할당량을 확인

3. **OCR 결과가 부정확**

   - 이미지 품질이 낮은 경우 전처리 옵션 활성화
   - OCR 신뢰도 임계값 조정

4. **메모리 부족**
   - 큰 파일 처리 시 청크 크기 조정
   - 배치 처리 크기 감소

## 기여

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 라이선스

이 프로젝트는 MIT 라이선스 하에 있습니다. 자세한 내용은 `LICENSE` 파일을 참조하세요.

## 연락처

프로젝트에 대한 질문이나 피드백이 있으시면 이슈를 생성해 주세요.
