# RAG 시스템 전체 아키텍처 설계

## 1. 시스템 개요

### 1.1 전체 구조

```
┌──────────────────────┐    ┌──────────────────────┐    ┌──────────────────────┐
│      Frontend        │    │       Backend        │    │        Agent         │
│    (SvelteKit)       │◄──►│      (NestJS)        │◄──►│   (Python/FastAPI)   │
│                      │    │                      │    │                      │
│   사용자 인터페이스   │    │   API 게이트웨이     │    │   AI 로직 처리       │
│   - 채팅 UI          │    │   - 인증/권한        │    │   - LLM 호출         │
│   - 파일 업로드      │    │   - 데이터 관리      │    │   - 임베딩 생성      │
│   - 검색 결과 표시   │    │   - 로깅/모니터링    │    │   - 벡터 검색        │
└──────────────────────┘    └──────────────────────┘    └──────────────────────┘
                              │                        │
                              ▼                        ▼
                       ┌─────────────────┐    ┌─────────────────┐
                       │   PostgreSQL    │    │   Vector DB     │
                       │   (메타데이터)    │    │   (Qdrant)      │
                       └─────────────────┘    └─────────────────┘
```

### 1.2 주요 컴포넌트

- **Frontend**: 사용자 인터페이스 및 채팅 UI
- **Backend**: API 서버, 인증, 파일 관리
- **Agent**: AI 로직 처리, 임베딩 모델, LLM 통합
- **Database**: 메타데이터 및 벡터 저장

## 2. 기술 스택

### 2.1 Frontend (SvelteKit)

```typescript
// 주요 라이브러리
{
  "dependencies": {
    "@sveltejs/kit": "^2.0.0",
    "svelte": "^4.0.0",
    "tailwindcss": "^3.3.0",
    "daisyui": "^4.0.0",
    "lucide-svelte": "^0.294.0",
    "socket.io-client": "^4.7.0",
    "marked": "^9.1.0",
    "prismjs": "^1.29.0"
  }
}
```

### 2.2 Backend (NestJS)

```typescript
// 주요 라이브러리
{
  "dependencies": {
    "@nestjs/core": "^10.0.0",
    "@nestjs/common": "^10.0.0",
    "@nestjs/typeorm": "^10.0.0",
    "@nestjs/passport": "^10.0.0",
    "@nestjs/jwt": "^10.1.0",
    "@nestjs/websockets": "^10.0.0",
    "typeorm": "^0.3.17",
    "pg": "^8.11.0",
    "multer": "^1.4.5",
    "socket.io": "^4.7.0",
    "class-validator": "^0.14.0",
    "class-transformer": "^0.5.1"
  }
}
```

### 2.3 Agent (Python/FastAPI)

```python
# Python 라이브러리
fastapi==0.104.1
sentence-transformers==2.2.2  # 임베딩 모델
qdrant-client==1.7.0          # 벡터 DB
google-generativeai==0.3.0    # Gemini API
pytesseract==0.3.10           # OCR
opencv-python==4.8.1          # 이미지 처리
torch==2.1.0                  # GPU 가속
transformers==4.35.0          # 허깅페이스 모델
pypdf2==3.0.1
python-docx==1.1.0
pandas==2.1.4
numpy==1.24.3
```

## 3. 데이터베이스 설계

### 3.1 PostgreSQL (메타데이터)

#### Users 테이블

```sql
CREATE TABLE users (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    email VARCHAR(255) UNIQUE NOT NULL,
    password_hash VARCHAR(255) NOT NULL,
    name VARCHAR(100) NOT NULL,
    role VARCHAR(20) DEFAULT 'user',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

#### Documents 테이블

```sql
CREATE TABLE documents (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID REFERENCES users(id) ON DELETE CASCADE,
    title VARCHAR(500) NOT NULL,
    file_path VARCHAR(1000) NOT NULL,
    file_type VARCHAR(50) NOT NULL,
    file_size BIGINT NOT NULL,
    chunk_count INTEGER DEFAULT 0,
    status VARCHAR(20) DEFAULT 'processing',
    metadata JSONB,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

#### Conversations 테이블

```sql
CREATE TABLE conversations (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID REFERENCES users(id) ON DELETE CASCADE,
    title VARCHAR(500),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

#### Messages 테이블

```sql
CREATE TABLE messages (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    conversation_id UUID REFERENCES conversations(id) ON DELETE CASCADE,
    role VARCHAR(20) NOT NULL, -- 'user' or 'assistant'
    content TEXT NOT NULL,
    sources JSONB, -- 참조된 문서 정보
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

### 3.2 Vector Database (Qdrant)

#### 컬렉션 구조

```python
# Qdrant 컬렉션 설정
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, CollectionInfo

# Qdrant 클라이언트 연결
client = QdrantClient(
    host="localhost",  # 자체 서버
    port=6333,
    prefer_grpc=True
)

# 컬렉션 생성
collection_config = {
    "collection_name": "rag_documents",
    "vectors_config": VectorParams(
        size=1536,  # OpenAI text-embedding-ada-002
        distance=Distance.COSINE
    )
}

client.create_collection(
    collection_name="rag_documents",
    vectors_config=VectorParams(size=1536, distance=Distance.COSINE)
)
```

#### Qdrant Docker 설정

```yaml
# docker-compose.yml
version: "3.8"
services:
  qdrant:
    image: qdrant/qdrant:v1.7.0
    container_name: rag_qdrant
    ports:
      - "6333:6333" # HTTP API
      - "6334:6334" # gRPC API
    volumes:
      - ./qdrant_storage:/qdrant/storage
    environment:
      - QDRANT__SERVICE__HTTP_PORT=6333
      - QDRANT__SERVICE__GRPC_PORT=6334
    restart: unless-stopped
```

#### 벡터 메타데이터 구조

```json
{
  "document_id": "uuid",
  "chunk_index": 0,
  "content": "텍스트 내용",
  "document_title": "문서 제목",
  "document_type": "pdf|docx|txt",
  "page_number": 1,
  "user_id": "uuid"
}
```

## 4. API 설계

### 4.1 인증 API

```typescript
// POST /auth/login
{
  "email": "user@example.com",
  "password": "password"
}

// POST /auth/register
{
  "email": "user@example.com",
  "password": "password",
  "name": "사용자명"
}

// GET /auth/me
// Authorization: Bearer {token}
```

### 4.2 문서 관리 API

```typescript
// POST /documents/upload
// multipart/form-data

// GET /documents
// 사용자의 문서 목록

// DELETE /documents/:id
// 문서 삭제

// GET /documents/:id/status
// 문서 처리 상태 확인
```

### 4.3 채팅 API

```typescript
// POST /conversations
// 새 대화 생성

// GET /conversations
// 대화 목록

// POST /conversations/:id/messages
{
  "content": "질문 내용"
}

// GET /conversations/:id/messages
// 대화 기록
```

### 4.4 RAG Agent API (FastAPI)

```python
# POST /process-document
{
  "file_path": "/path/to/document.pdf",
  "user_id": "uuid"
}

# Response
{
  "document_id": "uuid",
  "status": "processed",
  "embedding_count": 42
}

# POST /query
{
  "question": "질문 내용",
  "user_id": "uuid"
}

# Response
{
  "answer": "답변 내용",
  "sources": [
    {
      "document_id": "uuid",
      "file_path": "/path/to/source.pdf",
      "relevance_score": 0.85
    }
  ],
  "processing_time": 1.2
}
```

## 5. 시스템 플로우

### 5.1 문서 업로드 및 처리

```mermaid
sequenceDiagram
    participant U as User
    participant FE as Frontend
    participant BE as Backend
    participant A as Agent
    participant VDB as Vector DB

    U->>FE: 파일 업로드
    FE->>BE: POST /documents/upload
    BE->>BE: 파일 저장 및 메타데이터 생성
    BE->>A: 문서 처리 요청
    A->>A: 텍스트 추출 및 청킹
    A->>A: 임베딩 생성
    A->>VDB: 벡터 저장
    A->>BE: 처리 완료 알림
    BE->>FE: 상태 업데이트 (WebSocket)
```

### 5.2 질의응답 플로우

```mermaid
sequenceDiagram
    participant U as User
    participant FE as Frontend
    participant BE as Backend
    participant A as Agent
    participant VDB as Vector DB
    participant LLM as LLM Service

    U->>FE: 질문 입력
    FE->>BE: POST /conversations/:id/messages
    BE->>A: 질의 처리 요청
    A->>A: 질문 임베딩
    A->>VDB: 유사도 검색
    VDB->>A: 관련 문서 반환
    A->>A: 컨텍스트 구성
    A->>LLM: 답변 생성 요청
    LLM->>A: 답변 반환
    A->>BE: 답변 및 출처 정보
    BE->>BE: 메시지 저장
    BE->>FE: 실시간 응답 (WebSocket)
```

## 6. 보안 및 성능

### 6.1 보안 고려사항

- **JWT 기반 인증**
- **파일 업로드 검증** (크기, 확장자, 바이러스 스캔)
- **Rate Limiting** (API 호출 제한)
- **CORS 설정**
- **입력 데이터 검증 및 sanitization**

### 6.2 성능 최적화

- **캐싱 전략** (Redis 활용)
- **데이터베이스 인덱싱**
- **벡터 검색 최적화**
- **응답 스트리밍** (Server-Sent Events)
- **파일 압축 및 CDN 활용**

## 7. 모니터링 및 로깅

### 7.1 로깅 전략

- **구조화된 로깅** (JSON 형태)
- **분산 추적** (Correlation ID)
- **에러 모니터링** (Sentry)
- **성능 메트릭** (응답 시간, 처리량)

### 7.2 메트릭 수집

```typescript
// 주요 메트릭
{
  "query_response_time": "ms",
  "document_processing_time": "ms",
  "vector_search_accuracy": "score",
  "user_satisfaction": "rating",
  "system_resource_usage": "percentage"
}
```

## 8. 확장성 고려사항

### 8.1 수평 확장

- **마이크로서비스 아키텍처** 준비
- **로드 밸런서** 구성
- **데이터베이스 샤딩** 계획
- **캐시 클러스터링**

### 8.2 기능 확장

- **다중 언어 지원**
- **다양한 파일 형식 지원**
- **고급 검색 필터**
- **사용자 권한 관리**
- **API 버전 관리**

## 3. RAG 시스템 전체 구조와 Agent 역할

### 3.1 RAG 시스템 3계층 아키텍처

**전체 시스템은 3개 독립적인 서비스로 구성됩니다:**

```
┌──────────────────────┐    ┌───────────────────────┐    ┌───────────────────────┐
│      Frontend        │    │       Backend         │    │        Agent          │
│    (SvelteKit)       │◄──►│      (NestJS)         │◄──►│   (Python/FastAPI)    │
│                      │    │                       │    │                       │
│   USER INTERFACE     │    │   API GATEWAY         │    │   AI LOGIC PROCESSING │
│   - Chat UI          │    │   - Authentication    │    │   - LLM Calls         │
│   - File Upload      │    │   - Data Management   │    │   - Embedding Creation│
│   - Search Results   │    │   - Logging/Monitoring│    │   - Vector Search     │
└──────────────────────┘    └───────────────────────┘    └───────────────────────┘
```

### 3.2 각 컴포넌트의 역할과 책임

#### 3.2.1 Agent (AI 처리 엔진) - 핵심!

**Agent가 담고 있는 것들:**

```python
# Agent의 전체 구조 - 모든 AI 관련 로직이 여기 집중
class RAGAgent:
    """RAG 시스템의 두뇌 역할 - 모든 AI 처리를 담당"""

    def __init__(self):
        # 1. LLM 클라이언트 (텍스트 생성)
        self.gemini_client = self._init_gemini()

        # 2. 임베딩 모델들 (벡터 변환) ← 여기서 임베딩 모델 관리!
        self.embedding_models = {
            "text": SentenceTransformer('all-MiniLM-L6-v2'),
            "multimodal": CLIPModel.from_pretrained("openai/clip-vit-base-patch32"),
            "korean": SentenceTransformer('jhgan/ko-sroberta-multitask')
        }

        # 3. 벡터 데이터베이스 연결
        self.vector_db = QdrantClient("localhost", port=6333)

        # 4. 문서 처리 파이프라인
        self.document_processor = MultiModalDocumentProcessor()

        # 5. OCR 엔진들 (이미지에서 텍스트 추출)
        self.ocr_engines = {
            "tesseract": pytesseract,
            "paddleocr": PaddleOCR(),
            "easyocr": easyocr.Reader(['ko', 'en'])
        }

        # 6. 이미지 전처리 (화질 개선)
        self.image_enhancer = ImageEnhancer()

    def process_document(self, file_path: str):
        """문서 업로드 시 호출 - 벡터화해서 DB에 저장"""
        # 1. 파일 타입에 따른 처리
        if file_path.endswith('.pdf'):
            content = self._process_pdf(file_path)
        elif file_path.endswith(('.jpg', '.png')):
            content = self._process_image(file_path)

        # 2. 임베딩 생성 (Agent 내부에서 처리)
        embeddings = self.embedding_models["text"].encode(content)

        # 3. 벡터 DB에 저장
        self.vector_db.upsert(
            collection_name="documents",
            points=[{"id": uuid.uuid4(), "vector": embeddings, "payload": {"content": content}}]
        )

    def query(self, user_question: str) -> str:
        """사용자 질문 처리 - RAG의 핵심 로직"""
        # 1. 질문을 벡터로 변환 (임베딩)
        question_vector = self.embedding_models["text"].encode([user_question])

        # 2. 유사한 문서 검색 (벡터 유사도)
        search_results = self.vector_db.search(
            collection_name="documents",
            query_vector=question_vector[0],
            limit=5
        )

        # 3. 검색된 컨텍스트 + 질문을 LLM에 전달
        context = "\n".join([result.payload["content"] for result in search_results])

        # 4. Gemini로 최종 답변 생성
        prompt = f"컨텍스트: {context}\n질문: {user_question}\n답변:"
        response = self.gemini_client.generate_content(prompt)

        return response.text
```

#### 3.2.2 Backend (비즈니스 로직)

**Agent와 Frontend를 연결하는 중간 계층:**

```python
# Backend의 역할 - Agent를 호출하고 결과를 관리
@Controller('rag')
class RAGController:
    """NestJS Backend - API 엔드포인트 제공"""

    def __init__(self):
        self.agent_client = HTTPClient("http://agent-service:8000")  # Agent 호출
        self.user_service = UserService()
        self.document_service = DocumentService()

    @Post('upload')
    async def upload_document(self, file: File, user_id: str):
        """파일 업로드 처리"""
        # 1. 사용자 권한 확인
        if not await self.user_service.check_permission(user_id):
            raise UnauthorizedException()

        # 2. 파일 저장
        file_path = await self.save_file(file)

        # 3. Agent에게 문서 처리 요청 (여기서 임베딩 처리됨)
        result = await self.agent_client.post('/process-document', {
            'file_path': file_path,
            'user_id': user_id
        })

        # 4. 메타데이터 DB에 저장
        await self.document_service.save_metadata(file_path, user_id, result)

        return {"status": "success", "document_id": result.document_id}

    @Post('query')
    async def query(self, question: str, user_id: str):
        """사용자 질문 처리"""
        # 1. 사용자 권한 확인
        await self.user_service.validate_user(user_id)

        # 2. Agent에게 질문 전달
        answer = await self.agent_client.post('/query', {
            'question': question,
            'user_id': user_id
        })

        # 3. 대화 이력 저장
        await self.chat_service.save_conversation(user_id, question, answer)

        return {"answer": answer, "timestamp": new Date()}
```

#### 3.2.3 Frontend (사용자 인터페이스)

**사용자가 실제로 보는 화면:**

```svelte
<!-- SvelteKit Frontend - 채팅 인터페이스 -->
<script>
    import { onMount } from 'svelte';

    let messages = [];
    let userInput = '';
    let isLoading = false;

    async function sendMessage() {
        if (!userInput.trim()) return;

        // 사용자 메시지 추가
        messages = [...messages, { type: 'user', content: userInput }];
        const question = userInput;
        userInput = '';
        isLoading = true;

        try {
            // Backend API 호출 (Backend이 Agent 호출)
            const response = await fetch('/api/rag/query', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ question })
            });

            const result = await response.json();

            // AI 응답 추가
            messages = [...messages, { type: 'ai', content: result.answer }];
        } catch (error) {
            messages = [...messages, { type: 'error', content: '오류가 발생했습니다.' }];
        } finally {
            isLoading = false;
        }
    }

    async function uploadFile(event) {
        const file = event.target.files[0];
        if (!file) return;

        const formData = new FormData();
        formData.append('file', file);

        // Backend에 파일 업로드 (Backend이 Agent에게 처리 요청)
        const response = await fetch('/api/rag/upload', {
            method: 'POST',
            body: formData
        });

        const result = await response.json();
        messages = [...messages, { type: 'system', content: `파일 "${file.name}" 업로드 완료` }];
    }
</script>

<div class="chat-container">
    <!-- 파일 업로드 -->
    <input type="file" on:change={uploadFile} accept=".pdf,.jpg,.png" />

    <!-- 채팅 메시지들 -->
    {#each messages as message}
        <div class="message {message.type}">
            {message.content}
        </div>
    {/each}

    <!-- 입력창 -->
    <form on:submit|preventDefault={sendMessage}>
        <input bind:value={userInput} placeholder="질문을 입력하세요..." />
        <button type="submit" disabled={isLoading}>
            {isLoading ? '처리중...' : '전송'}
        </button>
    </form>
</div>
```
