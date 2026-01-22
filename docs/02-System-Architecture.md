# RAG 시스템 전체 아키텍처 설계

## 📊 구현 현황

| 컴포넌트 | 상태 | 설명 |
|---------|------|------|
| **Agent** | ✅ **완료** | Python/FastAPI 기반 RAG 엔진 (완전 동작) |
| **Qdrant** | ✅ **완료** | 벡터 데이터베이스 (문서 임베딩) |
| **MongoDB** | ✅ **운영** | 사용자, 기기, 알람 데이터 저장소 |
| **InfluxDB** | ✅ **운영** | 시계열 데이터 (온습도 센서) |
| **CLI** | ✅ **완료** | 테스트용 커맨드라인 인터페이스 |
| **Frontend** | 📝 계획 | SvelteKit 기반 웹 UI |
| **Backend** | 📝 계획 | NestJS 기반 API 게이트웨이 |

---

## 1. 시스템 개요

### 1.1 목표 아키텍처 (3계층)

```mermaid
graph LR
    subgraph Client["클라이언트"]
        User[사용자]
    end
    
    subgraph Frontend["Frontend Layer<br/>📝 계획"]
        UI[SvelteKit<br/>웹 인터페이스]
        Chat[채팅 UI]
        Upload[파일 업로드]
    end
    
    subgraph Backend["Backend Layer<br/>📝 계획"]
        API[NestJS<br/>API Gateway]
        Auth[인증/권한<br/>JWT]
        FileServ[파일 서비스]
        Meta[메타데이터 관리]
    end
    
    subgraph Agent["Agent Layer<br/>✅ 완료"]
        RAG[RAG Engine]
        LLM[LLM 서비스]
        Embed[임베딩]
        Search[벡터 검색]
        Process[문서 처리]
    end
    
    subgraph Storage["데이터 저장소"]
        Mongo[(MongoDB<br/>✅ 운영)]
        Influx[(InfluxDB<br/>✅ 운영)]
        QD[(Qdrant<br/>✅ 완료)]
    end
    
    User -->|브라우저| UI
    UI -->|HTTP/WS| API
    API -->|인증| Auth
    API -->|메타데이터| Meta
    Meta -->|저장| Mongo
    
    API -->|RAG 요청| RAG
    RAG -->|벡터 검색| Search
    Search -->|쿼리| QD
    RAG -->|DB 쿼리| Mongo
    RAG -->|센서 데이터| Influx
    
    style Agent fill:#d4edda,stroke:#28a745,stroke-width:3px
    style QD fill:#d4edda,stroke:#28a745,stroke-width:3px
    style Mongo fill:#d4edda,stroke:#28a745,stroke-width:3px
    style Influx fill:#d4edda,stroke:#28a745,stroke-width:3px
    style Frontend fill:#fff3cd,stroke:#ffc107,stroke-width:2px
    style Backend fill:#fff3cd,stroke:#ffc107,stroke-width:2px
```

### 1.2 현재 구현 상태

```mermaid
graph TB
    subgraph Current["현재 동작 중인 시스템"]
        CLI[CLI 도구<br/>테스트용]
        
        subgraph AgentNow["Agent Service ✅"]
            direction TB
            FastAPI[FastAPI<br/>웹 서버]
            
            subgraph Core["Core Modules"]
                RAGEngine[RAG Engine]
                
                subgraph LLMModule["llm/"]
                    Gemini[Gemini Service]
                    AnswerGen[Answer Generator]
                    QClass[Question Classifier]
                end
                
                subgraph RetModule["retrieval/"]
                    EmbedMgr[Embedding Manager]
                    VectorStore[Vector Store]
                end
                
                subgraph ProcModule["processing/"]
                    DocProc[Document Processor]
                    TextProc[Text Processor]
                end
                
                subgraph SessModule["session/"]
                    ConvMgr[Conversation Manager]
                    WSMgr[WebSocket Manager]
                end
            end
        end
        
        Qdrant[(Qdrant<br/>Vector DB<br/>✅)]
        
        CLI -->|HTTP API| FastAPI
        FastAPI --> RAGEngine
        RAGEngine --> LLMModule
        RAGEngine --> RetModule
        RAGEngine --> ProcModule
        RAGEngine --> SessModule
        VectorStore --> Qdrant
    end
    
    style Current fill:#e8f5e9
    style AgentNow fill:#c8e6c9
    style Qdrant fill:#a5d6a7
```

### 1.3 주요 컴포넌트 역할

| 계층 | 컴포넌트 | 책임 | 상태 |
|-----|---------|-----|------|
| **Frontend** | SvelteKit | 사용자 인터페이스, 채팅 UI, 파일 업로드 | 📝 계획 |
| **Backend** | NestJS | API 게이트웨이, 인증, 비즈니스 로직, 프록시 | 📝 계획 |
| **Agent** | FastAPI | **하이브리드 RAG**, 문서+DB 통합 쿼리 | ✅ **완료** |
| **MongoDB** | NoSQL | 사용자, 기기, 알람, 대화 메시지 | ✅ **운영** |
| **InfluxDB** | Time-Series | 센서 데이터 (온도, 습도, 이벤트) | ✅ **운영** |
| **Qdrant** | Vector DB | 문서 임베딩, 시맨틱 검색 | ✅ **완료** |

---

## 2. 기술 스택

### 2.1 Frontend (SvelteKit) - 📝 계획

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

**기능:**
- 실시간 채팅 인터페이스
- 파일 드래그 앤 드롭 업로드
- 마크다운 렌더링
- 코드 하이라이팅
- 반응형 디자인

### 2.2 Backend (NestJS) - 📝 계획

```typescript
// 주요 라이브러리
{
  "dependencies": {
    "@nestjs/core": "^10.0.0",
    "@nestjs/common": "^10.0.0",
    "@nestjs/mongoose": "^10.0.0",
    "@nestjs/passport": "^10.0.0",
    "@nestjs/jwt": "^10.1.0",
    "@nestjs/websockets": "^10.0.0",
    "mongoose": "^8.0.0",
    "multer": "^1.4.5",
    "socket.io": "^4.7.0",
    "class-validator": "^0.14.0",
    "class-transformer": "^0.5.1",
    "axios": "^1.6.0"
  }
}
```

**기능:**
- JWT 기반 인증
- 파일 업로드/저장
- Agent API 프록시
- WebSocket 게이트웨이
- 사용자 관리

### 2.3 Agent (Python/FastAPI) - ✅ 완료

```python
# 현재 설치된 라이브러리
fastapi==0.104.1
uvicorn==0.24.0

# RAG Core
sentence-transformers==2.2.2
qdrant-client==1.7.0
google-generativeai==0.3.0

# 문서 처리
pytesseract==0.3.10
opencv-python==4.8.1
torch==2.1.0
transformers==4.35.0
pypdf2==3.0.1
python-docx==1.1.0
pymupdf==1.23.0
paddlepaddle==2.5.1
paddleocr==2.7.0
easyocr==1.7.0

# 데이터베이스 연동 (하이브리드 RAG)
motor==3.3.2  # MongoDB async driver
influxdb-client==1.38.0  # InfluxDB client

pandas==2.1.4
numpy==1.24.3
```

**현재 구현된 기능:**
- ✅ 문서 업로드 및 처리 (PDF, DOCX, TXT)
- ✅ 다중 OCR (Tesseract, PaddleOCR, EasyOCR)
- ✅ 다국어 임베딩 (한국어, 영어, 멀티모달)
- ✅ Gemini LLM 통합
- ✅ Qdrant 벡터 검색
- ✅ 대화 컨텍스트 관리
- ✅ WebSocket 진행률 알림
- ✅ 사용자별 컬렉션 관리
- ✅ **하이브리드 RAG** (NEW!)
  - MongoDB 연동 (사용자, 기기, 알람 데이터)
  - InfluxDB 연동 (시계열 센서 데이터)
  - 통합 쿼리 (문서 + DB 데이터)

---

## 3. 하이브리드 RAG 아키텍처 ⭐

### 3.1 시스템 개념

기존 RAG는 **문서만** 검색했지만, **하이브리드 RAG**는 **문서 + 구조화 데이터 + 시계열 데이터**를 모두 활용합니다.

```mermaid
graph TB
    User[사용자 질문]
    
    subgraph HybridRAG["하이브리드 RAG 엔진"]
        Intent[의도 분류기<br/>Gemini]
        
        subgraph Sources["데이터 소스"]
            DocRAG[문서 RAG<br/>Qdrant]
            MongoDB[(MongoDB<br/>사용자/기기/알람)]
            InfluxDB[(InfluxDB<br/>센서 데이터)]
        end
        
        Integrator[통합 답변 생성<br/>Gemini]
    end
    
    Answer[통합 답변]
    
    User --> Intent
    Intent -->|문서 질문| DocRAG
    Intent -->|DB 질문| MongoDB
    Intent -->|센서 질문| InfluxDB
    Intent -->|복합 질문| DocRAG
    Intent -->|복합 질문| MongoDB  
    Intent -->|복합 질문| InfluxDB
    
    DocRAG --> Integrator
    MongoDB --> Integrator
    InfluxDB --> Integrator
    
    Integrator --> Answer
    
    style HybridRAG fill:#e3f2fd
    style DocRAG fill:#c8e6c9
    style MongoDB fill:#fff9c4
    style InfluxDB fill:#ffe0b2
```

### 3.2 실제 데이터베이스 구조

#### MongoDB 컬렉션 (NoSQL)

```javascript
// users 컬렉션 - 사용자 정보
{
  _id: ObjectId("..."),
  email: "user@example.com",
  name: "홍길동",
  password_hash: "...",
  role: "user",
  devices: ["device123", "device456"],  // 소유 기기 목록
  created_at: ISODate("2026-01-01T00:00:00Z")
}

// devices 컬렉션 - IoT 기기 정보
{
  _id: ObjectId("..."),
  device_id: "device123",
  user_id: ObjectId("..."),
  name: "거실 에어컨",
  type: "air_conditioner",
  model: "AC-2000",
  location: "거실",
  status: "online",
  settings: {
    target_temperature: 25,
    mode: "cool",
    fan_speed: "auto"
  },
  last_seen: ISODate("2026-01-19T14:00:00Z")
}

// alarms 컬렉션 - 알람/이벤트
{
  _id: ObjectId("..."),
  device_id: "device123",
  alarm_type: "high_temperature",
  severity: "warning",  // info, warning, critical
  message: "온도가 30도를 초과했습니다",
  value: 32.5,
  threshold: 30.0,
  timestamp: ISODate("2026-01-19T13:15:00Z"),
  acknowledged: false,
  user_id: ObjectId("...")
}

// conversations 컬렉션 - 대화 메시지
{
  _id: ObjectId("..."),
  user_id: ObjectId("..."),
  title: "에어컨 관련 질문",
  messages: [
    {
      role: "user",
      content: "거실 온도는?",
      timestamp: ISODate("2026-01-19T13:00:00Z")
    },
    {
      role: "assistant",
      content: "거실의 현재 온도는 25.3°C입니다.",
      sources: {
        database: "InfluxDB",
        query_type: "sensor_data"
      },
      timestamp: ISODate("2026-01-19T13:00:01Z")
    }
  ],
  created_at: ISODate("2026-01-19T13:00:00Z"),
  updated_at: ISODate("2026-01-19T13:00:01Z")
}
```

#### InfluxDB 측정값 (Time-Series)

```flux
// temperature 측정값
temperature,device_id=device123,location=거실,user_id=user456 value=25.3 1705661700000000000

// humidity 측정값
humidity,device_id=device123,location=거실 value=55.0 1705661700000000000

// alarms 이벤트
alarms,device_id=device123,type=high_temp,severity=warning message="온도 초과",value=32.5 1705661700000000000
```

#### Qdrant 벡터 (Document Embeddings)

```python
# 이미 구현된 문서 벡터 저장소
{
  "id": "chunk_uuid",
  "v

ector": [0.1, 0.2, ...],  # 768차원 임베딩
  "payload": {
    "document_id": "doc123",
    "content": "에어컨 사용 설명서...",
    "original_filename": "ac_manual.pdf",
    "page": 5,
    "user_id": "user456"
  }
}
```

### 3.3 하이브리드 쿼리 시나리오

#### 시나리오 1: 단순 DB 쿼리

```
질문: "최근 1시간 동안 알람이 있는 기기는?"

처리 과정:
1. Intent: alarm_db, time_range: 1 hour
2. MongoDB 쿼리:
   alarms.find({
     timestamp: { $gte: 1시간 전 },
     acknowledged: false
   })
3. 기기 정보 조회:
   devices.find({ device_id: { $in: [...] } })
4. LLM 답변 생성:
   "최근 1시간 동안 2개 기기에서 알람 발생:
    - 거실 에어컨: 고온 경보 (32.5°C)
    - 안방 제습기: 습도 경보 (65%)"
```

#### 시나리오 2: 시계열 데이터 쿼리

```
질문: "거실 온도가 가장 높았던 시간은?"

처리 과정:
1. Intent: sensor_db, device: "거실", metric: "temperature"
2. MongoDB: devices.find({ location: "거실" })
   → device_id: "device123"
3. InfluxDB 쿼리:
   from(bucket: "sensors")
     |> range(start: -24h)
     |> filter(fn: (r) => r["device_id"] == "device123")
     |> filter(fn: (r) => r["_field"] == "temperature")
     |> max()
4. LLM 답변:
   "거실 온도가 가장 높았던 시간은 오후 3시 15분으로 33.2°C를 기록했습니다."
```

#### 시나리오 3: 하이브리드 쿼리 (문서 + DB + 센서)

```
질문: "거실 에어컨에 알람이 떴는데, 현재 온도와 해결 방법은?"

처리 과정:
1. Intent: hybrid (alarm + sensor + document)

2. 병렬 쿼리:
   a) MongoDB alarms: 최근 알람 조회
      → "고온 경보: 32.5°C"
   
   b) MongoDB devices: 기기 정보
      → {model: "AC-2000", location: "거실"}
   
   c) InfluxDB: 현재 온도
      → 32.5°C
   
   d) Qdrant: 문서 검색 ("에어컨 고온 경보")
      → "AC-2000 매뉴얼 p.45: 필터 청소..."

3. 통합 컨텍스트 구성:
   ```
   알람: 고온 경보 (32.5°C, 설정 30°C)
   기기: 거실 AC-2000
   현재: 32.5°C
   매뉴얼: 온도 상승 시 필터 청소, 냉매 점검
   ```

4. LLM 통합 답변:
   "거실 에어컨(AC-2000)에서 고온 경보가 발생했습니다.
    
    **현재 상태:**
    - 현재 온도: 32.5°C
    - 설정 온도: 30°C
    - 초과: +2.5°C
    
    **해결 방법:**
    1. 즉시 조치: 설정 온도를 25°C로 낮추기
    2. 점검: 필터 청소 (매뉴얼 p.45)
    3. 확인: 냉매 부족 여부 점검
    4. 지속 시: A/S 센터(1588-xxxx) 문의
    
    참고: [AC-2000 사용 설명서 p.45]"
```

### 3.4 Agent 구현 아키텍처

```python
# agent/app/core/hybrid_rag_engine.py

class HybridRAGEngine:
    """문서 + DB + 시계열 통합 RAG"""
    
    def __init__(self):
        # 기존 문서 RAG
        self.document_rag = rag_engine
        
        # DB 커넥터
        self.mongo = MongoDBConnector()
        self.influx = InfluxDBConnector()
        
        # LLM
        self.gemini = gemini_service
    
    async def query(self, question: str, user_id: str):
        """통합 쿼리 처리"""
        
        # 1. 의도 분류
        intent = await self._classify_intent(question)
        
        # 2. 데이터 소스별 라우팅
        if intent["type"] == "document":
            return await self.document_rag.query(question, user_id)
        
        elif intent["type"] == "database":
            return await self._query_databases(question, intent)
        
        elif intent["type"] == "timeseries":
            return await self._query_timeseries(question, intent)
        
        elif intent["type"] == "hybrid":
            return await self._hybrid_query(question, user_id, intent)
    
    async def _classify_intent(self, question: str):
        """질문 의도 분류"""
        prompt = f"""
        질문: {question}
        
        데이터 소스 분류:
        - document: 매뉴얼, 사용법, 가이드
        - database: 사용자 정보, 기기 정보
        - alarm: 알람, 이벤트
        - timeseries: 온도, 습도, 시간별 데이터
        - hybrid: 여러 소스 조합
        
        JSON 응답:
        {{
          "type": "...",
          "entities": ["추출된 엔티티"],
          "time_range": {{"hours": 1}},
          "filters": {{}}
        }}
        """
        
        response = await self.gemini.generate_content(prompt)
        return json.loads(response.text)
    
    async def _query_databases(self, question: str, intent: dict):
        """MongoDB 쿼리"""
        
        # 알람 쿼리
        if "alarm" in intent["type"]:
            hours = intent.get("time_range", {}).get("hours", 1)
            alarms = await self.mongo.get_recent_alarms(hours)
            
            # 알람 있는 기기 정보 추가
            if alarms:
                device_ids = [a["device_id"] for a in alarms]
                devices = await self.mongo.get_devices_by_ids(device_ids)
                
                for alarm in alarms:
                    alarm["device_info"] = next(
                        (d for d in devices if d["device_id"] == alarm["device_id"]),
                        {}
                    )
        
        # LLM 답변 생성
        answer = await self._generate_answer_from_data(question, alarms)
        
        return {
            "answer": answer,
            "sources": {"alarms": alarms},
            "type": "database"
        }
    
    async def _query_timeseries(self, question: str, intent: dict):
        """InfluxDB 시계열 쿼리"""
        
        device_id = intent.get("device_id")
        metric = intent.get("metric", "temperature")
        hours = intent.get("time_range", {}).get("hours", 24)
        
        # InfluxDB 쿼리
        if metric == "temperature":
            data = await self.influx.get_temperature_range(device_id, hours)
        elif metric == "humidity":
            data = await self.influx.get_humidity_range(device_id, hours)
        
        # LLM 답변
        answer = await self._generate_answer_from_timeseries(question, data)
        
        return {
            "answer": answer,
            "sources": {"timeseries": data},
            "type": "timeseries"
        }
    
    async def _hybrid_query(self, question: str, user_id: str, intent: dict):
        """통합 쿼리 (문서 + DB + 센서)"""
        
        # 병렬 쿼리
        results = await asyncio.gather(
            self.document_rag.query(question, user_id),  # 문서
            self._query_databases(question, intent),      # DB
            self._query_timeseries(question, intent),      # 센서
            return_exceptions=True
        )
        
        # 결과 통합
        combined_context = self._combine_results(results)
        
        # LLM 최종 답변
        answer = await self.gemini.generate_content(
            f"다음 정보를 종합하여 답변:\n\n{combined_context}\n\n질문: {question}"
        )
        
        return {
            "answer": answer.text,
            "sources": {
                "documents": results[0].get("sources", []),
                "database": results[1].get("sources", {}),
                "timeseries": results[2].get("sources", {})
            },
            "type": "hybrid"
        }
```

### 3.5 데이터 흐름

```mermaid
sequenceDiagram
    participant User
    participant FE as Frontend
    participant BE as Backend<br/>(프록시)
    participant Agent as Hybrid RAG<br/>Agent
    participant Intent as Intent<br/>Classifier
    participant Mongo as MongoDB
    participant Influx as InfluxDB
    participant Qdrant as Qdrant
    participant Gemini as Gemini LLM

    User->>FE: "최근 1시간 알람 + 해결법"
    FE->>BE: POST /rag/query
    BE->>BE: JWT 검증
    BE->>Agent: 질문 전달
    
    Agent->>Intent: 의도 분류
    Intent-->>Agent: hybrid (alarm + document)
    
    par 병렬 쿼리
        Agent->>Mongo: 알람 조회
        Mongo-->>Agent: 2개 알람
        
        Agent->>Mongo: 기기 정보
        Mongo-->>Agent: 기기 상세
        
        Agent->>Influx: 현재 센서 데이터
        Influx-->>Agent: 온도 32.5°C
        
        Agent->>Qdrant: 문서 검색
        Qdrant-->>Agent: 매뉴얼 p.45
    end
    
    Agent->>Gemini: 통합 컨텍스트
    Gemini-->>Agent: 종합 답변
    
    Agent-->>BE: 결과
    BE->>BE: 사용량 기록
    BE-->>FE: 답변
    FE-->>User: 표시
```

### 3.6 실제 사용 예시

#### 예시 1: 온도 이상 알람 + 매뉴얼

**질문:** "에어컨 온도 알람이 떴는데 무슨 문제야?"

**처리:**
1. MongoDB: 최근 알람 → "고온 경보 32.5°C"
2. InfluxDB: 시간별 온도 → "30°C에서 점진적 상승"
3. Qdrant: "에어컨 고온 문제" → 매뉴얼 검색

**답변:**
```
에어컨에서 고온 경보가 발생했습니다.

현재 상태:
- 현재 온도: 32.5°C (설정: 30°C)
- 추이: 지난 1시간 동안 2.5°C 상승

원인 분석:
1. 필터 막힘 가능성 (가장 흔한 원인)
2. 냉매 부족
3. 실외기 과부하

즉시 조치:
1. 에어컨을 끄고 30분 대기
2. 필터 청소 (매뉴얼 p.45 참조)
3. 재가동 후 30분 모니터링

문제 지속 시: 1588-xxxx (A/S 센터)

참고 문서: [AC-2000 매뉴얼 p.45-47]
```

#### 예시 2: 사용자별 기기 통계

**질문:** "홍길동의 기기 중 온도가 30도 이상인 것은?"

**처리:**
1. MongoDB users: "홍길동" → user_id
2. MongoDB devices: user_id → 기기 목록 (5개)
3. InfluxDB: 각 기기 최근 온도
4. 필터링: 30도 이상만

**답변:**
```
홍길동님의 기기 중 온도가 30도 이상인 기기는 2개입니다:

1. 거실 에어컨 (device123)
   - 현재: 32.5°C
   - 상태: 고온 경보
   - 위치: 거실

2. 주방 센서 (device456)
   - 현재: 31.2°C
   - 상태: 정상
   - 위치: 주방

나머지 3개 기기는 정상 온도 범위입니다.
```

#### 예시 3: 시간대별 분석

**질문:** "지난 24시간 중 가장 더웠던 시간과 그때 무슨 기기가 켜져있었어?"

**처리:**
1. InfluxDB: 24시간 온도 데이터 → max() 
2. MongoDB: 해당 시간대 기기 상태 로그
3. LLM: 상관관계 분석

**답변:**
```
지난 24시간 중 가장 더웠던 시간은:

오후 3시 15분 - 33.2°C

그 시간 작동 중이던 기기:
1. 거실 에어컨 (냉방 모드, 25°C 목표)
2. 안방 선풍기 (강풍 모드)
3. 제습기 OFF

분석:
- 외부 온도가 높았던 시간 (한낮)
- 에어컨이 작동 중이었으나 외부 열기로 인해 온도 상승
- 커튼 차단 권장

권장 사항:
- 오후 2-4시: 블라인드 차단
- 에어컨 설정 온도 23°C로 조정
```

---

## 4. 데이터베이스 상세 설계

### 4.1 MongoDB (운영 중) - ✅

MongoDB는 **Section 3.2**에서 자세히 문서화되어 있습니다.

**주요 컬렉션:**
- `users` - 사용자 정보
- `devices` - IoT 기기 정보
- `alarms` - 알람/이벤트
- `conversations` - 대화 메시지

**👉 자세한 스키마는 Section 3.2 "실제 데이터베이스 구조" 참조**

### 4.2 InfluxDB (운영 중) - ✅

InfluxDB는 **Section 3.2**에서 자세히 문서화되어 있습니다.

**주요 측정값:**
- `temperature` - 온도 센서 데이터
- `humidity` - 습도 센서 데이터
- `alarms` - 이벤트 로그

**👉 자세한 스키마는 Section 3.2 "실제 데이터베이스 구조" 참조

### 4.3 Vector Database (Qdrant) - ✅ 완료

#### 컬렉션 구조

```python
# 현재 구현된 Qdrant 설정
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams

# Docker로 실행 중
client = QdrantClient(
    host="localhost",
    port=6333
)

# 사용자별 컬렉션 자동 생성
# 예: documents_user123, documents_testuser
collection_config = VectorParams(
    size=768,  # jhgan/ko-sroberta-multitask 임베딩 차원
    distance=Distance.COSINE
)
```

#### Docker Compose 설정

```yaml
# docker/docker-compose.yml - 현재 사용 중
version: "3.8"
services:
  qdrant:
    image: qdrant/qdrant:v1.7.0
    container_name: rag_qdrant
    ports:
      - "6333:6333" # HTTP API
      - "6334:6334" # gRPC API
    volumes:
      - ./qdrant/storage:/qdrant/storage
    environment:
      - QDRANT__SERVICE__HTTP_PORT=6333
      - QDRANT__SERVICE__GRPC_PORT=6334
    restart: unless-stopped
```

#### 벡터 메타데이터 구조

```json
{
  "document_id": "uuid",
  "original_filename": "report.pdf",
  "file_type": "pdf",
  "chunk_index": 0,
  "page_number": 1,
  "user_id": "test_user",
  "content_type": "text",
  "created_at": "2026-01-19T12:00:00"
}
```

---

## 4. API 설계

### 4.1 Backend API (NestJS) - 📝 계획

#### 인증 API

```typescript
// POST /auth/register
{
  "email": "user@example.com",
  "password": "securepassword",
  "name": "사용자명"
}

// POST /auth/login
{
  "email": "user@example.com",
  "password": "securepassword"
}
// Response: { "access_token": "jwt_token", "user": {...} }

// GET /auth/me
// Headers: Authorization: Bearer {token}
// Response: { "id": "uuid", "email": "...", "name": "..." }
```

#### 문서 관리 API

```typescript
// POST /documents/upload
// Content-Type: multipart/form-data
// Body: file, userId

// GET /documents
// Query: userId
// Response: [{ id, title, status, created_at, ... }]

// DELETE /documents/:id
// Response: { success: true }

// GET /documents/:id/status
// Response: { status: "processing", progress: 75 }
```

#### 채팅 API

```typescript
// POST /conversations
// Body: { userId }
// Response: { id: "uuid", title: null, created_at: "..." }

// GET /conversations
// Query: userId
// Response: [{ id, title, created_at, message_count }]

// POST /conversations/:id/messages
// Body: { content: "질문 내용" }
// Response: { id, role: "assistant", content: "답변", sources: [...] }

// GET /conversations/:id/messages
// Response: [{ id, role, content, created_at, sources }]
```

### 4.2 Agent API (FastAPI) - ✅ 완료

#### 문서 처리 API

```python
# POST /documents/upload
{
  "file": "binary",
  "user_id": "string",
  "document_type": "optional string"
}

# Response
{
  "document_id": "uuid",
  "filename": "report.pdf",
  "status": "processing",
  "message": "파일 업로드 완료",
  "processing_time": 1.2
}

# GET /documents/upload/{document_id}/status
# Response
{
  "document_id": "uuid",
  "filename": "report.pdf",
  "status": "completed",
  "current_step": "임베딩 생성",
  "overall_progress": 100.0,
  "result_data": {
    "text_chunks": 42,
    "image_chunks": 5,
    "total_embeddings": 47
  }
}
```

#### 질의응답 API

```python
# POST /query/search
{
  "question": "질문 내용",
  "user_id": "string",
  "conversation_id": "optional string",
  "max_results": 5,
  "score_threshold": 0.7
}

# Response
{
  "answer": "답변 내용...",
  "sources": [
    {
      "document_id": "uuid",
      "content": "관련 문서 내용",
      "score": 0.85,
      "metadata": {
        "original_filename": "report.pdf",
        "page": 3
      }
    }
  ],
  "confidence": 0.82,
  "processing_time": 2.3
}
```

#### 헬스 체크 API

```python
# GET /health
# Response
{
  "status": "healthy",
  "components": {
    "rag_engine": "healthy",
    "vector_db": "connected",
    "embedding_model": "loaded",
    "gemini_service": "initialized"
  }
}
```

---

## 5. 시스템 플로우

### 5.1 문서 업로드 및 처리

```mermaid
sequenceDiagram
    participant User
    participant FE as Frontend<br/>(SvelteKit)
    participant BE as Backend<br/>(NestJS)
    participant Agent as Agent<br/>(FastAPI)
    participant PG as PostgreSQL
    participant VDB as Qdrant

    User->>FE: 파일 선택 & 업로드
    FE->>BE: POST /documents/upload<br/>(multipart/form-data)
    
    BE->>BE: 파일 검증<br/>(크기, 타입)
    BE->>BE: 파일 저장<br/>(로컬/S3)
    BE->>PG: 문서 메타데이터 생성<br/>(status: processing)
    
    BE->>Agent: POST /documents/upload<br/>(file, user_id)
    Agent-->>BE: document_id, status
    BE-->>FE: { document_id, status }
    
    Note over FE,BE: WebSocket 연결
    FE->>BE: WS: /documents/progress/:id
    BE->>Agent: WS: /ws/progress/:id
    
    Note over Agent,VDB: 백그라운드 처리 시작
    
    Agent->>Agent: 텍스트 추출 (PDF, OCR)
    Agent-->>BE: Progress: 30%
    BE-->>FE: Progress: 30%
    
    Agent->>Agent: 텍스트 청킹
    Agent-->>BE: Progress: 50%  
    BE-->>FE: Progress: 50%
    
    Agent->>Agent: 임베딩 생성<br/>(SentenceTransformer)
    Agent-->>BE: Progress: 80%
    BE-->>FE: Progress: 80%
    
    Agent->>VDB: 벡터 저장
    VDB-->>Agent: 저장 완료
    
    Agent-->>BE: 처리 완료<br/>{ text_chunks, image_chunks }
    BE->>PG: 문서 상태 업데이트<br/>(status: completed)
    BE-->>FE: 완료 알림
    FE-->>User: "문서 처리 완료" 메시지
```

### 5.2 질의응답 플로우

```mermaid
sequenceDiagram
    participant User
    participant FE as Frontend
    participant BE as Backend
    participant PG as PostgreSQL
    participant Agent as Agent
    participant VDB as Qdrant
    participant LLM as Gemini

    User->>FE: 질문 입력
    FE->>BE: POST /conversations/:id/messages<br/>{ content: "질문" }
    
    BE->>PG: 메시지 저장<br/>(role: user)
    
    BE->>Agent: POST /query/search<br/>{ question, user_id }
    
    Note over Agent: RAG 파이프라인 시작
    
    Agent->>Agent: 대화 컨텍스트 분석
    Agent->>Agent: 질문 임베딩 생성
    
    Agent->>VDB: 벡터 유사도 검색<br/>(top 5)
    VDB-->>Agent: 관련 문서 청크 반환
    
    Agent->>Agent: 컨텍스트 구성
    
    Agent->>LLM: generate_content<br/>(context + question)
    LLM-->>Agent: 답변 생성
    
    Agent->>Agent: 답변 후처리<br/>(중복 제거 등)
    Agent-->>BE: { answer, sources, confidence }
    
    BE->>PG: 메시지 저장<br/>(role: assistant)
    BE-->>FE: 답변 반환
    
    FE-->>User: 답변 표시<br/>(스트리밍 가능)
```

### 5.3 사용자 인증 플로우

```mermaid
sequenceDiagram
    participant User
    participant FE as Frontend
    participant BE as Backend
    participant PG as PostgreSQL
    participant Agent as Agent

    User->>FE: 로그인 폼 입력
    FE->>BE: POST /auth/login<br/>{ email, password }
    
    BE->>PG: 사용자 조회
    PG-->>BE: User record
    
    BE->>BE: 비밀번호 검증<br/>(bcrypt)
    
    alt 인증 성공
        BE->>BE: JWT 토큰 생성
        BE-->>FE: { access_token, user }
        FE->>FE: 토큰 저장<br/>(localStorage)
        FE-->>User: 대시보드로 이동
    else 인증 실패
        BE-->>FE: { error: "Invalid credentials" }
        FE-->>User: 에러 메시지 표시
    end
    
    Note over FE,Agent: 이후 모든 요청
    
    FE->>BE: 요청 + Authorization header
    BE->>BE: JWT 검증
    BE->>Agent: Agent API 호출
    Agent-->>BE: 응답
    BE-->>FE: 응답
```

---

## 6. Agent 내부 아키텍처 - ✅ 완료

### 6.1 Core 모듈 구조

```mermaid
graph TB
    subgraph CoreModules["app/core/"]
        Config[config.py<br/>전역 설정]
        RAGEngine[rag_engine.py<br/>메인 오케스트레이터]
        
        subgraph LLM["llm/ - LLM 서비스"]
            Gemini[gemini_service.py<br/>Google Gemini API]
            AnswerGen[answer_generator.py<br/>답변 생성 전략]
            QClass[question_classifier.py<br/>질문 분류]
        end
        
        subgraph Retrieval["retrieval/ - 검색"]
            EmbedMgr[embedding_manager.py<br/>임베딩 모델 관리]
            VStore[vector_store.py<br/>Qdrant 인터페이스]
        end
        
        subgraph Processing["processing/ - 처리"]
            DocProc[document_processor.py<br/>문서 파싱]
            TextProc[text_processor.py<br/>텍스트 처리]
        end
        
        subgraph Session["session/ - 세션"]
            ConvMgr[conversation_manager.py<br/>대화 컨텍스트]
            WSMgr[websocket_manager.py<br/>WebSocket 통신]
        end
    end
    
    RAGEngine --> LLM
    RAGEngine --> Retrieval
    RAGEngine --> Processing
    RAGEngine --> Session
    
    EmbedMgr -.임베딩 모델.-> Korean["jhgan/ko-sroberta-multitask"]
    EmbedMgr -.임베딩 모델.-> English["all-MiniLM-L6-v2"]
    
    style CoreModules fill:#e8f5e9
    style LLM fill:#fff9c4
    style Retrieval fill:#e1f5fe
    style Processing fill:#f3e5f5
    style Session fill:#ffe0b2
```

### 6.2 RAG 파이프라인

```mermaid
flowchart LR
    Start([사용자 질문])
    
    subgraph Context["1. 컨텍스트 분석"]
        ConvHist[대화 히스토리 조회]
        QuestionEnh[질문 보완]
    end
    
    subgraph Classify["2. 질문 분류"]
        MetaCheck{메타 질문?}
        DirectAnswer[직접 답변]
    end
    
    subgraph Embed["3. 임베딩"]
        EmbedQ[질문 벡터화]
    end
    
    subgraph Search["4. 검색"]
        VectorSearch[벡터 유사도 검색]
        RankResults[결과 랭킹]
    end
    
    subgraph Generate["5. 답변 생성"]
        BuildContext[컨텍스트 구성]
        CallLLM[LLM 호출]
        PostProcess[후처리]
    end
    
    End([답변 반환])
    
    Start --> ConvHist
    ConvHist --> QuestionEnh
    QuestionEnh --> MetaCheck
    
    MetaCheck -->|Yes| DirectAnswer --> End
    MetaCheck -->|No| EmbedQ
    
    EmbedQ --> VectorSearch
    VectorSearch --> RankResults
    RankResults --> BuildContext
    BuildContext --> CallLLM
    CallLLM --> PostProcess
    PostProcess --> End
    
    style Context fill:#e3f2fd
    style Classify fill:#fff3e0
    style Embed fill:#f3e5f5
    style Search fill:#e8f5e9
    style Generate fill:#fce4ec
```

---

## 7. 보안 및 성능

### 7.1 보안 고려사항

#### 인증 및 권한

```typescript
// Backend JWT 전략
@Injectable()
export class JwtStrategy extends PassportStrategy(Strategy) {
  constructor() {
    super({
      jwtFromRequest: ExtractJwt.fromAuthHeaderAsBearerToken(),
      secretOrKey: process.env.JWT_SECRET,
    });
  }

  async validate(payload: any) {
    return { userId: payload.sub, email: payload.email };
  }
}

// 권한 가드
@UseGuards(JwtAuthGuard)
@Controller('documents')
export class DocumentsController {
  // ...
}
```

#### 파일 업로드 보안

```typescript
// 파일 검증 파이프
@Injectable()
export class FileValidationPipe implements PipeTransform {
  private readonly ALLOWED_TYPES = [
    'application/pdf',
    'application/msword',
    'text/plain',
  ];
  
  private readonly MAX_SIZE = 50 * 1024 * 1024; // 50MB

  transform(value: any) {
    if (!this.ALLOWED_TYPES.includes(value.mimetype)) {
      throw new BadRequestException('허용되지 않은 파일 타입');
    }
    
    if (value.size > this.MAX_SIZE) {
      throw new BadRequestException('파일 크기 초과');
    }
    
    return value;
  }
}
```

#### Rate Limiting

```typescript
// Backend Rate Limiter
import { ThrottlerModule } from '@nestjs/throttler';

@Module({
  imports: [
    ThrottlerModule.forRoot({
      ttl: 60,
      limit: 10, // 분당 10회
    }),
  ],
})
export class AppModule {}
```

### 7.2 성능 최적화

#### 캐싱 전략

```typescript
// Backend Redis 캐싱
@Injectable()
export class CacheService {
  constructor(private cacheManager: Cache) {}

  @Cacheable('documents', 300) // 5분 캐시
  async getDocument(id: string) {
    // ...
  }
}
```

```python
# Agent 임베딩 캐싱 (현재 구현됨)
from functools import lru_cache

@lru_cache(maxsize=5000)
def _cached_encode(text_hash: str, model_key: str) -> tuple:
    # 자주 사용되는 텍스트 임베딩 캐시
    pass
```

#### 데이터베이스 최적화

```sql
-- 인덱스 생성
CREATE INDEX idx_documents_user_status ON documents(user_id, status);
CREATE INDEX idx_messages_conversation_time ON messages(conversation_id, created_at DESC);

-- 파티셔닝 (대용량 메시지)
CREATE TABLE messages (
    -- ...
) PARTITION BY RANGE (created_at);

CREATE TABLE messages_2026_01 PARTITION OF messages
    FOR VALUES FROM ('2026-01-01') TO ('2026-02-01');
```

#### 벡터 검색 최적화

```python
# Qdrant HNSW 인덱스 파라미터 (현재 설정)
from qdrant_client.models import HnswConfigDiff

hnsw_config = HnswConfigDiff(
    m=16,  # 연결 수
    ef_construct=100,  # 구축 시 탐색 범위
    full_scan_threshold=10000
)
```

---

## 8. 배포 아키텍처

### 8.1 개발 환경

```mermaid
graph TB
    Dev[개발자 로컬]
    
    subgraph DevStack["개발 환경"]
        FEDev[Frontend:3000<br/>npm run dev]
        BEDev[Backend:3001<br/>npm run start:dev]
        AgentDev[Agent:8000<br/>uvicorn --reload]
        PGDev[(PostgreSQL:5432<br/>Docker)]
        QDDev[(Qdrant:6333<br/>Docker)]
    end
    
    Dev --> FEDev
    FEDev --> BEDev
    BEDev --> AgentDev
    BEDev --> PGDev
    AgentDev --> QDDev
```

### 8.2 프로덕션 환경

```mermaid
graph TB
    Internet([인터넷])
    
    subgraph Cloud["클라우드 (AWS/GCP)"]
        LB[로드 밸런서]
        
        subgraph Frontend["Frontend Tier"]
            FE1[SvelteKit<br/>Instance 1]
            FE2[SvelteKit<br/>Instance 2]
        end
        
        subgraph Backend["Backend Tier"]
            BE1[NestJS<br/>Instance 1]
            BE2[NestJS<br/>Instance 2]
        end
        
        subgraph Agent["Agent Tier"]
            A1[FastAPI<br/>Instance 1]
            A2[FastAPI<br/>Instance 2]
        end
        
        subgraph Data["Data Tier"]
            PG[(PostgreSQL<br/>RDS)]
            QD[(Qdrant<br/>Cluster)]
            Redis[(Redis<br/>ElastiCache)]
        end
    end
    
    Internet --> LB
    LB --> FE1
    LB --> FE2
    
    FE1 --> BE1
    FE2 --> BE2
    
    BE1 --> A1
    BE2 --> A2
    
    BE1 --> PG
    BE2 --> PG
    BE1 --> Redis
    BE2 --> Redis
    
    A1 --> QD
    A2 --> QD
    
    style Frontend fill:#e3f2fd
    style Backend fill:#fff3e0
    style Agent fill:#e8f5e9
    style Data fill:#fce4ec
```

### 8.3 Docker Compose (전체 스택)

```yaml
version: "3.8"

services:
  # Frontend
  frontend:
    build: ./frontend
    ports:
      - "3000:3000"
    environment:
      - BACKEND_URL=http://backend:3001
    depends_on:
      - backend

  # Backend
  backend:
    build: ./backend
    ports:
      - "3001:3001"
    environment:
      - DATABASE_URL=postgresql://user:pass@postgres:5432/ragdb
      - AGENT_URL=http://agent:8000
      - REDIS_URL=redis://redis:6379
      - JWT_SECRET=${JWT_SECRET}
    depends_on:
      - postgres
      - redis
      - agent

  # Agent
  agent:
    build: ./agent
    ports:
      - "8000:8000"
    environment:
      - QDRANT_HOST=qdrant
      - QDRANT_PORT=6333
      - GEMINI_API_KEY=${GEMINI_API_KEY}
    volumes:
      - ./agent/uploads:/app/uploads
    depends_on:
      - qdrant

  # PostgreSQL
  postgres:
    image: postgres:15
    ports:
      - "5432:5432"
    environment:
      - POSTGRES_USER=user
      - POSTGRES_PASSWORD=pass
      - POSTGRES_DB=ragdb
    volumes:
      - postgres_data:/var/lib/postgresql/data

  # Qdrant
  qdrant:
    image: qdrant/qdrant:v1.7.0
    ports:
      - "6333:6333"
      - "6334:6334"
    volumes:
      - qdrant_data:/qdrant/storage

  # Redis
  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"

volumes:
  postgres_data:
  qdrant_data:
```

---

## 9. 개발 로드맵

### Phase 1: Agent 완성 ✅ (완료)
- [x] FastAPI 기본 구조
- [x] 문서 처리 파이프라인
- [x] 임베딩 생성
- [x] Qdrant 통합
- [x] Gemini LLM 통합
- [x] WebSocket 진행률
- [x] 대화 컨텍스트 관리
- [x] Core 모듈 리팩토링

### Phase 2: Backend 개발 📝 (다음 단계)
- [ ] NestJS 프로젝트 초기화
- [ ] MongoDB 연동 (@nestjs/mongoose)
- [ ] JWT 인증 구현
- [ ] 파일 업로드 서비스
- [ ] Agent API 프록시
- [ ] WebSocket 게이트웨이
- [ ] 사용자 관리 CRUD

### Phase 3: Frontend 개발 📝
- [ ] SvelteKit 프로젝트 초기화
- [ ] 로그인/회원가입 UI
- [ ] 채팅 인터페이스
- [ ] 파일 업로드 UI
- [ ] 문서 관리 대시보드
- [ ] 실시간 진행률 표시
- [ ] 마크다운 렌더링

### Phase 4: 통합 및 테스트 📝
- [ ] E2E 통합 테스트
- [ ] 성능 최적화
- [ ] 보안 강화
- [ ] 배포 자동화
- [ ] 모니터링 설정

---

## 10. 개선 권장사항

### 10.1 현재 Agent의 강점
✅ **독립적으로 완전히 작동** - Backend 없이도 사용 가능  
✅ **모듈화된 구조** - llm, retrieval, processing, session으로 명확히 분리  
✅ **프로덕션 레디** - 에러 핸들링, 로깅, 타입 힌팅 완비  
✅ **멀티 사용자 지원** - user_id 기반 컬렉션 관리  

### 10.2 Backend 개발 시 권장사항

#### Option A: 간소화된 구조 (추천)
**Agent를 직접 노출하지 않고 Backend를 얇은 레이어로**

```
Frontend → Backend → Agent
         ↓
    PostgreSQL (사용자/메타데이터만)
```

**Backend의 역할을 최소화:**
- 인증/권한만 담당
- 파일 저장 (S3 등)
- 메타데이터 관리
- **Agent API를 그대로 프록시**

**장점:**
- Agent 코드 재사용
- 개발 시간 단축
- 유지보수 간소화

#### Option B: Full Backend Logic
**Backend에 비즈니스 로직 추가**

```
Frontend → Backend(복잡한 로직) → Agent(AI만)
         ↓
    PostgreSQL + Redis + 기타
```

**Backend의 역할 확대:**
- 복잡한 권한 관리
- 사용량 추적/빌링
- 여러 Agent 인스턴스 로드밸런싱
- 멀티 테넌시

**권장 상황:**
- 엔터프라이즈 고객 대상
- 복잡한 권한 체계 필요
- 여러 Agent 버전 동시 운영

### 10.3 데이터베이스 전략

**현재 상황:** 
- ✅ MongoDB: 사용자, 기기, 알람, 대화 메시지
- ✅ InfluxDB: 시계열 센서 데이터
- ✅ Qdrant: 문서 임베딩

**MongoDB 활용:**

| 데이터 타입 | MongoDB | 이유 |
|-----------|---------|-----|
| 사용자 정보 | ✅ 저장 중 | 유연한 스키마 |
| 기기 정보 | ✅ 저장 중 | 동적 설정값 |
| 알람/이벤트 | ✅ 저장 중 | 빠른 쓰기 |
| 대화 메시지 | ✅ 저장 중 | 임베디드 문서 |
| 문서 메타 | Qdrant | 벡터와 함께 |

**InfluxDB 활용:**

| 데이터 타입 | InfluxDB | 이유 |
|-----------|----------|-----|
| 온도 데이터 | ✅ 저장 중 | 시계열 최적화 |
| 습도 데이터 | ✅ 저장 중 | 집계 쿼리 |
| 이벤트 로그 | ✅ 저장 중 | 시간 범위 쿼리 |

**권장:**
- MongoDB + InfluxDB 조합 유지
- Qdrant는 문서 전용
- Redis 추가 고려 (세션 캐싱)

### 10.4 Agent API 개선 제안

현재 Agent는 잘 동작하지만, Backend 통합을 위해 추가하면 좋을 것들:

```python
# 1. 사용자 격리 강화
@router.get("/documents/list")
async def list_user_documents(
    user_id: str,
    skip: int = 0,
    limit: int = 10
):
    # 페이지네이션 추가
    pass

# 2. 통계 API 추가
@router.get("/stats/user/{user_id}")
async def get_user_stats(user_id: str):
    return {
        "total_documents": 42,
        "total_queries": 156,
        "avg_confidence": 0.82
    }

# 3. Bulk 작업 지원
@router.post("/documents/bulk-delete")
async def bulk_delete_documents(
    document_ids: List[str],
    user_id: str
):
    # 여러 문서 한번에 삭제
    pass
```

---

## 11. 요약

### 현재 상태
- ✅ **Agent**: 완전히 작동하는 하이브리드 RAG 시스템
- ✅ **MongoDB**: 사용자, 기기, 알람, 대화 저장
- ✅ **InfluxDB**: 시계열 센서 데이터
- ✅ **Qdrant**: 벡터 데이터베이스 운영 중
- ✅ **CLI**: 테스트 및 개발용 도구

### 다음 단계 (우선순위)
1. **하이브리드 RAG 구현** - MongoDB/InfluxDB 커넥터
2. **Backend (NestJS)** - 인증, API 게이트웨이, MongoDB 연동
3. **Frontend (SvelteKit)** - 사용자 인터페이스

### 핵심 권장사항
1. **Backend를 얇게** - Agent를 최대한 재사용
2. **MongoDB + InfluxDB 유지** - 현재 인프라 활용
3. **Agent는 그대로 유지** - 이미 잘 작동하는 코드베이스
4. **보안 우선** - JWT, Rate Limiting 먼저 구현

### 시스템 아키텍처 요약

```
Frontend (계획) → Backend (계획) → Agent (완료)
                       ↓               ↓
                   MongoDB       InfluxDB + Qdrant
                   (운영 중)        (운영 중)
```

이 문서의 아키텍처대로 개발하면 **문서 검색 + IoT 데이터 통합**이 가능한 확장 가능하고 유지보수하기 좋은 시스템이 될 것입니다! 🚀
