# Frontend (SvelteKit) 개발 프로세스 및 계획

## 📋 개발 일정 (5일)

### Day 1: 프로젝트 초기 설정 및 기본 구조

- [ ] SvelteKit 프로젝트 초기화
- [ ] TailwindCSS + DaisyUI 설정
- [ ] 기본 라우팅 구조 생성
- [ ] 레이아웃 및 네비게이션 컴포넌트

### Day 2: 인증 시스템 및 상태 관리

- [ ] 로그인/회원가입 페이지
- [ ] JWT 토큰 관리
- [ ] 사용자 상태 관리 (Svelte Store)
- [ ] 보호된 라우트 구현

### Day 3: 문서 관리 UI

- [ ] 파일 업로드 컴포넌트
- [ ] 문서 목록 및 상태 표시
- [ ] 드래그 앤 드롭 기능
- [ ] 처리 상태 실시간 업데이트

### Day 4: 채팅 인터페이스

- [ ] 채팅 UI 컴포넌트
- [ ] 메시지 스트리밍 처리
- [ ] 출처 정보 표시
- [ ] 대화 히스토리 관리

### Day 5: 최적화 및 테스트

- [ ] 성능 최적화
- [ ] 반응형 디자인 완성
- [ ] 에러 핸들링
- [ ] 테스트 코드 작성

## 🛠 기술 스택

### 핵심 라이브러리

```json
{
  "devDependencies": {
    "@sveltejs/adapter-auto": "^2.0.0",
    "@sveltejs/kit": "^1.20.4",
    "@typescript-eslint/eslint-plugin": "^6.0.0",
    "@typescript-eslint/parser": "^6.0.0",
    "autoprefixer": "^10.4.14",
    "eslint": "^8.28.0",
    "eslint-config-prettier": "^8.5.0",
    "eslint-plugin-svelte": "^2.30.0",
    "postcss": "^8.4.24",
    "prettier": "^2.8.0",
    "prettier-plugin-svelte": "^2.10.1",
    "svelte": "^4.0.5",
    "svelte-check": "^3.4.3",
    "tailwindcss": "^3.3.0",
    "tslib": "^2.4.1",
    "typescript": "^5.0.0",
    "vite": "^4.4.2"
  },
  "dependencies": {
    "daisyui": "^4.4.0",
    "lucide-svelte": "^0.294.0",
    "socket.io-client": "^4.7.0",
    "marked": "^9.1.0",
    "prismjs": "^1.29.0",
    "js-cookie": "^3.0.5",
    "@types/js-cookie": "^3.0.6"
  }
}
```

## 📁 폴더 구조

```
src/
├── lib/
│   ├── components/           # 재사용 가능한 컴포넌트
│   │   ├── ui/              # 기본 UI 컴포넌트
│   │   ├── chat/            # 채팅 관련 컴포넌트
│   │   ├── documents/       # 문서 관리 컴포넌트
│   │   └── auth/            # 인증 관련 컴포넌트
│   ├── stores/              # Svelte 스토어
│   ├── utils/               # 유틸리티 함수
│   ├── api/                 # API 호출 함수
│   └── types/               # TypeScript 타입 정의
├── routes/                  # 페이지 라우트
│   ├── (auth)/             # 인증 관련 페이지
│   ├── (app)/              # 메인 앱 페이지
│   └── +layout.svelte      # 기본 레이아웃
├── app.html                # HTML 템플릿
└── app.postcss            # 전역 스타일
```

## 🎨 UI/UX 설계

### 디자인 시스템

```scss
// Tailwind 커스텀 설정
module.exports = {
  content: ['./src/**/*.{html,js,svelte,ts}'],
  theme: {
    extend: {
      colors: {
        primary: '#3B82F6',
        secondary: '#8B5CF6',
        accent: '#F59E0B',
        neutral: '#374151',
        'base-100': '#FFFFFF',
        'base-200': '#F9FAFB',
        'base-300': '#F3F4F6'
      }
    }
  },
  plugins: [require('daisyui')],
  daisyui: {
    themes: ['light', 'dark']
  }
}
```

### 주요 컴포넌트 설계

#### 1. 레이아웃 컴포넌트

```typescript
// src/lib/components/ui/Layout.svelte
interface LayoutProps {
  title?: string;
  showSidebar?: boolean;
  showHeader?: boolean;
}
```

#### 2. 채팅 컴포넌트

```typescript
// src/lib/components/chat/ChatInterface.svelte
interface Message {
  id: string;
  role: "user" | "assistant";
  content: string;
  sources?: Source[];
  timestamp: Date;
}

interface Source {
  document_id: string;
  title: string;
  page?: number;
  relevance_score: number;
}
```

#### 3. 파일 업로드 컴포넌트

```typescript
// src/lib/components/documents/FileUpload.svelte
interface UploadFile {
  file: File;
  status: "pending" | "uploading" | "processing" | "completed" | "error";
  progress: number;
  error?: string;
}
```

## 🔄 상태 관리

### Svelte Store 구조

```typescript
// src/lib/stores/auth.ts
interface AuthStore {
  user: User | null;
  token: string | null;
  isAuthenticated: boolean;
  isLoading: boolean;
}

// src/lib/stores/documents.ts
interface DocumentsStore {
  documents: Document[];
  isLoading: boolean;
  error: string | null;
}

// src/lib/stores/chat.ts
interface ChatStore {
  conversations: Conversation[];
  currentConversation: Conversation | null;
  messages: Message[];
  isTyping: boolean;
}
```

### 스토어 구현 예시

```typescript
// src/lib/stores/auth.ts
import { writable } from "svelte/store";
import { browser } from "$app/environment";
import Cookies from "js-cookie";

interface User {
  id: string;
  email: string;
  name: string;
  role: string;
}

interface AuthState {
  user: User | null;
  token: string | null;
  isAuthenticated: boolean;
  isLoading: boolean;
}

const createAuthStore = () => {
  const { subscribe, set, update } = writable<AuthState>({
    user: null,
    token: null,
    isAuthenticated: false,
    isLoading: false,
  });

  return {
    subscribe,
    login: async (email: string, password: string) => {
      update((state) => ({ ...state, isLoading: true }));
      try {
        const response = await fetch("/api/auth/login", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ email, password }),
        });

        if (response.ok) {
          const { user, token } = await response.json();
          Cookies.set("token", token, { expires: 7 });
          set({ user, token, isAuthenticated: true, isLoading: false });
          return { success: true };
        }
      } catch (error) {
        update((state) => ({ ...state, isLoading: false }));
        return { success: false, error: error.message };
      }
    },
    logout: () => {
      Cookies.remove("token");
      set({
        user: null,
        token: null,
        isAuthenticated: false,
        isLoading: false,
      });
    },
    initialize: () => {
      if (browser) {
        const token = Cookies.get("token");
        if (token) {
          // 토큰 검증 및 사용자 정보 가져오기
        }
      }
    },
  };
};

export const authStore = createAuthStore();
```

## 🔌 API 통신

### API 클라이언트 구현

```typescript
// src/lib/api/client.ts
import { authStore } from "$lib/stores/auth";
import { get } from "svelte/store";

class ApiClient {
  private baseUrl: string;

  constructor(baseUrl: string = "/api") {
    this.baseUrl = baseUrl;
  }

  private async request<T>(
    endpoint: string,
    options: RequestInit = {}
  ): Promise<T> {
    const auth = get(authStore);
    const url = `${this.baseUrl}${endpoint}`;

    const config: RequestInit = {
      headers: {
        "Content-Type": "application/json",
        ...(auth.token && { Authorization: `Bearer ${auth.token}` }),
        ...options.headers,
      },
      ...options,
    };

    const response = await fetch(url, config);

    if (!response.ok) {
      throw new Error(`API Error: ${response.statusText}`);
    }

    return response.json();
  }

  // 인증 API
  async login(email: string, password: string) {
    return this.request("/auth/login", {
      method: "POST",
      body: JSON.stringify({ email, password }),
    });
  }

  // 문서 API
  async uploadDocument(file: File) {
    const formData = new FormData();
    formData.append("file", file);

    return this.request("/documents/upload", {
      method: "POST",
      body: formData,
      headers: {}, // FormData는 Content-Type 자동 설정
    });
  }

  // 채팅 API
  async sendMessage(conversationId: string, content: string) {
    return this.request(`/conversations/${conversationId}/messages`, {
      method: "POST",
      body: JSON.stringify({ content }),
    });
  }
}

export const apiClient = new ApiClient();
```

## 🔄 실시간 통신

### WebSocket 연결 관리

```typescript
// src/lib/utils/websocket.ts
import { io, Socket } from "socket.io-client";
import { authStore } from "$lib/stores/auth";
import { get } from "svelte/store";

class WebSocketService {
  private socket: Socket | null = null;

  connect() {
    const auth = get(authStore);
    if (!auth.token) return;

    this.socket = io(import.meta.env.VITE_WS_URL || "http://localhost:3000", {
      auth: { token: auth.token },
    });

    this.socket.on("connect", () => {
      console.log("WebSocket connected");
    });

    this.socket.on("document_processed", (data) => {
      // 문서 처리 완료 이벤트 처리
    });

    this.socket.on("message_response", (data) => {
      // 채팅 응답 이벤트 처리
    });
  }

  disconnect() {
    if (this.socket) {
      this.socket.disconnect();
      this.socket = null;
    }
  }

  emit(event: string, data: any) {
    if (this.socket) {
      this.socket.emit(event, data);
    }
  }
}

export const wsService = new WebSocketService();
```

## 🎯 주요 페이지 구현

### 1. 대시보드 페이지

```svelte
<!-- src/routes/(app)/dashboard/+page.svelte -->
<script lang="ts">
  import { onMount } from 'svelte';
  import { documentsStore } from '$lib/stores/documents';
  import DocumentList from '$lib/components/documents/DocumentList.svelte';
  import UploadButton from '$lib/components/documents/UploadButton.svelte';
  import StatsCards from '$lib/components/dashboard/StatsCards.svelte';

  onMount(() => {
    documentsStore.loadDocuments();
  });
</script>

<div class="container mx-auto p-6">
  <div class="mb-8">
    <h1 class="text-3xl font-bold">Dashboard</h1>
    <p class="text-gray-600">RAG 시스템 관리 대시보드</p>
  </div>

  <StatsCards />

  <div class="grid grid-cols-1 lg:grid-cols-3 gap-6 mt-8">
    <div class="lg:col-span-2">
      <DocumentList />
    </div>
    <div>
      <UploadButton />
    </div>
  </div>
</div>
```

### 2. 채팅 페이지

```svelte
<!-- src/routes/(app)/chat/+page.svelte -->
<script lang="ts">
  import { onMount, onDestroy } from 'svelte';
  import { chatStore } from '$lib/stores/chat';
  import ChatInterface from '$lib/components/chat/ChatInterface.svelte';
  import ConversationList from '$lib/components/chat/ConversationList.svelte';
  import { wsService } from '$lib/utils/websocket';

  onMount(() => {
    wsService.connect();
    chatStore.loadConversations();
  });

  onDestroy(() => {
    wsService.disconnect();
  });
</script>

<div class="flex h-screen">
  <div class="w-1/4 border-r">
    <ConversationList />
  </div>
  <div class="flex-1">
    <ChatInterface />
  </div>
</div>
```

## 🧪 테스트 전략

### 단위 테스트

```typescript
// src/lib/components/__tests__/FileUpload.test.ts
import { render, fireEvent } from "@testing-library/svelte";
import FileUpload from "../documents/FileUpload.svelte";

describe("FileUpload Component", () => {
  test("renders upload button", () => {
    const { getByText } = render(FileUpload);
    expect(getByText("파일 업로드")).toBeInTheDocument();
  });

  test("handles file selection", async () => {
    const { getByLabelText } = render(FileUpload);
    const input = getByLabelText("파일 선택");

    const file = new File(["test"], "test.pdf", { type: "application/pdf" });
    await fireEvent.change(input, { target: { files: [file] } });

    // 파일 선택 후 상태 확인
  });
});
```

## 🚀 배포 및 최적화

### 빌드 최적화

```typescript
// vite.config.ts
import { sveltekit } from "@sveltejs/kit/vite";
import { defineConfig } from "vite";

export default defineConfig({
  plugins: [sveltekit()],
  build: {
    rollupOptions: {
      output: {
        manualChunks: {
          vendor: ["socket.io-client", "marked"],
        },
      },
    },
  },
  optimizeDeps: {
    include: ["socket.io-client"],
  },
});
```

### 성능 모니터링

```typescript
// src/lib/utils/performance.ts
export const trackPageView = (page: string) => {
  if (typeof window !== "undefined" && window.gtag) {
    window.gtag("config", "GA_MEASUREMENT_ID", {
      page_title: page,
      page_location: window.location.href,
    });
  }
};

export const trackEvent = (
  action: string,
  category: string,
  label?: string
) => {
  if (typeof window !== "undefined" && window.gtag) {
    window.gtag("event", action, {
      event_category: category,
      event_label: label,
    });
  }
};
```

## ✅ 체크리스트

### 개발 완료 기준

- [ ] 모든 페이지가 반응형으로 작동
- [ ] 인증 시스템 완전 구현
- [ ] 파일 업로드 및 실시간 상태 업데이트
- [ ] 채팅 인터페이스 완전 구현
- [ ] 에러 핸들링 및 로딩 상태 처리
- [ ] 접근성(a11y) 기본 요구사항 충족
- [ ] 성능 최적화 적용
- [ ] 기본 테스트 코드 작성
- [ ] 문서화 완료
- [ ] 배포 준비 완료
