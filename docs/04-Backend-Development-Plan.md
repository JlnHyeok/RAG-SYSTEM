# Backend (NestJS) 개발 프로세스 및 계획

## 📋 개발 일정 (5일)

### Day 1: 프로젝트 초기 설정 및 기본 구조

- [ ] NestJS 프로젝트 초기화
- [ ] MongoDB 데이터베이스 설정
- [ ] Mongoose 스키마 정의
- [ ] 기본 모듈 구조 생성

### Day 2: 인증 시스템 구현

- [ ] JWT 인증 모듈 구현
- [ ] 사용자 등록/로그인 API
- [ ] 가드(Guard) 및 미들웨어 설정
- [ ] 패스워드 암호화 및 검증

### Day 3: 문서 관리 API

- [ ] 파일 업로드 API 구현
- [ ] 문서 메타데이터 관리
- [ ] 파일 검증 및 보안
- [ ] 문서 상태 관리

### Day 4: 채팅 및 WebSocket

- [ ] 대화 관리 API
- [ ] 메시지 CRUD API
- [ ] WebSocket 실시간 통신
- [ ] Agent 연동 API

### Day 5: 최적화 및 테스트

- [ ] API 문서화 (Swagger)
- [ ] 에러 핸들링 및 로깅
- [ ] 성능 최적화
- [ ] 단위 테스트 작성

## 🛠 기술 스택

### 핵심 라이브러리

```json
{
  "dependencies": {
    "@nestjs/common": "^10.0.0",
    "@nestjs/core": "^10.0.0",
    "@nestjs/platform-express": "^10.0.0",
    "@nestjs/mongoose": "^10.0.0",
    "@nestjs/config": "^3.0.0",
    "@nestjs/jwt": "^10.1.0",
    "@nestjs/passport": "^10.0.0",
    "@nestjs/websockets": "^10.0.0",
    "@nestjs/platform-socket.io": "^10.0.0",
    "@nestjs/swagger": "^7.1.0",
    "mongoose": "^8.0.0",
    "bcryptjs": "^2.4.3",
    "passport": "^0.6.0",
    "passport-jwt": "^4.0.1",
    "passport-local": "^1.0.0",
    "class-validator": "^0.14.0",
    "class-transformer": "^0.5.1",
    "multer": "^1.4.5",
    "socket.io": "^4.7.0",
    "winston": "^3.10.0",
    "helmet": "^7.0.0",
    "compression": "^1.7.4"
  },
  "devDependencies": {
    "@nestjs/cli": "^10.0.0",
    "@nestjs/schematics": "^10.0.0",
    "@nestjs/testing": "^10.0.0",
    "@types/express": "^4.17.17",
    "@types/jest": "^29.5.2",
    "@types/node": "^20.3.1",
    "@types/passport-jwt": "^3.0.9",
    "@types/passport-local": "^1.0.35",
    "@types/bcryptjs": "^2.4.2",
    "@types/multer": "^1.4.7",
    "@types/mongoose": "^5.11.97",
    "jest": "^29.5.0",
    "source-map-support": "^0.5.21",
    "supertest": "^6.3.3",
    "ts-jest": "^29.1.0",
    "ts-loader": "^9.4.3",
    "ts-node": "^10.9.1",
    "tsconfig-paths": "^4.2.1",
    "typescript": "^5.1.3"
  }
}
```

## 📁 폴더 구조

```
src/
├── auth/                    # 인증 관련 모듈
│   ├── dto/                 # Data Transfer Objects
│   ├── entities/            # 사용자 엔티티
│   ├── guards/              # 인증 가드
│   ├── strategies/          # Passport 전략
│   ├── auth.controller.ts
│   ├── auth.service.ts
│   └── auth.module.ts
├── documents/               # 문서 관리 모듈
│   ├── dto/
│   ├── entities/
│   ├── documents.controller.ts
│   ├── documents.service.ts
│   └── documents.module.ts
├── conversations/           # 대화 관리 모듈
│   ├── dto/
│   ├── entities/
│   ├── conversations.controller.ts
│   ├── conversations.service.ts
│   └── conversations.module.ts
├── messages/                # 메시지 관리 모듈
│   ├── dto/
│   ├── entities/
│   ├── messages.controller.ts
│   ├── messages.service.ts
│   └── messages.module.ts
├── agent/                   # RAG Agent 연동 모듈
│   ├── dto/
│   ├── agent.controller.ts
│   ├── agent.service.ts
│   └── agent.module.ts
├── websocket/              # WebSocket 모듈
│   ├── websocket.gateway.ts
│   └── websocket.module.ts
├── common/                 # 공통 모듈
│   ├── decorators/         # 커스텀 데코레이터
│   ├── filters/            # 예외 필터
│   ├── guards/             # 공통 가드
│   ├── interceptors/       # 인터셉터
│   ├── pipes/              # 파이프
│   └── utils/              # 유틸리티 함수
├── config/                 # 설정 파일
│   ├── database.config.ts
│   ├── jwt.config.ts
│   └── app.config.ts
├── app.controller.ts
├── app.service.ts
├── app.module.ts
└── main.ts
```

## 🗄 데이터베이스 설계

### 스키마 정의

#### User Schema

```typescript
// src/auth/schemas/user.schema.ts
import { Prop, Schema, SchemaFactory } from '@nestjs/mongoose';
import { Document, Types } from 'mongoose';
import { Exclude } from 'class-transformer';

@Schema({ timestamps: true, collection: 'users' })
export class User extends Document {
  @Prop({ required: true, unique: true })
  email: string;

  @Prop({ required: true })
  @Exclude()
  passwordHash: string;

  @Prop({ required: true })
  name: string;

  @Prop({ default: 'user' })
  role: string;

  @Prop({ type: [String], default: [] })
  devices: string[];  // 소유 기기 목록

  // Mongoose는 자동으로 createdAt, updatedAt 추가 (timestamps: true)
}

export const UserSchema = SchemaFactory.createForClass(User);

// 인덱스 추가
UserSchema.index({ email: 1 });
```

#### Document Schema

```typescript
// src/documents/schemas/document.schema.ts
import { Prop, Schema, SchemaFactory } from '@nestjs/mongoose';
import { Document as MongooseDocument, Types } from 'mongoose';

@Schema({ timestamps: true, collection: 'documents' })
export class Document extends MongooseDocument {
  @Prop({ type: Types.ObjectId, ref: 'User', required: true })
  userId: Types.ObjectId;

  @Prop({ required: true, maxlength: 500 })
  title: string;

  @Prop({ required: true, maxlength: 1000 })
  filePath: string;

  @Prop({ required: true, maxlength: 50 })
  fileType: string;

  @Prop({ required: true, type: Number })
  fileSize: number;

  @Prop({ default: 0 })
  chunkCount: number;

  @Prop({ default: 'processing' })
  status: string;

  @Prop({ type: Object })
  metadata?: Record<string, any>;
}

export const DocumentSchema = SchemaFactory.createForClass(Document);

// 인덱스
DocumentSchema.index({ userId: 1, createdAt: -1 });
DocumentSchema.index({ status: 1 });
```

#### Conversation Schema

```typescript
// src/conversations/schemas/conversation.schema.ts
import { Prop, Schema, SchemaFactory } from '@nestjs/mongoose';
import { Document, Types } from 'mongoose';
import { Message } from '../../messages/schemas/message.schema';

@Schema({ timestamps: true, collection: 'conversations' })
export class Conversation extends Document {
  @Prop({ type: Types.ObjectId, ref: 'User', required: true })
  userId: Types.ObjectId;

  @Prop({ maxlength: 500 })
  title?: string;

  // Embedded messages (MongoDB 방식)
  @Prop({
    type: [{
      role: { type: String, required: true },
      content: { type: String, required: true },
      sources: { type: Object },
      timestamp: { type: Date, default: Date.now }
    }],
    default: []
  })
  messages: Array<{
    role: string;
    content: string;
    sources?: any;
    timestamp: Date;
  }>;
}

export const ConversationSchema = SchemaFactory.createForClass(Conversation);

// 인덱스
ConversationSchema.index({ userId: 1, updatedAt: -1 });
```

#### Message Schema (참고용 - Conversation에 임베디드)

> **Note**: MongoDB 특성상 메시지는 Conversation 스키마에 임베디드 문서로 저장됩니다.
> 별도 컬렉션이 필요한 경우에만 아래 스키마를 사용하세요.

```typescript
// src/messages/schemas/message.schema.ts (선택적)
import { Prop, Schema, SchemaFactory } from '@nestjs/mongoose';
import { Document, Types } from 'mongoose';

@Schema({ timestamps: true, collection: 'messages' })
export class Message extends Document {
  @Prop({ type: Types.ObjectId, ref: 'Conversation', required: true })
  conversationId: Types.ObjectId;

  @Prop({ required: true, maxlength: 20 })
  role: string; // 'user' or 'assistant'

  @Prop({ required: true })
  content: string;

  @Prop({ type: Object })
  sources?: any;
}

export const MessageSchema = SchemaFactory.createForClass(Message);

// 인덱스
MessageSchema.index({ conversationId: 1, createdAt: -1 });
```

## 🔐 인증 시스템

> **Note**: 아래 코드 예시는 TypeORM 패턴입니다. MongoDB/Mongoose로 변환 시:
> - `@InjectRepository(User)` → `@InjectModel(User.name)`
> - `Repository<User>` → `Model<User>`
> - `userRepository.findOne({ where: { email } })` → `userModel.findOne({ email })`
> - `userRepository.create()` → `new userModel()`
> - `user.id` → `user._id`

### JWT 전략 구현

```typescript
// src/auth/strategies/jwt.strategy.ts
import { Injectable, UnauthorizedException } from "@nestjs/common";
import { PassportStrategy } from "@nestjs/passport";
import { ExtractJwt, Strategy } from "passport-jwt";
import { ConfigService } from "@nestjs/config";
import { AuthService } from "../auth.service";

@Injectable()
export class JwtStrategy extends PassportStrategy(Strategy) {
  constructor(
    private configService: ConfigService,
    private authService: AuthService
  ) {
    super({
      jwtFromRequest: ExtractJwt.fromAuthHeaderAsBearerToken(),
      ignoreExpiration: false,
      secretOrKey: configService.get<string>("JWT_SECRET"),
    });
  }

  async validate(payload: any) {
    const user = await this.authService.validateUserById(payload.sub);
    if (!user) {
      throw new UnauthorizedException();
    }
    return user;
  }
}
```

### 인증 서비스

```typescript
// src/auth/auth.service.ts
import { Injectable, UnauthorizedException } from "@nestjs/common";
import { JwtService } from "@nestjs/jwt";
import { InjectRepository } from "@nestjs/typeorm";
import { Repository } from "typeorm";
import * as bcrypt from "bcryptjs";
import { User } from "./entities/user.entity";
import { LoginDto } from "./dto/login.dto";
import { RegisterDto } from "./dto/register.dto";

@Injectable()
export class AuthService {
  constructor(
    @InjectRepository(User)
    private userRepository: Repository<User>,
    private jwtService: JwtService
  ) {}

  async register(registerDto: RegisterDto) {
    const { email, password, name } = registerDto;

    // 이메일 중복 검사
    const existingUser = await this.userRepository.findOne({
      where: { email },
    });
    if (existingUser) {
      throw new UnauthorizedException("이메일이 이미 사용 중입니다.");
    }

    // 비밀번호 해싱
    const saltRounds = 10;
    const passwordHash = await bcrypt.hash(password, saltRounds);

    // 사용자 생성
    const user = this.userRepository.create({
      email,
      passwordHash,
      name,
    });

    const savedUser = await this.userRepository.save(user);

    // JWT 토큰 생성
    const token = this.generateToken(savedUser);

    return {
      user: this.excludePassword(savedUser),
      token,
    };
  }

  async login(loginDto: LoginDto) {
    const { email, password } = loginDto;

    const user = await this.userRepository.findOne({ where: { email } });
    if (!user) {
      throw new UnauthorizedException("이메일 또는 비밀번호가 잘못되었습니다.");
    }

    const isPasswordValid = await bcrypt.compare(password, user.passwordHash);
    if (!isPasswordValid) {
      throw new UnauthorizedException("이메일 또는 비밀번호가 잘못되었습니다.");
    }

    const token = this.generateToken(user);

    return {
      user: this.excludePassword(user),
      token,
    };
  }

  async validateUserById(id: string): Promise<User> {
    return await this.userRepository.findOne({ where: { id } });
  }

  private generateToken(user: User): string {
    const payload = { email: user.email, sub: user.id };
    return this.jwtService.sign(payload);
  }

  private excludePassword(user: User): Partial<User> {
    const { passwordHash, ...userWithoutPassword } = user;
    return userWithoutPassword;
  }
}
```

## 📁 파일 업로드 시스템

### 파일 업로드 컨트롤러

```typescript
// src/documents/documents.controller.ts
import {
  Controller,
  Post,
  Get,
  Delete,
  Param,
  UseInterceptors,
  UploadedFile,
  UseGuards,
  Request,
  Body,
  ParseUUIDPipe,
} from "@nestjs/common";
import { FileInterceptor } from "@nestjs/platform-express";
import { JwtAuthGuard } from "../auth/guards/jwt-auth.guard";
import { DocumentsService } from "./documents.service";
import {
  ApiTags,
  ApiOperation,
  ApiConsumes,
  ApiBearerAuth,
} from "@nestjs/swagger";
import { diskStorage } from "multer";
import { extname } from "path";

@ApiTags("Documents")
@ApiBearerAuth()
@Controller("documents")
@UseGuards(JwtAuthGuard)
export class DocumentsController {
  constructor(private readonly documentsService: DocumentsService) {}

  @Post("upload")
  @ApiOperation({ summary: "문서 업로드" })
  @ApiConsumes("multipart/form-data")
  @UseInterceptors(
    FileInterceptor("file", {
      storage: diskStorage({
        destination: "./uploads",
        filename: (req, file, callback) => {
          const uniqueSuffix =
            Date.now() + "-" + Math.round(Math.random() * 1e9);
          callback(null, `${uniqueSuffix}${extname(file.originalname)}`);
        },
      }),
      fileFilter: (req, file, callback) => {
        const allowedTypes = /pdf|docx?|txt/;
        const extName = allowedTypes.test(
          extname(file.originalname).toLowerCase()
        );
        const mimeType = allowedTypes.test(file.mimetype);

        if (extName && mimeType) {
          callback(null, true);
        } else {
          callback(new Error("지원하지 않는 파일 형식입니다."), false);
        }
      },
      limits: {
        fileSize: 50 * 1024 * 1024, // 50MB
      },
    })
  )
  async uploadDocument(
    @UploadedFile() file: Express.Multer.File,
    @Request() req
  ) {
    return this.documentsService.uploadDocument(file, req.user.id);
  }

  @Get()
  @ApiOperation({ summary: "사용자 문서 목록 조회" })
  async getDocuments(@Request() req) {
    return this.documentsService.getDocumentsByUserId(req.user.id);
  }

  @Get(":id")
  @ApiOperation({ summary: "문서 상세 조회" })
  async getDocument(@Param("id", ParseUUIDPipe) id: string, @Request() req) {
    return this.documentsService.getDocument(id, req.user.id);
  }

  @Delete(":id")
  @ApiOperation({ summary: "문서 삭제" })
  async deleteDocument(@Param("id", ParseUUIDPipe) id: string, @Request() req) {
    return this.documentsService.deleteDocument(id, req.user.id);
  }

  @Get(":id/status")
  @ApiOperation({ summary: "문서 처리 상태 조회" })
  async getDocumentStatus(
    @Param("id", ParseUUIDPipe) id: string,
    @Request() req
  ) {
    return this.documentsService.getDocumentStatus(id, req.user.id);
  }
}
```

### 문서 서비스

```typescript
// src/documents/documents.service.ts
import {
  Injectable,
  NotFoundException,
  ForbiddenException,
} from "@nestjs/common";
import { InjectRepository } from "@nestjs/typeorm";
import { Repository } from "typeorm";
import { Document } from "./entities/document.entity";
import { AgentService } from "../agent/agent.service";
import { WebSocketGateway } from "../websocket/websocket.gateway";

@Injectable()
export class DocumentsService {
  constructor(
    @InjectRepository(Document)
    private documentRepository: Repository<Document>,
    private agentService: AgentService,
    private websocketGateway: WebSocketGateway
  ) {}

  async uploadDocument(file: Express.Multer.File, userId: string) {
    // 문서 메타데이터 저장
    const document = this.documentRepository.create({
      userId,
      title: file.originalname,
      filePath: file.path,
      fileType: file.mimetype,
      fileSize: file.size,
      status: "processing",
    });

    const savedDocument = await this.documentRepository.save(document);

    // 백그라운드에서 문서 처리
    this.processDocumentAsync(savedDocument);

    return savedDocument;
  }

  async getDocumentsByUserId(userId: string) {
    return await this.documentRepository.find({
      where: { userId },
      order: { createdAt: "DESC" },
    });
  }

  async getDocument(id: string, userId: string) {
    const document = await this.documentRepository.findOne({
      where: { id, userId },
    });

    if (!document) {
      throw new NotFoundException("문서를 찾을 수 없습니다.");
    }

    return document;
  }

  async deleteDocument(id: string, userId: string) {
    const document = await this.getDocument(id, userId);

    // 파일 시스템에서 파일 삭제
    // Vector DB에서 벡터 삭제
    await this.agentService.deleteDocumentVectors(id);

    await this.documentRepository.remove(document);

    return { message: "문서가 삭제되었습니다." };
  }

  async getDocumentStatus(id: string, userId: string) {
    const document = await this.getDocument(id, userId);
    return {
      id: document.id,
      status: document.status,
      chunkCount: document.chunkCount,
    };
  }

  private async processDocumentAsync(document: Document) {
    try {
      // Agent 서비스를 통해 문서 처리
      const result = await this.agentService.processDocument(document);

      // 문서 상태 업데이트
      await this.documentRepository.update(document.id, {
        status: "completed",
        chunkCount: result.chunkCount,
        metadata: result.metadata,
      });

      // WebSocket을 통해 클라이언트에 알림
      this.websocketGateway.notifyDocumentProcessed(document.userId, {
        documentId: document.id,
        status: "completed",
        chunkCount: result.chunkCount,
      });
    } catch (error) {
      // 에러 처리
      await this.documentRepository.update(document.id, {
        status: "error",
        metadata: { error: error.message },
      });

      this.websocketGateway.notifyDocumentProcessed(document.userId, {
        documentId: document.id,
        status: "error",
        error: error.message,
      });
    }
  }
}
```

## 🔌 WebSocket 구현

### WebSocket Gateway

```typescript
// src/websocket/websocket.gateway.ts
import {
  WebSocketGateway,
  WebSocketServer,
  SubscribeMessage,
  MessageBody,
  ConnectedSocket,
  OnGatewayConnection,
  OnGatewayDisconnect,
} from "@nestjs/websockets";
import { Server, Socket } from "socket.io";
import { UseGuards } from "@nestjs/common";
import { WsJwtGuard } from "../auth/guards/ws-jwt.guard";

@WebSocketGateway({
  cors: {
    origin: process.env.FRONTEND_URL || "http://localhost:5173",
    credentials: true,
  },
})
export class WebSocketGateway
  implements OnGatewayConnection, OnGatewayDisconnect
{
  @WebSocketServer()
  server: Server;

  private connectedClients = new Map<string, string>(); // socketId -> userId

  async handleConnection(client: Socket) {
    try {
      // JWT 토큰 검증
      const token = client.handshake.auth.token;
      const user = await this.validateToken(token);

      if (user) {
        this.connectedClients.set(client.id, user.id);
        client.join(`user-${user.id}`);
        console.log(`User ${user.id} connected`);
      } else {
        client.disconnect();
      }
    } catch (error) {
      client.disconnect();
    }
  }

  handleDisconnect(client: Socket) {
    const userId = this.connectedClients.get(client.id);
    if (userId) {
      this.connectedClients.delete(client.id);
      console.log(`User ${userId} disconnected`);
    }
  }

  @SubscribeMessage("join-conversation")
  @UseGuards(WsJwtGuard)
  async joinConversation(
    @MessageBody() data: { conversationId: string },
    @ConnectedSocket() client: Socket
  ) {
    client.join(`conversation-${data.conversationId}`);
  }

  // 문서 처리 완료 알림
  notifyDocumentProcessed(userId: string, data: any) {
    this.server.to(`user-${userId}`).emit("document_processed", data);
  }

  // 채팅 응답 전송
  sendMessageResponse(conversationId: string, data: any) {
    this.server
      .to(`conversation-${conversationId}`)
      .emit("message_response", data);
  }

  // 실시간 타이핑 상태
  notifyTyping(conversationId: string, isTyping: boolean) {
    this.server
      .to(`conversation-${conversationId}`)
      .emit("typing", { isTyping });
  }

  private async validateToken(token: string) {
    // JWT 토큰 검증 로직
    // 실제로는 JwtService를 사용해야 함
    return null;
  }
}
```

## 🤖 Agent 연동 서비스

### Agent 서비스

```typescript
// src/agent/agent.service.ts
import { Injectable } from "@nestjs/common";
import { HttpService } from "@nestjs/axios";
import { ConfigService } from "@nestjs/config";
import { firstValueFrom } from "rxjs";
import { Document } from "../documents/entities/document.entity";

@Injectable()
export class AgentService {
  private readonly agentUrl: string;

  constructor(
    private readonly httpService: HttpService,
    private readonly configService: ConfigService
  ) {
    this.agentUrl =
      this.configService.get<string>("AGENT_URL") || "http://localhost:8000";
  }

  async processDocument(document: Document) {
    try {
      const response = await firstValueFrom(
        this.httpService.post(`${this.agentUrl}/process-document`, {
          document_id: document.id,
          file_path: document.filePath,
          file_type: document.fileType,
          user_id: document.userId,
        })
      );

      return response.data;
    } catch (error) {
      throw new Error(`문서 처리 실패: ${error.message}`);
    }
  }

  async queryAgent(query: string, userId: string, conversationId?: string) {
    try {
      const response = await firstValueFrom(
        this.httpService.post(`${this.agentUrl}/query`, {
          query,
          user_id: userId,
          conversation_id: conversationId,
        })
      );

      return response.data;
    } catch (error) {
      throw new Error(`Agent 쿼리 실패: ${error.message}`);
    }
  }

  async deleteDocumentVectors(documentId: string) {
    try {
      await firstValueFrom(
        this.httpService.delete(
          `${this.agentUrl}/documents/${documentId}/vectors`
        )
      );
    } catch (error) {
      console.error(`벡터 삭제 실패: ${error.message}`);
    }
  }
}
```

## 📝 API 문서화

### Swagger 설정

```typescript
// src/main.ts
import { NestFactory } from "@nestjs/core";
import { SwaggerModule, DocumentBuilder } from "@nestjs/swagger";
import { ValidationPipe } from "@nestjs/common";
import { AppModule } from "./app.module";
import * as helmet from "helmet";
import * as compression from "compression";

async function bootstrap() {
  const app = await NestFactory.create(AppModule);

  // 보안 미들웨어
  app.use(helmet());
  app.use(compression());

  // CORS 설정
  app.enableCors({
    origin: process.env.FRONTEND_URL || "http://localhost:5173",
    credentials: true,
  });

  // 전역 유효성 검사 파이프
  app.useGlobalPipes(
    new ValidationPipe({
      whitelist: true,
      forbidNonWhitelisted: true,
      transform: true,
    })
  );

  // Swagger 설정
  const config = new DocumentBuilder()
    .setTitle("RAG System API")
    .setDescription("RAG 시스템 백엔드 API 문서")
    .setVersion("1.0")
    .addBearerAuth()
    .build();

  const document = SwaggerModule.createDocument(app, config);
  SwaggerModule.setup("api", app, document);

  await app.listen(3000);
}
bootstrap();
```

## 🧪 테스트 전략

### 단위 테스트 예시

```typescript
// src/auth/auth.service.spec.ts
import { Test, TestingModule } from "@nestjs/testing";
import { getRepositoryToken } from "@nestjs/typeorm";
import { JwtService } from "@nestjs/jwt";
import { AuthService } from "./auth.service";
import { User } from "./entities/user.entity";
import * as bcrypt from "bcryptjs";

describe("AuthService", () => {
  let service: AuthService;
  let userRepository: any;
  let jwtService: JwtService;

  beforeEach(async () => {
    const module: TestingModule = await Test.createTestingModule({
      providers: [
        AuthService,
        {
          provide: getRepositoryToken(User),
          useValue: {
            findOne: jest.fn(),
            create: jest.fn(),
            save: jest.fn(),
          },
        },
        {
          provide: JwtService,
          useValue: {
            sign: jest.fn(),
          },
        },
      ],
    }).compile();

    service = module.get<AuthService>(AuthService);
    userRepository = module.get(getRepositoryToken(User));
    jwtService = module.get<JwtService>(JwtService);
  });

  describe("register", () => {
    it("should register a new user successfully", async () => {
      const registerDto = {
        email: "test@example.com",
        password: "password123",
        name: "테스트 사용자",
      };

      userRepository.findOne.mockResolvedValue(null);
      userRepository.create.mockReturnValue({ id: "1", ...registerDto });
      userRepository.save.mockResolvedValue({ id: "1", ...registerDto });
      jwtService.sign.mockReturnValue("jwt-token");

      const result = await service.register(registerDto);

      expect(result.user.email).toBe(registerDto.email);
      expect(result.token).toBe("jwt-token");
    });
  });
});
```

### E2E 테스트 예시

```typescript
// test/auth.e2e-spec.ts
import { Test, TestingModule } from "@nestjs/testing";
import { INestApplication } from "@nestjs/common";
import * as request from "supertest";
import { AppModule } from "../src/app.module";

describe("AuthController (e2e)", () => {
  let app: INestApplication;

  beforeEach(async () => {
    const moduleFixture: TestingModule = await Test.createTestingModule({
      imports: [AppModule],
    }).compile();

    app = moduleFixture.createNestApplication();
    await app.init();
  });

  it("/auth/register (POST)", () => {
    return request(app.getHttpServer())
      .post("/auth/register")
      .send({
        email: "test@example.com",
        password: "password123",
        name: "테스트 사용자",
      })
      .expect(201)
      .expect((res) => {
        expect(res.body.user.email).toBe("test@example.com");
        expect(res.body.token).toBeDefined();
      });
  });
});
```

## 🔧 성능 최적화

### 캐싱 구현

```typescript
// src/common/interceptors/cache.interceptor.ts
import {
  Injectable,
  NestInterceptor,
  ExecutionContext,
  CallHandler,
} from "@nestjs/common";
import { Observable, of } from "rxjs";
import { tap } from "rxjs/operators";
import * as Redis from "ioredis";

@Injectable()
export class CacheInterceptor implements NestInterceptor {
  private redis = new Redis(process.env.REDIS_URL);

  intercept(context: ExecutionContext, next: CallHandler): Observable<any> {
    const request = context.switchToHttp().getRequest();
    const cacheKey = this.generateCacheKey(request);

    return this.getFromCache(cacheKey).pipe(
      switchMap((cachedData) => {
        if (cachedData) {
          return of(cachedData);
        }

        return next.handle().pipe(
          tap((data) => this.setCache(cacheKey, data, 300)) // 5분 캐시
        );
      })
    );
  }

  private generateCacheKey(request: any): string {
    return `${request.method}:${request.url}:${JSON.stringify(request.query)}`;
  }

  private async getFromCache(key: string): Promise<any> {
    const cached = await this.redis.get(key);
    return cached ? JSON.parse(cached) : null;
  }

  private async setCache(key: string, data: any, ttl: number): Promise<void> {
    await this.redis.setex(key, ttl, JSON.stringify(data));
  }
}
```

## ✅ 체크리스트

### 개발 완료 기준

- [ ] 모든 API 엔드포인트 구현 완료
- [ ] JWT 인증 시스템 완전 구현
- [ ] 파일 업로드 및 검증 시스템 구현
- [ ] WebSocket 실시간 통신 구현
- [ ] Agent 연동 API 구현
- [ ] 데이터베이스 마이그레이션 완료
- [ ] API 문서화 (Swagger) 완료
- [ ] 에러 핸들링 및 로깅 시스템 구현
- [ ] 단위 테스트 및 E2E 테스트 작성
- [ ] 성능 최적화 적용
- [ ] 보안 설정 적용
- [ ] 배포 환경 설정 완료
