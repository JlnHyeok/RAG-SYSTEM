import { NestFactory } from '@nestjs/core';
import { AppModule } from './app.module';
import { WinstonModule, utilities } from 'nest-winston';
import * as winston from 'winston';
import * as winstonDaily from 'winston-daily-rotate-file';
import {
  HttpException,
  HttpStatus,
  LoggerService,
  ValidationPipe,
} from '@nestjs/common';
import { json, urlencoded } from 'body-parser';
import { LoggerInterceptor } from './logger/logger.interceptor';
import * as fs from 'fs';

async function bootstrap() {
  const defaultLogger = createWinstonLogger();

  const app = await NestFactory.create(AppModule, {
    //* Winston Logger 적용
    logger: defaultLogger,
    cors: true,
    httpsOptions:
      process.env.HTTPS_ENABLED == 'Y'
        ? {
            key: fs.readFileSync('./cert/key.pem'),
            cert: fs.readFileSync('./cert/cert.pem'),
          }
        : undefined,
  });

  //* Nest URL/포트 초기화
  const nestHost = process.env.HOST ? process.env.HOST : 'localhost';
  const nestPort = process.env.PORT ? process.env.PORT : '3000';

  //* REST Global Route 설정
  app.setGlobalPrefix('ai');

  //* 전역 로깅을  위해 Interceptor 등록
  app.useGlobalInterceptors(new LoggerInterceptor(defaultLogger));
  // app.useGlobalInterceptors(new TimeoutInterceptor());

  //* Validation Pipe 설정 (REST API 요청 유효성 검사)
  app.useGlobalPipes(createValidationPipe(defaultLogger));

  //* HTTP Request 사이즈 설정 (이미지 업로드)
  app.use(json({ limit: '50mb' }));
  app.use(urlencoded({ limit: '50mb', extended: true }));

  // app.use((req: IncomingMessage, res, next) => {
  //   console.log(req.url, req.method);
  //   if (req.url == '/ai/product-history' && req.method == 'POST') {
  //     setTimeout(() => next(), 3000);
  //   } else {
  //     next();
  //   }
  // });

  //* Nest Listen
  await app.listen(nestPort, nestHost);

  defaultLogger.log(
    `공구 생애주기 판단 솔루션 v1.0 successfully started\n+-------------------------------------------------+
|    공구 생애주기 판단 솔루션 v1.0               |
+-------------------------------------------------+`,
    ['공구 생애주기 판단 솔루션 v1.0'],
  );
}

function createWinstonLogger() {
  return WinstonModule.createLogger({
    transports: [
      new winston.transports.Console({
        level: process.env.LOGGER_NAME == 'dev' ? 'silly' : 'info',
        format: winston.format.combine(
          winston.format.timestamp({
            format: 'YYYY-MM-DD HH:mm:ss',
          }),
          utilities.format.nestLike(
            process.env.npm_package_name.toUpperCase(),
            {
              prettyPrint: true,
              colors: true,
            },
          ),
        ),
      }),
      new winstonDaily({
        level: process.env.LOGGER_NAME == 'dev' ? 'silly' : 'info',
        format: winston.format.combine(
          winston.format.timestamp({
            format: 'YYYY-MM-DD HH:mm:ss',
          }),
          utilities.format.nestLike(
            process.env.npm_package_name.toUpperCase(),
            {
              prettyPrint: true,
              colors: false,
            },
          ),
        ),
        datePattern: 'YYYY-MM-DD',
        dirname: 'logs',
        filename: '%DATE%.log',
        maxSize: 256 * 1024 * 1024,
        maxFiles: 128,
        zippedArchive: false,
      }),
    ],
  });
}
function createValidationPipe(logger: LoggerService) {
  return new ValidationPipe({
    exceptionFactory: (errors) => {
      // 유효성 에러에 대한 처리
      logger.error(
        `Request Error (${JSON.stringify(errors[0].constraints)})`,
        'HTTP',
      );

      throw new HttpException(
        {
          isSuccess: false,
          message: errors[0].constraints,
        },
        HttpStatus.BAD_REQUEST,
      );
    },
  });
}

bootstrap();
