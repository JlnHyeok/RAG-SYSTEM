import {
  CallHandler,
  ExecutionContext,
  Injectable,
  LoggerService,
  NestInterceptor,
} from '@nestjs/common';
import { IncomingMessage, ServerResponse } from 'http';
import { Observable, tap } from 'rxjs';

@Injectable()
export class LoggerInterceptor implements NestInterceptor {
  // 전역 Interceptor 등록 시 직접 전달하기 때문에 Decorator 사용 X
  constructor(private readonly logger: LoggerService) {}

  intercept(context: ExecutionContext, next: CallHandler): Observable<any> {
    const currentRequestType = context.getType();

    // HTTP Logging
    if (currentRequestType == 'http') {
      const currentRequest: IncomingMessage = context.getArgByIndex(0);
      const currentResponse: ServerResponse = context.getArgByIndex(1);
      return this.printHttpLog(currentRequest, currentResponse, next);
    }

    // GraphQL Logging
    const currentRequest: IncomingMessage = context.getArgByIndex(2);
    return this.printGraphQLLog(currentRequest, next);
  }

  private printHttpLog(
    req: IncomingMessage,
    res: ServerResponse,
    next: CallHandler,
  ) {
    const requestBegin = Date.now();
    const contextName = 'HTTP';

    this.logger.log(
      `Request Received (${req['route']['path']}, ${req.method})`,
      contextName,
    );
    this.logger.debug(
      `Request Param (${JSON.stringify(req.method == 'POST' ? req['body'] : req['query'])})`,
      contextName,
    );

    return next.handle().pipe(
      tap((response) => {
        this.logger.log(
          `Response Result (${res.statusCode}, ${Date.now() - requestBegin}ms)`,
          contextName,
        );
        this.logger.debug(
          `Response Body (${JSON.stringify(response)})`,
          contextName,
        );
      }),
    );
  }
  private printGraphQLLog(req: IncomingMessage, next: CallHandler) {
    // Query/Mutation 요청일 때만 로그 기록 (Subscription은 별도 기록)
    if (!req['req']['subscriptions']) {
      const requestBegin = Date.now();
      const contextName = 'GraphQL';

      this.logger.log(
        `Request Received (${req['req']['body']['operationName']})`,
        contextName,
      );
      this.logger.debug(
        `Request Param\n${JSON.stringify(req['req']['body']['query']).replaceAll('\\n', '\n')}`,
        contextName,
      );

      return next.handle().pipe(
        tap((response) => {
          this.logger.log(
            `Request Result (Success, ${Date.now() - requestBegin}ms)`,
            contextName,
          );
          // TSDB 조회 API는 데이터 양이 많아 제외
          const currentOperationName: string =
            req['req']['body']['operationName'] ?? '';
          if (
            currentOperationName.toLowerCase() != 'raws' &&
            currentOperationName.toLowerCase() != 'rawTCodeRange' &&
            currentOperationName.toLowerCase() != 'productInfoReports' &&
            currentOperationName.toLowerCase() != 'productSumReports' &&
            currentOperationName.toLowerCase() != 'toolHistoryReports' &&
            currentOperationName.toLowerCase() != 'operationSumReports' &&
            currentOperationName.toLowerCase() != 'abnormalDetail'
          ) {
            this.logger.debug(
              `Response Body\n${JSON.stringify(response)}`,
              contextName,
            );
          }
        }),
      );
    }

    return next.handle().pipe(tap(() => {}));
  }
}
