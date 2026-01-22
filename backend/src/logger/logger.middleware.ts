import {
  Inject,
  Injectable,
  Logger,
  LoggerService,
  NestMiddleware,
} from '@nestjs/common';
import { NextFunction, Request, Response } from 'express';

@Injectable()
export class LoggerMiddleware implements NestMiddleware {
  constructor(
    @Inject(Logger)
    private readonly logger: LoggerService,
  ) {}

  use(req: Request, res: Response, next: NextFunction) {
    next();
  }
}
