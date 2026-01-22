import {
  CanActivate,
  ExecutionContext,
  ForbiddenException,
  Injectable,
} from '@nestjs/common';
import { Reflector } from '@nestjs/core';
import { Observable } from 'rxjs';
import { ROLES_KEY } from './role.decorator';
import { JwtService } from '@nestjs/jwt';
import { GqlExecutionContext } from '@nestjs/graphql';
import { IncomingMessage } from 'http';

@Injectable()
export class RoleGuard implements CanActivate {
  constructor(
    private readonly jwtService: JwtService,
    private reflector: Reflector,
  ) {}

  canActivate(
    context: ExecutionContext,
  ): boolean | Promise<boolean> | Observable<boolean> {
    // 환경 변수를 통해 Guard 사용 여부를 판단 (개발 편의성을 위해)
    if (process.env.GUARD_ENABLE == 'N') {
      return true;
    }

    // Request 객체 초기화
    const ctx = GqlExecutionContext.create(context);
    const request: IncomingMessage = ctx.getContext().req;

    // 기준 권한 초기화
    const requiredRole = this.reflector.getAllAndOverride<number>(ROLES_KEY, [
      context.getHandler(),
      context.getClass(),
    ]);

    // JWT Payload 초기화
    const jwtPayload = this.jwtService.decode(
      request.headers.authorization.split(' ')[1],
    );

    // 권한 유효성 검증
    if (jwtPayload['userRole'] < requiredRole) {
      // 권한이 없을 경우 Forbidden
      throw new ForbiddenException();
    }

    return true;
  }
}
