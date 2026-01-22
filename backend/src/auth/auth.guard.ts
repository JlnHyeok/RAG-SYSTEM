import { CACHE_MANAGER, Cache } from '@nestjs/cache-manager';
import {
  CanActivate,
  ExecutionContext,
  Inject,
  Injectable,
  UnauthorizedException,
} from '@nestjs/common';
import { GqlExecutionContext } from '@nestjs/graphql';
import { JwtService } from '@nestjs/jwt';
import { IncomingMessage } from 'http';

@Injectable()
export class AuthGuard implements CanActivate {
  constructor(
    private readonly jwtService: JwtService,
    @Inject(CACHE_MANAGER)
    private readonly cacheManager: Cache,
  ) {}

  async canActivate(context: ExecutionContext): Promise<boolean> {
    // 환경 변수를 통해 Guard 사용 여부를 판단 (개발 편의성을 위해)
    if (process.env.GUARD_ENABLE == 'N') {
      return true;
    }

    const ctx = GqlExecutionContext.create(context);
    const request: IncomingMessage = ctx.getContext().req;

    // * 인증 헤더 유효성 검증
    if (!request.headers.authorization) {
      // 인증 헤더가 없을 경우 Unauthorized
      throw new UnauthorizedException();
    }

    // * 인증 헤더 형식 유효성 검증
    const authString = request.headers.authorization;
    const authArray = authString.split(' ');

    if (authArray.length < 2) {
      // 토큰 문자열이 맞지 않을 경우 Unauthorized
      throw new UnauthorizedException();
    }

    // * JWT 캐시 비교
    const currentToken = authArray[1];
    const decoded = this.jwtService.decode(currentToken);
    const activeToken = await this.cacheManager.get(decoded.userId);
    if (!activeToken || currentToken != activeToken) {
      // 캐싱된 토큰과 동일하지 않을 경우 Unauthorized
      throw new UnauthorizedException();
    }

    // * JWT 유효성 검증
    await this.jwtService.verifyAsync(authArray[1], {
      secret: process.env.JWT_SECRET_KEY,
    });

    return true;
  }
}
