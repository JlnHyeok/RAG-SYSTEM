import { forwardRef, Module } from '@nestjs/common';
import { UsersService } from './users.service';
import { UsersResolver } from './users.resolver';
import { MongooseModule, SchemaFactory } from '@nestjs/mongoose';
import { User } from './entities/user.entity';
import { PubsubModule } from 'src/pubsub/pubsub.module';
import { JwtModule } from '@nestjs/jwt';
import { ConfigService } from '@nestjs/config';
import { WorkshopModule } from '../workshop/workshop.module';
import { LineModule } from '../line/line.module';
import { OperationModule } from '../operation/operation.module';
import { CacheModule } from '@nestjs/cache-manager';

@Module({
  imports: [
    MongooseModule.forFeature([
      { name: User.name, schema: SchemaFactory.createForClass(User) },
    ]),
    forwardRef(() => WorkshopModule),
    forwardRef(() => LineModule),
    forwardRef(() => OperationModule),
    PubsubModule,
    JwtModule.registerAsync({
      global: true,
      inject: [ConfigService],
      useFactory: (config: ConfigService) => ({
        secret: config.get<string>('JWT_SECRET_KEY'),
        signOptions: { expiresIn: config.get<string>('JWT_EXPIRES_IN') },
      }),
    }),
    CacheModule.register({
      isGlobal: true,
    }),
  ],
  providers: [UsersResolver, UsersService],
})
export class UsersModule {}
