//* NEST Common
import { Logger, Module } from '@nestjs/common';
import { ApolloDriver, ApolloDriverConfig } from '@nestjs/apollo';

//* NEST External Module
import { GraphQLModule } from '@nestjs/graphql';
import { ConfigModule } from '@nestjs/config';
import { MongooseModule } from '@nestjs/mongoose';
import { CacheModule } from '@nestjs/cache-manager';

//* Custom Module
// Util
import { InfluxModule } from './influx/influx.module';
import { PubsubModule } from './pubsub/pubsub.module';
// GraphQL Module
// Master
import { UsersModule } from './master/user/users.module';
import { WorkshopModule } from './master/workshop/workshop.module';
import { LineModule } from './master/line/line.module';
import { OperationModule } from './master/operation/operation.module';
import { MachineModule } from './master/machine/machine.module';
import { ThresholdModule } from './master/threshold/threshold.module';
import { ToolModule } from './master/tool/tool.module';

import { MonitorModule } from './monitor/monitor.module';
// REST API
import { LoggingPlugin } from './logger/logger.plugin';
import { ProductModule } from './product/product.module';
import { AbnormalModule } from './abnormal/abnormal.module';
import { RawModule } from './raw/raw.module';
import { ToolHistoryModule } from './tool-history/tool-history.module';
import { ToolChangeModule } from './tool-change/tool-change.module';
import { CommonModule } from './common/common.module';

@Module({
  imports: [
    // GraphQL Module
    GraphQLModule.forRoot<ApolloDriverConfig>({
      driver: ApolloDriver,
      playground: true,
      autoSchemaFile: 'schema.gql',
      sortSchema: true,
      context: ({ req }) => ({ req }),
      // Pub-Sub 옵션 설정
      subscriptions: {
        'graphql-ws': {
          path: '/graphql',
          // 이벤트 핸들러를 통해 Pub-Sub 로깅 처리
          onClose: (ctx, code, reason) => {
            Logger.log(`Subscribe Closed (${code}, ${reason})`, 'GraphQL');
          },
          onSubscribe: (ctx, message) => {
            Logger.log(
              `Subscribe Received (${message.payload.operationName})`,
              'GraphQL',
            );
            Logger.debug(
              `Subscribe Payload\n(${JSON.stringify(message.payload.query).replaceAll('\\n', '\n')})`,
              'GraphQL',
            );
          },
          onNext: (ctx, message, args, result) => {
            Logger.log(
              `Published Payload(${JSON.stringify(message.payload).replaceAll('\\n', '\n')})`,
              'GraphQL',
            );
          },
        },
      },
      formatError: (error) => {
        return {
          isSuccess: false,
          message: error.message,
        };
      },
    }),

    // Config Module
    ConfigModule.forRoot({
      // env 파일 경로 설정
      envFilePath: '.env',
      isGlobal: true,
    }),

    // Static Module
    // ServeStaticModule.forRoot({
    //   rootPath: join(process.cwd(), 'spectrogram'),
    // }),

    // Mongoose Module
    MongooseModule.forRoot(process.env.MONGODB_URL, {
      dbName: process.env.MONGODB_DATABASE_NAME,
      auth: {
        username: process.env.MONGODB_USER,
        password: process.env.MONGODB_PASSWORD,
      },
      ignoreUndefined: true,
    }),

    // Common Module
    InfluxModule,
    PubsubModule,
    CommonModule,

    // Master Module
    UsersModule,
    WorkshopModule,
    LineModule,
    OperationModule,
    MachineModule,
    ThresholdModule,
    ToolModule,

    // Module
    MonitorModule,
    ProductModule,
    AbnormalModule,
    RawModule,
    ToolHistoryModule,
    ToolChangeModule,
  ],
  providers: [Logger, LoggingPlugin],
})
export class AppModule {}
