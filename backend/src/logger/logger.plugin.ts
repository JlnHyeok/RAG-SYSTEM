import {
  ApolloServerPlugin,
  GraphQLRequestContextWillSendResponse,
  GraphQLRequestListener,
} from '@apollo/server';
import { Plugin } from '@nestjs/apollo';
import { Inject, LoggerService, Logger } from '@nestjs/common';

@Plugin()
export class LoggingPlugin implements ApolloServerPlugin {
  constructor(
    @Inject(Logger)
    private readonly logger: LoggerService,
  ) {}

  async requestDidStart(): Promise<GraphQLRequestListener<any>> {
    const gqlLogger = this.logger;

    return {
      async willSendResponse(
        requestContext: GraphQLRequestContextWillSendResponse<any>,
      ) {
        const { request: currentRequest, errors: currentErrors } =
          requestContext;
        const operationName = `${currentRequest.operationName}`;

        if (operationName == 'IntrospectionQuery') {
          return;
        }

        if (currentErrors) {
          gqlLogger.log(`Request Received (${operationName})`, 'GraphQL');
          gqlLogger.debug(
            `Request Param\n${JSON.stringify(currentRequest.query).replaceAll('\\n', '\n')}`,
            'GraphQL',
          );
          gqlLogger.error(
            `Request Error (${currentErrors[0].message})`,
            'GraphQL',
          );

          return;
        }
      },
    };
  }
}
