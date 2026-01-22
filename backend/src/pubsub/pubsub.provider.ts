import { Provider } from '@nestjs/common';
import { PubSub } from 'graphql-subscriptions';

export const PUB_SUB = 'PUB_SUB';

export function createPubSubProvider(): Provider {
  return {
    provide: PUB_SUB,
    useClass: PubSub,
  };
}
