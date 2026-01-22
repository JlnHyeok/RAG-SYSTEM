import { Module } from '@nestjs/common';

import { PUB_SUB, createPubSubProvider } from './pubsub.provider';

@Module({
  providers: [createPubSubProvider()],
  exports: [PUB_SUB],
})
export class PubsubModule {}
