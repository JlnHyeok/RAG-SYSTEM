import { Provider } from '@nestjs/common';
import { PubSub } from 'graphql-subscriptions';
import { Raw } from './raw/entities/raw.entity';
import { ProductInflux } from './product/entities/product-influx.entity';
import { ToolHistoryInflux } from './tool-history/entities/tool-history-influx.entity';

// import { Raw } from './rams_bak/raw/entities/raw.entity';

// TODO: Provider 구현 위치 확인 필요

// Provider Key
export const RAW_ENTITY = 'RAW_ENTITY';
export const PRODUCT_ENTITY = 'PRODUCT_ENTITY';
export const TOOL_ENTITY = 'TOOL_ENTITY';
export const PUB_SUB = 'PUB_SUB';

// Provider 생성 함수
export function createRawProvider(): Provider {
  return {
    provide: RAW_ENTITY,
    useClass: Raw,
  };
}

export function createProductInfluxProvider(): Provider {
  return {
    provide: PRODUCT_ENTITY,
    useClass: ProductInflux,
  };
}

export function createToolInfluxProvider(): Provider {
  return {
    provide: TOOL_ENTITY,
    useClass: ToolHistoryInflux,
  };
}

export function createPubSubProvider(): Provider {
  return {
    provide: PUB_SUB,
    useClass: PubSub,
  };
}
