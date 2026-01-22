import { InputType, Int, Field, Float } from '@nestjs/graphql';

@InputType()
export class CreateToolInput {
  @Field(() => String, { description: '설비 코드' })
  machineCode: string;

  @Field(() => String, { description: '공구 번호' })
  toolCode: string;

  @Field(() => String, { description: '공구명' })
  toolName: string;

  @Field(() => Int, { description: '공구 순서' })
  toolOrder: number;

  @Field(() => Int, { description: '한계 수명' })
  maxCount: number;

  @Field(() => Float, { description: '경고 임계치' })
  warnRate: number;
}
