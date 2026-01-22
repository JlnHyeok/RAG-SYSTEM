import { InputType, Field, Int, Float } from '@nestjs/graphql';

@InputType()
export class UpdateToolInput {
  @Field(() => String, { description: '공구명', nullable: true })
  toolName?: string;

  @Field(() => Int, { description: '공구 순서', nullable: true })
  toolOrder?: number;

  @Field(() => Int, { description: '한계 수명', nullable: true })
  maxCount?: number;

  @Field(() => Float, { description: '경고 임계치', nullable: true })
  warnRate?: number;
}
