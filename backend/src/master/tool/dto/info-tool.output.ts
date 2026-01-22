import { Field, Float, Int, ObjectType, PartialType } from '@nestjs/graphql';
import { CommonMutationOutput } from 'src/common/dto/common.output';

@ObjectType()
export class ToolInfoOutput extends PartialType(CommonMutationOutput) {
  @Field(() => String, { description: '공구 번호' })
  toolCode: string;

  @Field(() => Int, { description: '공구 순서' })
  toolOrder: number;

  @Field(() => Int, { description: '한계 수명' })
  maxCount: number;

  @Field(() => Float, { description: '경고 임계치' })
  warnRate: number;
}
