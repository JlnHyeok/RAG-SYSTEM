import { Field, Float, Int, ObjectType, PartialType } from '@nestjs/graphql';
import { CommonMutationOutput } from 'src/common/dto/common.output';

@ObjectType()
export class ToolMutationOutput extends PartialType(CommonMutationOutput) {
  @Field(() => String, { description: '설비 코드', nullable: true })
  machineCode?: string;

  @Field(() => String, { description: '공구 번호', nullable: true })
  toolCode?: string;

  @Field(() => String, { description: '공구명', nullable: true })
  toolName?: string;

  @Field(() => Int, { description: '공구 순서', nullable: true })
  toolOrder?: number;

  @Field(() => Int, { description: '한계 수명', nullable: true })
  maxCount?: number;

  @Field(() => Float, { description: '경고 임계치', nullable: true })
  warnRate?: number;

  @Field(() => Date, { description: '생성 일시', nullable: true })
  createAt?: Date;

  @Field(() => Date, { description: '수정 일시', nullable: true })
  updateAt?: Date;
}
