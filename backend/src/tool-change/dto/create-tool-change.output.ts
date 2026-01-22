import { ObjectType, PartialType, Field, Int } from '@nestjs/graphql';
import { CommonMutationOutput } from 'src/common/dto/common.output';

@ObjectType()
export class CreateToolChangeOutput extends PartialType(CommonMutationOutput) {
  @Field(() => String, { description: '공구 번호', nullable: true })
  toolCode?: string;

  @Field(() => String, { description: '공구 교체 사유 코드', nullable: true })
  reasonCode?: string;

  @Field(() => Int, { description: '공구 사용 수량', nullable: true })
  toolUseCount?: number;

  @Field(() => Date, { description: '공구 교체 일시', nullable: true })
  changeDate?: Date;
}
