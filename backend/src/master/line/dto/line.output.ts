import { Field, ObjectType, PartialType } from '@nestjs/graphql';
import { CommonMutationOutput } from 'src/common/dto/common.output';

@ObjectType()
export class LineMutationOutput extends PartialType(CommonMutationOutput) {
  @Field(() => String, { description: '공장 코드', nullable: true })
  workshopCode?: string;

  @Field(() => String, { description: '라인 코드', nullable: true })
  lineCode?: string;

  @Field(() => String, { description: '라인명', nullable: true })
  lineName?: string;

  @Field(() => Date, { description: '생성 일시', nullable: true })
  createAt?: Date;

  @Field(() => Date, { description: '수정 일시', nullable: true })
  updateAt?: Date;
}
