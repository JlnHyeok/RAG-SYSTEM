import { Field, ObjectType, PartialType } from '@nestjs/graphql';
import { CommonMutationOutput } from 'src/common/dto/common.output';

@ObjectType()
export class WorkshopMutationOutput extends PartialType(CommonMutationOutput) {
  @Field(() => String, { description: '공장 코드', nullable: true })
  workshopCode?: string;

  @Field(() => String, { description: '공장명', nullable: true })
  workshopName?: string;

  @Field(() => Date, { description: '생성 일시', nullable: true })
  createAt?: Date;

  @Field(() => Date, { description: '수정 일시', nullable: true })
  updateAt?: Date;
}
