import { Field, Float, Int, ObjectType, PartialType } from '@nestjs/graphql';
import { CommonMutationOutput } from 'src/common/dto/common.output';

@ObjectType()
export class UserMutationOutput extends PartialType(CommonMutationOutput) {
  @Field(() => String, { description: '사용자 ID', nullable: true })
  userId?: string;

  @Field(() => String, { description: '비밀번호', nullable: true })
  password?: string;

  @Field(() => String, { description: '사용자 이름', nullable: true })
  userName?: string;

  @Field(() => Int, { description: '사용자 권한', nullable: true })
  userRole?: number;

  @Field(() => String, { description: '패스워드 재설정 여부', nullable: true })
  resetFlag?: string;

  @Field(() => Date, { description: '생성 일시', nullable: true })
  createAt?: Date;

  @Field(() => Date, { description: '수정 일시', nullable: true })
  updateAt?: Date;
}
