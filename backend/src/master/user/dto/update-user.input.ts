import { Field, InputType, Int } from '@nestjs/graphql';

@InputType()
export class UpdateUserInput {
  @Field(() => String, { description: '비밀번호', nullable: true })
  password?: string;

  @Field(() => String, { description: '사용자 이름', nullable: true })
  userName?: string;

  @Field(() => Int, { description: '사용자 권한', nullable: true })
  userRole?: number;

  @Field(() => String, { description: '비밀번호 재설정 여부', nullable: true })
  resetFlag?: string;
}
