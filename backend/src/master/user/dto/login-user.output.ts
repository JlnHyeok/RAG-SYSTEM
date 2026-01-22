import { Field, ObjectType } from '@nestjs/graphql';

@ObjectType()
export class LoginUserOutput {
  @Field(() => Boolean, { description: '로그인 성공 여부' })
  isSuccess: boolean;

  @Field(() => Date, { description: '로그인 일시', nullable: true })
  loginDate?: Date;

  @Field(() => String, { description: 'Access Token', nullable: true })
  accessToken?: string;

  @Field(() => String, { description: '비밀번호 재설정 여부', nullable: true })
  resetFlag?: string;
}
