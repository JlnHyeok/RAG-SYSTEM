import { InputType, Field } from '@nestjs/graphql';

@InputType()
export class LoginUserInput {
  @Field(() => String, { description: '사용자 ID' })
  userId: string;

  @Field(() => String, { description: '비밀번호' })
  password: string;
}
