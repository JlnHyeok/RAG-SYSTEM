import { Field, ObjectType } from '@nestjs/graphql';

@ObjectType()
export class LogoutUserOutput {
  @Field(() => Boolean, { description: '로그 아웃 성공 여부' })
  isSuccess: boolean;

  @Field(() => Date, { description: '로그 아웃 일시', nullable: true })
  logoutDate?: Date;
}
