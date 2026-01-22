import { Field, InputType } from '@nestjs/graphql';

@InputType()
export class FilterUserInput {
  @Field(() => [String], { description: '사용자 ID 목록' })
  userIds: [string];
}
