import { Field, InputType } from '@nestjs/graphql';

@InputType()
export class FilterWorkshopInput {
  @Field(() => [String], { description: '공장 코드 목록' })
  workshopCodes: [string];
}
