import { InputType, Field } from '@nestjs/graphql';

@InputType()
export class FilterOperationInput {
  @Field(() => String, { description: '공장 코드', nullable: true })
  workshopCode?: string;

  @Field(() => String, { description: '라인 코드', nullable: true })
  lineCode?: string;
}
