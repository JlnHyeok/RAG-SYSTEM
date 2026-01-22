import { InputType, Field } from '@nestjs/graphql';

@InputType()
export class CreateLineInput {
  @Field(() => String, { description: '공장 코드' })
  workshopCode: string;

  @Field(() => String, { description: '라인 코드' })
  lineCode: string;

  @Field(() => String, { description: '라인명' })
  lineName: string;
}
