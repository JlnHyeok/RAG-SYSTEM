import { InputType, Field } from '@nestjs/graphql';

@InputType()
export class CreateWorkshopInput {
  @Field(() => String, { description: '공장 코드' })
  workshopCode: string;

  @Field(() => String, { description: '공장명' })
  workshopName: string;
}
