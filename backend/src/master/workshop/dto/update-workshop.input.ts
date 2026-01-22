import { InputType, Field } from '@nestjs/graphql';

@InputType()
export class UpdateWorkshopInput {
  @Field(() => String, { description: '공장명' })
  workshopName: string;
}
