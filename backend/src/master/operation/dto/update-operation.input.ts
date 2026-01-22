import { InputType, Field } from '@nestjs/graphql';

@InputType()
export class UpdateOperationInput {
  @Field(() => String, { description: '공정명' })
  opName: string;
}
