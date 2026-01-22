import { InputType, Field } from '@nestjs/graphql';

@InputType()
export class UpdateLineInput {
  @Field(() => String)
  lineName: string;
}
