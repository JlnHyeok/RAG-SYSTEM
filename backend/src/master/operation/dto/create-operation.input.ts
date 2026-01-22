import { InputType, Field } from '@nestjs/graphql';

@InputType()
export class CreateOperationInput {
  @Field(() => String, { description: '공장 코드' })
  workshopCode: string;

  @Field(() => String, { description: '라인 코드' })
  lineCode: string;

  @Field(() => String, { description: '공정 코드' })
  opCode: string;

  @Field(() => String, { description: '공정명' })
  opName: string;
}
