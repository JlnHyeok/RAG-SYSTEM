import { InputType, Field } from '@nestjs/graphql';

@InputType()
export class FilterToolInput {
  // @Field(() => String, { description: '공장 코드', nullable: true })
  // workshopCode?: string;

  // @Field(() => String, { description: '라인 코드', nullable: true })
  // lineCode?: string;

  // @Field(() => String, { description: '공정 코드', nullable: true })
  // opCode?: string;

  @Field(() => String, { description: '설비 코드', nullable: true })
  machineCode?: string;
}

@InputType()
export class FilterToolCodeInput {
  @Field(() => String, { description: '설비 코드' })
  machineCode: string;

  @Field(() => String, { description: '공구 번호' })
  toolCode: string;
}
