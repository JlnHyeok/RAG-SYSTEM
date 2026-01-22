import { InputType, Field } from '@nestjs/graphql';

@InputType()
export class UpdateAbnormalSummaryInput {
  @Field(() => String, { description: '부하 이상 발생 여부' })
  abnormalAi: string;
  @Field(() => String, { description: '부하 이상 값' })
  abnormalAiValue: number;
}
