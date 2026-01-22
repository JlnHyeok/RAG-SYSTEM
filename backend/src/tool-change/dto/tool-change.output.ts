import { ObjectType, Field, Int, Float } from '@nestjs/graphql';

@ObjectType()
export class ToolChangeLastReportOutput {
  @Field(() => String, { description: '공구 번호' })
  toolCode: string;

  @Field(() => String, { description: '공구명' })
  toolName: string;

  @Field(() => String, { description: '공구 교체 사유 코드' })
  reasonCode: string;

  @Field(() => Date, { description: '공구 교체 일시', nullable: true })
  changeDate?: Date;

  @Field(() => Int, { description: '공구 사용 수량' })
  toolUseCount: number;

  @Field(() => Float, { description: '평균 사용 수량' })
  toolUseCountAvg: number;
}

@ObjectType()
export class ToolChangeAverageReportOutput {
  @Field(() => String, { description: '공구 번호' })
  toolCode: string;

  @Field(() => String, { description: '공구명' })
  toolName: string;

  @Field(() => Int, { description: '공구 교체 횟수' })
  changeCount: number;

  @Field(() => Float, { description: '평균 사용 수량' })
  toolUseCountAvg: number;
}

@ObjectType()
export class ToolChangeAverageMonthlyReportOutput {
  @Field(() => String, { description: '집계 일자' })
  reportDate: string;

  @Field(() => [ToolChangeAverageReportOutput], {
    description: '공구 사용 집계',
  })
  toolUseCount: ToolChangeAverageReportOutput[];
}

@ObjectType()
export class ToolChangeSumReportOutput {
  @Field(() => String, { description: '집계 일자' })
  reportDate: string;

  @Field(() => [ToolChangeSumOutput], { description: '공구 사용 집계' })
  toolUseCount: ToolChangeSumOutput[];
}
@ObjectType()
export class ToolChangeSumOutput {
  @Field(() => String, { description: '공구 번호' })
  toolCode: string;

  @Field(() => String, { description: '공구명' })
  toolName: string;

  @Field(() => Float, { description: '공구 사용 수량' })
  toolUseCountSum: number;
}
