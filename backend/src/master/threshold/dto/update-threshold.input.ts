import { InputType, Field, Float, PartialType } from '@nestjs/graphql';

@InputType()
export class UpdateThresholdInput {
  @Field(() => Float, { description: 'C/T 임계치 상한값', nullable: true })
  maxThresholdCt?: number;

  @Field(() => Float, { description: 'C/T 임계치 하한값', nullable: true })
  minThresholdCt?: number;

  @Field(() => Float, { description: 'LoadSum 임계치 상한값', nullable: true })
  maxThresholdLoad?: number;

  @Field(() => Float, { description: 'LoadSum 임계치 하한값', nullable: true })
  minThresholdLoad?: number;

  @Field(() => Float, { description: '오차율 임계치', nullable: true })
  thresholdLoss?: number;

  @Field(() => Float, { description: 'AI 예측 구간', nullable: true })
  predictPeriod?: number;

  @Field(() => Float, { description: 'Tool1 임계치', nullable: true })
  tool1Threshold: number;

  @Field(() => Float, { description: 'Tool2 임계치', nullable: true })
  tool2Threshold: number;

  @Field(() => Float, { description: 'Tool3 임계치', nullable: true })
  tool3Threshold: number;

  @Field(() => Float, { description: 'Tool4 임계치', nullable: true })
  tool4Threshold: number;

  @Field(() => String, { description: '추가 설명', nullable: true })
  remark: string;

  @Field(() => String, { description: '선택 결과', nullable: true })
  selected: string;

  //@Prop({ name: 'treshold_id' })
  //@Field(() => String, { description: '임계치 아이디', nullable: true })
  //thresholdId: string;
}

@InputType()
export class UpdateMultiThresholdInput extends PartialType(
  UpdateThresholdInput,
) {
  @Field(() => String, { description: '임계치 ID' })
  thresholdId: string;
}
