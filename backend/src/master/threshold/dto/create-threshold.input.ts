import { InputType, Field, Float, Int } from '@nestjs/graphql';

@InputType()
export class CreateThresholdInput {
  @Field(() => String, { description: '공장 코드', nullable: true })
  workshopCode?: string;

  @Field(() => String, { description: '라인 코드', nullable: true })
  lineCode?: string;

  @Field(() => String, { description: '공정 코드', nullable: true })
  opCode?: string;

  @Field(() => String, { description: '설비 코드', nullable: true })
  machineCode?: string;

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
  tool1Threshold?: number;

  @Field(() => Float, { description: 'Tool2 임계치', nullable: true })
  tool2Threshold?: number;

  @Field(() => Float, { description: 'Tool3 임계치', nullable: true })
  tool3Threshold?: number;

  @Field(() => Float, { description: 'Tool4 임계치', nullable: true })
  tool4Threshold?: number;

  @Field(() => String, { description: '추가 설명', nullable: true })
  remark?: string;

  @Field(() => String, { description: '선택 결과', nullable: true })
  selected?: string;
}
