import { Field, Float, ObjectType, PartialType } from '@nestjs/graphql';
import { IsNumber } from 'class-validator';
import { CommonMutationOutput } from 'src/common/dto/common.output';

@ObjectType()
export class ThresholdQueryOutput {
  @Field(() => String, { description: '공장 코드', nullable: true })
  workshopCode?: string;

  @Field(() => String, { description: '라인 코드', nullable: true })
  lineCode?: string;

  @Field(() => String, { description: '공정 코드' })
  opCode: string;

  @Field(() => String, { description: '공정 코드' })
  opName: string;

  @Field(() => String, { description: '설비 코드' })
  machineCode: string;

  @Field(() => String, { description: '공정 코드' })
  machineName: string;

  @Field(() => Float, { description: 'C/T 임계치 상한값', nullable: true })
  maxThresholdCt?: number;

  @Field(() => Float, {
    description: 'C/T 임계치 하한값',
    nullable: true,
  })
  minThresholdCt: number;

  @Field(() => Float, { description: 'LoadSum 임계치 상한값', nullable: true })
  maxThresholdLoad?: number;

  @Field(() => Float, {
    description: 'LoadSum 임계치 하한값',
    nullable: true,
  })
  minThresholdLoad: number;

  @Field(() => Float, { description: '오차율 임계치', nullable: true })
  thresholdLoss?: number;

  @Field(() => Float, { description: 'AI 예측 구간', nullable: true })
  predictPeriod?: number;

  @Field(() => Date, { description: '생성 일시', nullable: true })
  createAt?: Date;

  @Field(() => Date, { description: '수정 일시', nullable: true })
  updateAt?: Date;

  @Field(() => Float, { description: 'Tool1 임계치', nullable: true })
  tool1Threshold?: number;

  @Field(() => Float, { description: 'Tool2 임계치', nullable: true })
  tool2Threshold?: number;

  @Field(() => Float, { description: 'Tool3 임계치', nullable: true })
  tool3Threshold?: number;

  @Field(() => Float, { description: 'Tool4 임계치', nullable: true })
  tool4Threshold?: number;

  @Field(() => String, { description: 'Tool1 공구명', nullable: true })
  tool1Name?: string;

  @Field(() => String, { description: 'Tool2 공구명', nullable: true })
  tool2Name?: string;

  @Field(() => String, { description: 'Tool3 공구명', nullable: true })
  tool3Name?: string;

  @Field(() => String, { description: 'Tool4 공구명', nullable: true })
  tool4Name?: string;

  @Field(() => String, { description: '추가 설명', nullable: true })
  remark?: string;

  @Field(() => String, { description: '선택 결과', nullable: true })
  selected?: string;

  @Field(() => String, { description: '임계치 아이디', nullable: true })
  thresholdId?: string;
}

@ObjectType()
export class ThresholdMutationOutput extends PartialType(CommonMutationOutput) {
  // @Field(() => String, { description: '공장 코드', nullable: true })
  // workshopCode?: string;

  // @Field(() => String, { description: '라인 코드', nullable: true })
  // lineCode?: string;

  // @Field(() => String, { description: '공정 코드', nullable: true })
  // opCode?: string;

  @Field(() => String, { description: '설비 코드', nullable: true })
  machineCode?: string;

  @Field(() => Float, { description: 'C/T 임계치', nullable: true })
  maxThresholdCt?: number;

  @Field(() => Float, { description: 'C/T 임계치 하한값', nullable: true })
  minThresholdCt?: number;

  @Field(() => Float, { description: 'LoadSum 임계치', nullable: true })
  maxThresholdLoad?: number;

  @Field(() => Float, { description: 'LoadSum 임계치 하한값', nullable: true })
  minThresholdLoad?: number;

  @Field(() => Float, { description: '오차율 임계치', nullable: true })
  thresholdLoss?: number;

  @Field(() => Float, { description: 'AI 예측 구간', nullable: true })
  predictPeriod?: number;

  @Field(() => Date, { description: '생성 일시', nullable: true })
  createAt?: Date;

  @Field(() => Date, { description: '수정 일시', nullable: true })
  updateAt?: Date;

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

  @Field(() => String, { description: '임계치 아이디', nullable: true })
  thresholdId?: string;
}

@ObjectType()
export class ThresholdPredictOutput {
  @IsNumber()
  tool1Threshold?: number;

  @IsNumber()
  tool2Threshold?: number;

  @IsNumber()
  tool3Threshold?: number;

  @IsNumber()
  tool4Threshold?: number;
}
