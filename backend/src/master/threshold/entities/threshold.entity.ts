import { ObjectType, Field, Float, Int } from '@nestjs/graphql';
import { Prop, Schema } from '@nestjs/mongoose';

@Schema({ collection: 'thresholdMaster' })
@ObjectType()
export class Threshold {
  @Prop({ name: 'workshop_code' })
  @Field(() => String, { description: '공장 코드' })
  workshopCode: string;

  @Prop({ name: 'line_code' })
  @Field(() => String, { description: '라인 코드' })
  lineCode: string;

  @Prop({ name: 'op_code' })
  @Field(() => String, { description: '공정 코드' })
  opCode: string;

  @Prop({ name: 'machine_code' })
  @Field(() => String, { description: '설비 코드' })
  machineCode: string;

  @Prop({ name: 'threshold_ct' })
  @Field(() => Float, { description: 'C/T 임계치 상한값' })
  maxThresholdCt: number;

  @Prop({ name: 'min_threshold_ct' })
  @Field(() => Float, { description: 'C/T 임계치 하한값' })
  minThresholdCt: number;

  @Prop({ name: 'threshold_load' })
  @Field(() => Float, { description: 'LoadSum 임계치 상한값' })
  maxThresholdLoad: number;

  @Prop({ name: 'min_threshold_load' })
  @Field(() => Float, { description: 'LoadSum 임계치 하한값' })
  minThresholdLoad: number;

  @Prop({ name: 'threshold_loss' })
  @Field(() => Float, { description: '오차율 임계치' })
  thresholdLoss: number;

  @Prop({ name: 'predict_period' })
  @Field(() => Float, { description: 'AI 예측 구간' })
  predictPeriod: number;

  @Prop({ name: 'create_at', default: new Date() })
  @Field(() => Date, { description: '생성 일시' })
  createAt?: Date;

  @Prop({ name: 'update_at', default: new Date() })
  @Field(() => Date, { description: '수정 일시', nullable: true })
  updateAt?: Date;

  @Prop({ name: 'tool1_threshold' })
  @Field(() => Float, { description: 'Tool1 임계치' })
  tool1Threshold: number;

  @Prop({ name: 'tool2_threshold' })
  @Field(() => Float, { description: 'Tool2 임계치' })
  tool2Threshold: number;

  @Prop({ name: 'tool3_threshold' })
  @Field(() => Float, { description: 'Tool3 임계치' })
  tool3Threshold: number;

  @Prop({ name: 'tool4_treshhold' })
  @Field(() => Float, { description: 'Tool4 임계치' })
  tool4Threshold: number;

  @Prop({ name: 'additional_explanation' })
  @Field(() => String, { description: '추가 설명' })
  remark: string;

  @Prop({ name: 'selected_threshold' })
  @Field(() => String, { description: '선택 결과' })
  selected: string;
}
