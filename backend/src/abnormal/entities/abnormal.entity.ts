import { ObjectType, Field, Float, Int } from '@nestjs/graphql';
import { Prop, Schema } from '@nestjs/mongoose';
import { IsEnum } from 'class-validator';
import { AbnormalMinMax } from 'src/common/dto/common.enum';

@Schema({ collection: 'abnormals' })
@ObjectType()
export class Abnormal {
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
  @Field(() => String, { description: '설비 코드', nullable: true })
  machineCode?: string;

  @Prop({ name: 'product_no' })
  @Field(() => String, { description: '생산 번호' })
  productNo: string;

  @Prop({ name: 'abnormal_code' })
  @Field(() => String, { description: '이상 감지 구분' })
  abnormalCode: string;

  @Prop({ name: 'abnormal_division' })
  @IsEnum(AbnormalMinMax)
  @Field(() => AbnormalMinMax, { description: '상/하한 구분', nullable: true })
  abnormalDivision?: AbnormalMinMax;

  @Prop({ name: 'abnormal_begin_date' })
  @Field(() => Date, { description: '시작 일시' })
  abnormalBeginDate: Date;

  @Prop({ name: 'abnormal_end_date' })
  @Field(() => Date, { description: '종료 일시' })
  abnormalEndDate: Date;

  @Prop({ name: 'abnormal_tool' })
  @Field(() => String, { description: '이상 감지 공구', nullable: true })
  abnormalTool?: string;

  @Prop({ name: 'abnormal_value' })
  @Field(() => Float, { description: '이상 감지 값', nullable: true })
  abnormalValue: number;

  @Prop({ name: 'abnormal_threshold' })
  @Field(() => Float, { description: '임계치', nullable: true })
  abnormalThreshold?: number;

  @Prop({ name: 'abnormal_loss_count' })
  @Field(() => Int, { description: '임계치 데이터 수', nullable: true })
  abnormalLossCount?: number;

  // CNC 정적 파라미터 추가
  @Prop({ name: 'mainProgramNo' })
  @Field(() => String, {
    name: 'mainProgramNo',
    description: '메인 프로그램 번호',
  })
  mainProgramNo: string;

  @Prop({ name: 'subProgramNo' })
  @Field(() => String, {
    name: 'subProgramNo',
    description: '서브 프로그램 번호',
    nullable: true,
  })
  subProgramNo?: string;

  @Prop({ name: 'tCode' })
  @Field(() => String, { name: 'tCode', description: 'T Code' })
  tCode: string;

  @Prop({ name: 'mCode' })
  @Field(() => String, { name: 'mCode', description: 'M Code' })
  mCode: string;

  @Prop({ name: 'fov' })
  @Field(() => Float, { name: 'fov', description: 'FOV(%)' })
  fov: number;

  @Prop({ name: 'sov' })
  @Field(() => Float, { name: 'sov', description: 'SOV(%)' })
  sov: number;

  @Prop({ name: 'offsetX' })
  @Field(() => Float, { name: 'offsetX', description: 'Tool Offset X Axis' })
  offsetX: number;

  @Prop({ name: 'offsetZ' })
  @Field(() => Float, { name: 'offsetZ', description: 'Tool Offset Z Axis' })
  offsetZ: number;

  @Prop({ name: 'feed' })
  @Field(() => Float, { name: 'feed', description: 'Feedrate' })
  feed: number;

  @Prop({ name: 'create_at', default: new Date() })
  @Field(() => Date, { description: '생성 일시' })
  createAt?: Date;

  @Prop({ name: 'update_at', default: new Date() })
  @Field(() => Date, { description: '수정 일시', nullable: true })
  updateAt?: Date;
}

@Schema({ collection: 'abnormalSummary' })
@ObjectType()
export class AbnormalSummary {
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
  @Field(() => String, { description: '설비 코드', nullable: true })
  machineCode?: string;

  @Prop({ name: 'product_no' })
  @Field(() => String, { description: '생산 번호' })
  productNo: string;

  @Prop({ name: 'abnormal_ct' })
  @Field(() => String, { description: 'CT 이상 발생 여부' })
  abnormalCt: string;
  @Prop({ name: 'abnormal_ct_value' })
  @Field(() => Float, { description: 'CT 이상 값' })
  abnormalCtValue: number;
  @Prop({ name: 'abnormal_ct_threshold' })
  @Field(() => Float, { description: 'CT 이상 임계치' })
  abnormalCtThreshold: number;
  @Prop({ name: 'abnormal_ct_min_threshold' })
  @Field(() => Float, { description: 'CT 이상 임계치', nullable: true })
  abnormalMinCtThreshold?: number;

  @Prop({ name: 'abnormal_load' })
  @Field(() => String, { description: 'LoadSum 이상 발생 여부' })
  abnormalLoad: string;
  @Prop({ name: 'abnormal_load_value' })
  @Field(() => Float, { description: 'LoadSum 값' })
  abnormalLoadValue: number;
  @Prop({ name: 'abnormal_load_threshold' })
  @Field(() => Float, { description: 'LoadSum 임계치' })
  abnormalLoadThreshold: number;
  @Prop({ name: 'abnormal_load_min_threshold' })
  @Field(() => Float, { description: 'LoadSum 임계치', nullable: true })
  abnormalMinLoadThreshold?: number;

  @Prop({ name: 'abnormal_ai' })
  @Field(() => String, { description: '부하 이상 발생 여부' })
  abnormalAi: string;
  @Prop({ name: 'abnormal_ai_value' })
  @Field(() => Int, { description: '부하 이상 값' })
  abnormalAiValue: number;
  @Prop({ name: 'abnormal_ai_threshold' })
  @Field(() => Float, { description: '부하 이상 임계치' })
  abnormalAiThreshold: number;
  @Prop({ name: 'abnormal_ai_count' })
  @Field(() => Int, { description: '임계치 데이터 수', nullable: true })
  abnormalAiCount?: number;

  @Prop({ name: 'abnormal_begin_date' })
  @Field(() => Date, { description: '시작 일시' })
  abnormalBeginDate: Date;

  @Prop({ name: 'abnormal_end_date' })
  @Field(() => Date, { description: '종료 일시' })
  abnormalEndDate: Date;

  // CNC 정적 파라미터 추가
  @Prop({ name: 'mainProgramNo' })
  @Field(() => String, {
    name: 'mainProgramNo',
    description: '메인 프로그램 번호',
  })
  mainProgramNo: string;

  @Prop({ name: 'subProgramNo' })
  @Field(() => String, {
    name: 'subProgramNo',
    description: '서브 프로그램 번호',
    nullable: true,
  })
  subProgramNo?: string;

  @Prop({ name: 'tCode' })
  @Field(() => String, { name: 'tCode', description: 'T Code' })
  tCode: string;

  @Prop({ name: 'mCode' })
  @Field(() => String, { name: 'mCode', description: 'M Code' })
  mCode: string;

  @Prop({ name: 'fov' })
  @Field(() => Float, { name: 'fov', description: 'FOV(%)' })
  fov: number;

  @Prop({ name: 'sov' })
  @Field(() => Float, { name: 'sov', description: 'SOV(%)' })
  sov: number;

  @Prop({ name: 'offsetX' })
  @Field(() => Float, { name: 'offsetX', description: 'Tool Offset X Axis' })
  offsetX: number;

  @Prop({ name: 'offsetZ' })
  @Field(() => Float, { name: 'offsetZ', description: 'Tool Offset Z Axis' })
  offsetZ: number;

  @Prop({ name: 'feed' })
  @Field(() => Float, { name: 'feed', description: 'Feedrate' })
  feed: number;

  @Prop({ name: 'create_at', default: new Date() })
  @Field(() => Date, { description: '생성 일시' })
  createAt?: Date;

  @Prop({ name: 'update_at', default: new Date() })
  @Field(() => Date, { description: '수정 일시', nullable: true })
  updateAt?: Date;
}
