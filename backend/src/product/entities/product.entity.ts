import { ObjectType, Field, Int, Float } from '@nestjs/graphql';
import { Prop, Schema } from '@nestjs/mongoose';

const COLLECTION_NAME = 'products';

@Schema({ collection: COLLECTION_NAME })
@ObjectType()
export class Product {
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

  @Prop({ name: 'product_begin_date' })
  @Field(() => Date, { description: '시작 일시' })
  startTime: Date;

  @Prop({ name: 'product_end_date' })
  @Field(() => Date, { description: '종료 일시' })
  endTime: Date;

  @Prop({ name: 'product_result' })
  @Field(() => String, { description: '생산 결과' })
  productResult: string;

  @Prop({ name: 'complete_status' })
  @Field(() => Int, { description: '생산 완료 상태', nullable: true })
  completeStatus?: number;

  @Prop({ name: 'product_count' })
  @Field(() => Int, { description: '생산 수량' })
  count: number;

  @Prop({ name: 'product_ct' })
  @Field(() => Float, { description: 'Cycle Time' })
  ct: number;

  @Prop({ name: 'product_ct_result' })
  @Field(() => String, { description: '생산 결과' })
  ctResult: string;

  @Prop({ name: 'product_ai' })
  @Field(() => Float, { description: 'AI 분석 결과' })
  ai: number;

  @Prop({ name: 'product_ai_result' })
  @Field(() => String, { description: '생산 결과' })
  aiResult: string;

  @Prop({ name: 'product_load_sum' })
  @Field(() => Float, { description: '부하 Sum' })
  loadSum: number;

  @Prop({ name: 'product_load_sum_result' })
  @Field(() => String, { description: '생산 결과' })
  loadSumResult: string;

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
  })
  subProgramNo: string;

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
