import { ObjectType, Field, Int, Float } from '@nestjs/graphql';
import { Prop, Schema } from '@nestjs/mongoose';

@Schema({ collection: 'tool_history' })
@ObjectType()
export class ToolHistory {
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

  @Prop({ name: 'tool_code' })
  @Field(() => String, { description: '공구 번호' })
  toolCode: string;

  @Prop({ name: 'tool_use_count' })
  @Field(() => Int, { description: '공구 사용 수량' })
  toolUseCount: number;

  // 추가 항목이므로 nullable 처리
  @Prop({ name: 'tool_ct' })
  @Field(() => Float, { description: '공구 CT', nullable: true })
  toolCt?: number;

  // 추가 항목이므로 nullable 처리
  @Prop({ name: 'tool_load_sum' })
  @Field(() => Float, { description: '공구 LoadSum', nullable: true })
  toolLoadSum?: number;

  @Prop({ name: 'tool_use_start_date', default: new Date() })
  @Field(() => Date, { description: '공구 사용 시작 일시' })
  toolUseStartDate: Date;

  @Prop({ name: 'tool_use_end_date', default: new Date() })
  @Field(() => Date, { description: '공구 사용 시작 일시' })
  toolUseEndDate: Date;

  @Prop({ name: 'create_at', default: new Date() })
  @Field(() => Date, { description: '생성 일시' })
  createAt?: Date;

  @Prop({ name: 'update_at', default: new Date() })
  @Field(() => Date, { description: '수정 일시', nullable: true })
  updateAt?: Date;
}
