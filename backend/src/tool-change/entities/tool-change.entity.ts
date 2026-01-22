import { ObjectType, Field, Int } from '@nestjs/graphql';
import { Prop, Schema } from '@nestjs/mongoose';

@Schema({ collection: 'tool_change' })
@ObjectType()
export class ToolChange {
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

  @Prop({ name: 'reason_code' })
  @Field(() => String, { description: '공구 교체 사유 코드' })
  reasonCode: string;

  @Prop({ name: 'tool_use_count' })
  @Field(() => Int, { description: '공구 사용 수량' })
  toolUseCount: number;

  @Prop({ name: 'change_date', default: new Date() })
  @Field(() => Date, { description: '공구 교체 일시' })
  changeDate: Date;

  @Prop({ name: 'create_at', default: new Date() })
  @Field(() => Date, { description: '생성 일시' })
  createAt?: Date;

  @Prop({ name: 'update_at', default: new Date() })
  @Field(() => Date, { description: '수정 일시', nullable: true })
  updateAt?: Date;
}
