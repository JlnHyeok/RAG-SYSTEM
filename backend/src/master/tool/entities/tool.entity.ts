import { ObjectType, Field, Int, Float } from '@nestjs/graphql';
import { Prop, Schema } from '@nestjs/mongoose';

@Schema({ collection: 'toolMaster' })
@ObjectType()
export class Tool {
  @Prop({ name: 'machine_code' })
  @Field(() => String, { description: '설비 코드' })
  machineCode: string;

  @Prop({ name: 'tool_code' })
  @Field(() => String, { description: '공구 번호' })
  toolCode: string;

  @Prop({ name: 'sub_tool_code' })
  @Field(() => String, { description: '예비 공구 번호', nullable: true })
  subToolCode?: string;

  @Prop({ name: 'tool_name' })
  @Field(() => String, { description: '공구명' })
  toolName: string;

  @Prop({ name: 'tool_order' })
  @Field(() => Int, { description: '공구 순서' })
  toolOrder: number;

  @Prop({ name: 'max_count' })
  @Field(() => Int, { description: '한계 수명' })
  maxCount: number;

  @Prop({ name: 'warn_rate' })
  @Field(() => Float, { description: '경고 임계치' })
  warnRate: number;

  @Prop({ name: 'create_at', default: new Date() })
  @Field(() => Date, { description: '생성 일시' })
  createAt?: Date;

  @Prop({ name: 'update_at', default: new Date() })
  @Field(() => Date, { description: '수정 일시', nullable: true })
  updateAt?: Date;
}
