import { ObjectType, Field, Int } from '@nestjs/graphql';
import { Prop, Schema } from '@nestjs/mongoose';

@Schema({ collection: 'machineMaster' })
@ObjectType()
export class Machine {
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

  @Prop({ name: 'machine_name' })
  @Field(() => String, { description: '설비명' })
  machineName: string;

  @Prop({ name: 'machine_ip' })
  @Field(() => String, { description: '설비 IP' })
  machineIp: string;

  @Prop({ name: 'machine_port' })
  @Field(() => Int, { description: '설비 포트' })
  machinePort: number;

  @Prop({ name: 'create_at', default: new Date() })
  @Field(() => Date, { description: '생성 일시' })
  createAt: Date;

  @Prop({ name: 'update_at', default: new Date() })
  @Field(() => Date, { description: '수정 일시', nullable: true })
  updateAt?: Date;
}
