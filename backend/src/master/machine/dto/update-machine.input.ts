import { InputType, Field, Int } from '@nestjs/graphql';

@InputType()
export class UpdateMachineInput {
  @Field(() => String, { description: '설비명', nullable: true })
  machineName?: string;

  @Field(() => String, { description: '설비 IP', nullable: true })
  machineIp?: string;

  @Field(() => Int, { description: '설비 포트', nullable: true })
  machinePort?: number;
}
