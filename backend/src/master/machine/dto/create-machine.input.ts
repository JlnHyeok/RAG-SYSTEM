import { InputType, Field, Int } from '@nestjs/graphql';

@InputType()
export class CreateMachineInput {
  @Field(() => String, { description: '공장 코드' })
  workshopCode: string;

  @Field(() => String, { description: '라인 코드' })
  lineCode: string;

  @Field(() => String, { description: '공정 코드' })
  opCode: string;

  @Field(() => String, { description: '설비 코드' })
  machineCode: string;

  @Field(() => String, { description: '설비명' })
  machineName: string;

  @Field(() => String, { description: '설비 IP' })
  machineIp: string;

  @Field(() => Int, { description: '설비 포트' })
  machinePort: number;
}
