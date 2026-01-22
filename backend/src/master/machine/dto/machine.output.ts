import { Field, Int, ObjectType, PartialType } from '@nestjs/graphql';
import { CommonMutationOutput } from 'src/common/dto/common.output';

@ObjectType()
export class MachineQueryOutput {
  @Field(() => String, { description: '공장 코드' })
  workshopCode: string;

  @Field(() => String, { description: '라인 코드' })
  lineCode: string;

  @Field(() => String, { description: '공정 코드' })
  opCode: string;

  @Field(() => String, { description: '공정명' })
  opName: string;

  @Field(() => String, { description: '설비 코드' })
  machineCode: string;

  @Field(() => String, { description: '설비명' })
  machineName: string;

  @Field(() => String, { description: '설비 IP' })
  machineIp: string;

  @Field(() => Int, { description: '설비 포트' })
  machinePort: number;

  @Field(() => Date, { description: '생성 일시' })
  createAt: Date;

  @Field(() => Date, { description: '수정 일시', nullable: true })
  updateAt?: Date;
}

@ObjectType()
export class MachineMutationOutput extends PartialType(CommonMutationOutput) {
  @Field(() => String, { description: '공장 코드', nullable: true })
  workshopCode?: string;

  @Field(() => String, { description: '라인 코드', nullable: true })
  lineCode?: string;

  @Field(() => String, { description: '공정 코드', nullable: true })
  opCode?: string;

  @Field(() => String, { description: '설비 코드', nullable: true })
  machineCode?: string;

  @Field(() => String, { description: '설비명', nullable: true })
  machineName?: string;

  @Field(() => String, { description: '설비 IP', nullable: true })
  machineIp?: string;

  @Field(() => Int, { description: '설비 포트', nullable: true })
  machinePort?: number;

  @Field(() => Date, { description: '생성 일시', nullable: true })
  createAt?: Date;

  @Field(() => Date, { description: '수정 일시', nullable: true })
  updateAt?: Date;
}
