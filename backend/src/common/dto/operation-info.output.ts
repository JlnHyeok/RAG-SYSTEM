import { Field, ObjectType } from '@nestjs/graphql';

@ObjectType()
export class WorkshopListOutput {
  @Field(() => String, { description: '공장 코드' })
  workshopCode: string;

  @Field(() => String, { description: '공장명' })
  workshopTitle: string;

  @Field(() => [LineListOutput], { description: '라인 배열', nullable: true })
  lineList?: LineListOutput[];
}

@ObjectType()
export class LineListOutput {
  @Field(() => String, { description: '라인 코드', nullable: true })
  lineCode?: string;

  @Field(() => String, { description: '라인명', nullable: true })
  lineTitle?: string;

  @Field(() => [OperationListOutput], {
    description: '공정 배열',
    nullable: true,
  })
  operationList?: OperationListOutput[];
}

@ObjectType()
export class OperationListOutput {
  @Field(() => String, { description: '공정 코드' })
  operationCode: string;

  @Field(() => String, { description: '공정명' })
  operationTitle: string;

  @Field(() => [MachineListOutput], {
    description: '설비 배열',
    nullable: true,
  })
  machineList?: MachineListOutput[];
}

@ObjectType()
export class MachineListOutput {
  @Field(() => String, { description: '설비 코드' })
  machineCode: string;

  @Field(() => String, { description: '설비명' })
  machineTitle: string;
}
