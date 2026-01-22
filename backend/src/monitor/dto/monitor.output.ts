import { Field, Int, ObjectType } from '@nestjs/graphql';
import { ProductLastOutput } from 'src/product/dto/product.output';
import { RawOutput } from 'src/raw/dto/raw.output';
import { ToolCountLastOutput } from 'src/tool-history/dto/tool-history.output';

@ObjectType()
export class MonitorOutput {
  @Field(() => String, { description: '공정 코드' })
  opCode: string;

  @Field(() => String, { description: '공정명' })
  opName: string;

  @Field(() => Int, { description: '가동 시간(ms)' })
  operationTime: number;

  @Field(() => ProductLastOutput, { description: '생산 정보', nullable: true })
  product?: ProductLastOutput;

  @Field(() => [ToolCountLastOutput], {
    description: '공구 정보',
    nullable: true,
  })
  toolCount?: ToolCountLastOutput[];

  @Field(() => RawOutput, { description: 'CNC 파라미터 정보', nullable: true })
  parameter?: RawOutput;

  // @Field(() => [Abnormal], {
  //   description: '이상 감지 내용',
  //   nullable: true,
  // })
  // abnormal?: Abnormal[];

  // @Field(() => [ToolStatusOutput], {
  //   description: '공구 상태 감지 정보',
  //   nullable: true,
  // })
  // toolStatus?: ToolStatusOutput[];
}

@ObjectType()
export class LineMonitorOutput {
  @Field(() => String, { description: '라인 코드' })
  lineCode: string;

  @Field(() => String, { description: '라인 명' })
  lineName: string;

  @Field(() => [MonitorOutput], {
    description: '공정 데이터 배열',
    nullable: true,
  })
  operationMonitors?: MonitorOutput[];
}

@ObjectType()
export class WorkshopMonitorOutput {
  @Field(() => String, { description: '공장 코드' })
  workshopCode: string;

  @Field(() => String, { description: '공장 명' })
  workshopName: string;

  @Field(() => [LineMonitorOutput], {
    description: '라인 데이터 배열',
    nullable: true,
  })
  lineMonitors?: LineMonitorOutput[];
}
