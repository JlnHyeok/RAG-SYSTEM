import { ObjectType, Field, Int, Float } from '@nestjs/graphql';

@ObjectType()
export class ToolCountLastOutput {
  @Field(() => String, { description: '공구 번호' })
  code: string;

  @Field(() => Int, { description: '공구 순서' })
  no: number;

  @Field(() => Int, { description: '공구 사용 수량' })
  useCount: number;

  @Field(() => Int, { description: '공구 기준 수량' })
  maxCount: number;

  @Field(() => String, { description: '공구 상태 분석 결과 (기준 정보)' })
  toolStatusCount: string;

  @Field(() => Boolean, { description: '공구 업데이트 여부' })
  isUpdateTool: boolean;

  @Field(() => Float, { description: '공구 업데이트 시간', nullable: true })
  useStartTime?: number;
}

@ObjectType()
export class ToolHistoryInfluxOutput {
  // Timestamp
  @Field(() => Date, { name: 'time', description: '수집 일시' })
  time: Date = new Date();

  // Tags
  @Field(() => String, { description: '공장 코드', nullable: true })
  WorkshopCode: string = '';
  @Field(() => String, { description: '라인 코드', nullable: true })
  LineCode: string = '';
  @Field(() => String, { description: '공정 코드', nullable: true })
  OpCode: string = '';
  @Field(() => String, { description: '설비 코드', nullable: true })
  MachineCode: string = '';
  @Field(() => String, { description: '공구 번호' })
  TCode: string = '';
  // @Field(() => Date, { description: '시작 일시' })
  // startTime: Date;
  // @Field(() => Date, { description: '종료 일시' })
  // endTime: Date;

  // Fields
  @Field(() => Float, { description: 'Cycle Time' })
  CT: number;
  @Field(() => Float, { description: '부하 Sum' })
  LoadSum: number;
  @Field(() => Float, { description: '평균 부하' })
  Loss: number;
  @Field(() => Float, { description: '데이터 수' })
  Count: number;
}
