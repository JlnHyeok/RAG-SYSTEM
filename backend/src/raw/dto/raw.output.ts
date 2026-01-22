import { ObjectType, Field, Float, Int } from '@nestjs/graphql';

@ObjectType()
export class RawOutput {
  // 2024.11.07 TSDB 조회 성능 이슈로 인한 스키마 변경
  @Field(() => Int, { name: 'Index', description: 'Index' })
  Idx: number = 0;

  // Timestamp
  @Field(() => Date, { name: 'time', description: '수집 일시' })
  time: Date = new Date();

  @Field(() => String, { description: '공장 코드' })
  WorkshopCode: string = '';
  @Field(() => String, { description: '라인 코드' })
  LineCode: string = '';
  @Field(() => String, { description: '공정 코드' })
  OpCode: string = '';
  @Field(() => String, { description: '설비 코드' })
  MachineCode: string = '';
  @Field(() => String, { description: '제품 코드' })
  ProductId: string = '';

  @Field(() => String, { description: '설비 상태' })
  Run: string = '';
  @Field(() => String, {
    description: '메인 프로그램 번호',
  })
  MainProgram: string = '';
  @Field(() => String, {
    description: '서브 프로그램 번호',
  })
  SubProgram: string = '';
  @Field(() => String, { description: 'T Code' })
  TCode: string = '';
  @Field(() => String, { description: 'M Code' })
  MCode: string = '';

  // Fields
  @Field(() => Float, { description: 'Feed' })
  Feed: number = 0;
  @Field(() => Float, { description: 'FOV(%)' })
  Fov: number = 0;
  @Field(() => Float, { description: 'SOV(%)' })
  Sov: number = 0;
  @Field(() => Float, { description: 'Tool Offset X Axis' })
  SV_X_Offset: number = 0;
  @Field(() => Float, { description: 'Tool Offset Z Axis' })
  SV_Z_Offset: number = 0;
  @Field(() => Float, { description: 'X 상대 좌표' })
  SV_X_Pos: number = 0;
  @Field(() => Float, { description: 'Z 상대 좌표' })
  SV_Z_Pos: number = 0;
  @Field(() => Float, { description: '공구 사용 수량 1' })
  TCount1: number = 0;
  @Field(() => Float, { description: '공구 사용 수량 1' })
  TCount2: number = 0;
  @Field(() => Float, { description: '공구 사용 수량 1' })
  TCount3: number = 0;
  @Field(() => Float, { description: '공구 사용 수량 1' })
  TCount4: number = 0;

  // AI Engine 데이터
  @Field(() => Float, { description: 'Spindle 부하 (수집)' })
  Load: number = 0;
  @Field(() => Float, {
    name: 'Predict',
    description: 'Spindle 부하 (예측)',
    nullable: true,
  })
  Predict?: number = 0;
  @Field(() => Float, { name: 'Loss', description: '오차율', nullable: true })
  Loss?: number = 0;

  // @Field(() => Int, { name: 'Index', description: 'Index' })
  // Idx: number = 0;

  // // Timestamp
  // @Field(() => Date, { name: 'time', description: '수집 일시' })
  // time: Date = new Date();

  // // Tags
  // @Field(() => String, { description: '공장 코드' })
  // WorkshopCode: string = '';
  // @Field(() => String, { description: '라인 코드' })
  // LineCode: string = '';
  // @Field(() => String, { description: '공정 코드' })
  // OpCode: string = '';
  // @Field(() => String, { description: '설비 코드' })
  // MachineCode: string = '';
  // // @Field(() => String, { description: '설비 모드' })
  // // Aut: string = '';
  // @Field(() => String, { description: '설비 상태' })
  // Run: string = '';
  // @Field(() => String, {
  //   description: '메인 프로그램 번호',
  // })
  // MainProgram: string = '';
  // @Field(() => String, {
  //   description: '서브 프로그램 번호',
  // })
  // SubProgram: string = '';
  // @Field(() => String, { description: 'T Code' })
  // TCode: string = '';
  // @Field(() => String, { description: 'M Code' })
  // MCode: string = '';

  // // Fields
  // @Field(() => Float, { description: 'FOV(%)' })
  // Fov: number = 0;
  // @Field(() => Float, { description: 'SOV(%)' })
  // Sov: number = 0;
  // @Field(() => Float, { description: 'Tool Offset X Axis' })
  // SV_X_Offset: number = 0;
  // @Field(() => Float, { description: 'Tool Offset Z Axis' })
  // SV_Z_Offset: number = 0;
  // @Field(() => Float, { description: 'X 상대 좌표' })
  // SV_X_Pos: number = 0;
  // @Field(() => Float, { description: 'Z 상대 좌표' })
  // SV_Z_Pos: number = 0;

  // @Field(() => Float, { description: 'Feedrate' })
  // Feed: number = 0;
  // @Field(() => Float, { description: 'Spindle 부하 (수집)' })
  // Load: number = 0;

  // // AI Engine 데이터
  // @Field(() => Float, {
  //   name: 'Predict',
  //   description: 'Spindle 부하 (예측)',
  //   nullable: true,
  // })
  // Predict?: number = 0;
  // @Field(() => Float, { name: 'Loss', description: '오차율', nullable: true })
  // Loss?: number = 0;
  // @Field(() => Float, {
  //   name: 'PredictFlag',
  //   description: '오차율',
  //   nullable: true,
  // })
  // PredictFlag?: number = 0;
}

@ObjectType()
export class RawTCodeOutput {
  @Field(() => String, { description: 'T Code' })
  TCode: string = '';

  @Field(() => [RawTCodeSingleOutput], { description: 'T Code 사용 범위' })
  TCodeRange: RawTCodeSingleOutput[];
}
@ObjectType()
export class RawTCodeSingleOutput {
  @Field(() => String, { description: '제품 번호', nullable: true })
  productNo?: string = '';

  @Field(() => Int, { description: '공구 사용 시작 IDX', nullable: true })
  beginIdx?: number = 0;

  @Field(() => Int, { description: '공구 사용 종료 IDX', nullable: true })
  endIdx?: number = 0;

  @Field(() => Date, {
    name: 'beginTime',
    description: '공구 사용 시작 일시',
    nullable: true,
  })
  beginTime?: Date;

  @Field(() => Date, {
    name: 'endTime',
    description: '공구 사용 종료 일시',
    nullable: true,
  })
  endTime?: Date;
}

@ObjectType()
export class RawOperationReportOutput {
  // *** 기능 구현을 위해 모니터링 데이터로 대체 ***
  // Timestamp
  @Field(() => Date, { name: 'reportDate', description: '집계 일시' })
  reportDate: Date;

  // Tags
  @Field(() => String, { description: '공장 코드' })
  WorkshopCode: string = '';
  @Field(() => String, { description: '라인 코드' })
  LineCode: string = '';
  @Field(() => String, { description: '공정 코드' })
  OpCode: string = '';
  @Field(() => String, { description: '설비 코드' })
  MachineCode: string = '';

  // @Field(() => String, { description: '설비 모드' })
  // Aut: string = '';
  // @Field(() => String, { description: '설비 상태' })
  // Run: string = '';
  // @Field(() => String, {
  //   description: '메인 프로그램 번호',
  // })
  // MainProgram: string = '';
  // @Field(() => String, {
  //   description: '서브 프로그램 번호',
  // })
  // SubProgram: string = '';
  // @Field(() => String, { description: 'T Code' })
  // TCode: string = '';
  // @Field(() => String, { description: 'M Code' })
  // MCode: string = '';

  // Fields (Aggregate를 위해 Float 타입으로 지정)
  @Field(() => Float, { name: 'operationTime', description: '가동 시간' })
  operationTime: number = 0;
}
@ObjectType()
export class RawOperationPeriodReportOutput {
  // Timestamp
  @Field(() => Date, { name: 'time', description: '수집 일시' })
  time: Date = new Date();

  // Tags
  @Field(() => String, { description: '공장 코드' })
  WorkshopCode: string = '';
  @Field(() => String, { description: '라인 코드' })
  LineCode: string = '';
  @Field(() => String, { description: '공정 코드' })
  OpCode: string = '';
  @Field(() => String, { description: '설비 코드' })
  MachineCode: string = '';
  // @Field(() => String, { description: '제품 코드' })
  // ProductId: string = '';
  // @Field(() => Date, { description: '시작 일시' })
  // startTime: Date;
  // @Field(() => Date, { description: '종료 일시' })
  // endTime: Date;

  // Fields
  @Field(() => Int, { description: '가동 시간(ms)' })
  Run: number;
}
