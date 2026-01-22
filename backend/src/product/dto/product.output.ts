import { Field, Float, Int, ObjectType, PartialType } from '@nestjs/graphql';
import { Abnormal } from 'src/abnormal/entities/abnormal.entity';
import { Product } from '../entities/product.entity';
import { CommonMutationOutput } from 'src/common/dto/common.output';

@ObjectType()
export class ProductPaginationOutput {
  @Field(() => Int, {
    name: 'pageCount',
    description: '총 페이지 수',
    nullable: true,
  })
  pageCount: number;

  @Field(() => [Product], { name: 'products', description: '생산 이력' })
  products: Product[];
}

@ObjectType()
export class ProductLastOutput {
  @Field(() => String, { description: '생산 번호' })
  productNo: string;

  @Field(() => Date, { description: '생산 시작 일시' })
  productBeginDate: Date;

  @Field(() => Date, { description: '생산 종료 일시' })
  productEndDate: Date;

  // @Field(() => String, { description: '생산 결과' })
  // productResult: string;

  @Field(() => Float, { description: 'Cycle Time' })
  productCt: number;

  @Field(() => Float, { description: 'AI 분석 결과' })
  productAi: number;

  @Field(() => Float, { description: '부하 Sum' })
  productLoadSum: number;

  @Field(() => Float, { description: '평균 Cycle Time' })
  productAvgCt: number;

  @Field(() => Float, { description: '평균 Cycle Time', nullable: true })
  productAvgLoadSum?: number;

  @Field(() => Float, { description: '평균 Cycle Time' })
  productDailyCount: number;

  @Field(() => Int, { description: '생산 완료 상태', nullable: true })
  productCompleteStatus?: number;
}

@ObjectType()
export class ProductAbnormalOutput {
  @Field(() => String, { description: '생산 번호' })
  productNo: string;

  @Field(() => Date, { description: '생산 시작 일시' })
  productBeginDate: Date;

  @Field(() => Date, { description: '생산 종료 일시' })
  productEndDate: Date;

  @Field(() => String, { description: '생산 결과' })
  productResult: string;

  @Field(() => Float, { description: 'Cycle Time' })
  productCt: number;

  @Field(() => String, { description: '생산 결과' })
  productCtResult: string;

  @Field(() => Float, { description: 'AI 분석 결과' })
  productAi: number;

  @Field(() => String, { description: '생산 결과' })
  productAiResult: string;

  @Field(() => Float, { description: '부하 Sum' })
  productLoadSum: number;

  @Field(() => String, { description: '부하 Sum' })
  productLoadSumResult: string;

  @Field(() => Int, { description: '생산 완료 상태', nullable: true })
  productCompleteStatus?: number;

  // TODO: Abnormal 모듈 수정 후 구현
  @Field(() => [Abnormal], { description: '이상감지 이력' })
  abnormals?: Abnormal[];
}

@ObjectType()
export class ProductCountOutput {
  @Field(() => Int, { description: '생산 수량' })
  productCount: number;
}

@ObjectType()
export class ProductInfluxOutput {
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
  // @Field(() => String, { description: '제품 코드' })
  // ProductId: string = '';
  // @Field(() => Date, { description: '시작 일시' })
  // startTime: Date;
  // @Field(() => Date, { description: '종료 일시' })
  // endTime: Date;

  // Fields
  @Field(() => Int, { description: '생산 수량' })
  Count: number;
  @Field(() => Float, { description: 'Cycle Time' })
  CT: number;
  @Field(() => Float, { description: '부하 Sum' })
  LoadSum: number;
}

@ObjectType()
export class ProductSumReportOutput {
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
  @Field(() => Int, { description: '생산 수량' })
  Count: number;
}

@ObjectType()
export class ProductSubscriptionOutput {
  @Field(() => String, { description: '생산 번호' })
  productNo: string;

  @Field(() => Date, { description: '생산 시작 일시' })
  productBeginDate: Date;

  @Field(() => Date, { description: '생산 종료 일시' })
  productEndDate: Date;

  @Field(() => String, { description: '생산 결과' })
  productResult: string;

  @Field(() => Int, { description: '당일 생산 수량' })
  productCount: number;

  @Field(() => Float, { description: 'Cycle Time' })
  productCt: number;

  @Field(() => String, { description: 'Cycle Time 이상 유무' })
  productCtResult: string;

  @Field(() => Float, { description: 'AI 분석 결과', nullable: true })
  productAi?: number;

  @Field(() => String, { description: 'Cycle Time 이상 유무', nullable: true })
  productAiResult?: string;

  @Field(() => Float, { description: '부하 Sum' })
  productLoadSum: number;

  @Field(() => String, { description: '부하 Sum 이상 유무' })
  productLoadSumResult: string;

  @Field(() => Float, { description: '평균 Cycle Time' })
  productAvgCt: number;

  @Field(() => Float, { description: '평균 Cycle Time', nullable: true })
  productAvgLoadSum?: number;

  @Field(() => Int, { description: '생산 완료 상태', nullable: true })
  productCompleteStatus?: number;
}

@ObjectType()
export class ProductMutationOutput extends PartialType(CommonMutationOutput) {
  @Field(() => String, { description: '공장 코드', nullable: true })
  workshopCode?: string;

  @Field(() => String, { description: '라인 코드', nullable: true })
  lineCode?: string;

  @Field(() => String, { description: '공정 코드', nullable: true })
  opCode?: string;

  @Field(() => String, { description: '설비 코드', nullable: true })
  machineCode?: string;

  @Field(() => String, { description: '생산 번호', nullable: true })
  productNo?: string;

  @Field(() => Date, { description: '시작 일시', nullable: true })
  startTime?: Date;

  @Field(() => Date, { description: '종료 일시', nullable: true })
  endTime?: Date;

  @Field(() => String, { description: '생산 결과', nullable: true })
  productResult?: string;

  @Field(() => Int, { description: '생산 완료 상태', nullable: true })
  completeStatus?: number;

  @Field(() => Int, { description: '생산 수량', nullable: true })
  count?: number;

  @Field(() => Float, { description: 'Cycle Time', nullable: true })
  ct?: number;

  @Field(() => String, { description: '생산 결과', nullable: true })
  ctResult?: string;

  @Field(() => Float, { description: 'AI 분석 결과', nullable: true })
  ai?: number;

  @Field(() => String, { description: '생산 결과', nullable: true })
  aiResult?: string;

  @Field(() => Float, { description: '부하 Sum', nullable: true })
  loadSum?: number;

  @Field(() => String, { description: '생산 결과', nullable: true })
  loadSumResult?: string;

  // CNC 정적 파라미터 추가
  @Field(() => String, {
    name: 'mainProgramNo',
    description: '메인 프로그램 번호',
    nullable: true,
  })
  mainProgramNo?: string;

  @Field(() => String, {
    name: 'subProgramNo',
    description: '서브 프로그램 번호',
    nullable: true,
  })
  subProgramNo?: string;

  @Field(() => String, { name: 'tCode', description: 'T Code', nullable: true })
  tCode?: string;

  @Field(() => String, { name: 'mCode', description: 'M Code', nullable: true })
  mCode?: string;

  @Field(() => Float, { name: 'fov', description: 'FOV(%)', nullable: true })
  fov?: number;

  @Field(() => Float, { name: 'sov', description: 'SOV(%)', nullable: true })
  sov?: number;

  @Field(() => Float, {
    name: 'offsetX',
    description: 'Tool Offset X Axis',
    nullable: true,
  })
  offsetX?: number;

  @Field(() => Float, {
    name: 'offsetZ',
    description: 'Tool Offset Z Axis',
    nullable: true,
  })
  offsetZ?: number;

  @Field(() => Float, { name: 'feed', description: 'Feedrate', nullable: true })
  feed?: number;

  @Field(() => Date, { description: '생성 일시', nullable: true })
  createAt?: Date;

  @Field(() => Date, { description: '수정 일시', nullable: true })
  updateAt?: Date;
}
