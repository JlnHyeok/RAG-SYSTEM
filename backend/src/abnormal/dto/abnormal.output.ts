import { Field, Float, Int, ObjectType, PartialType } from '@nestjs/graphql';
import { Abnormal, AbnormalSummary } from '../entities/abnormal.entity';
import { RawOutput } from 'src/raw/dto/raw.output';
import { CommonMutationOutput } from 'src/common/dto/common.output';

@ObjectType()
export class AbnormalPaginationOutput {
  @Field(() => Int, {
    name: 'pageCount',
    description: '총 페이지 수',
    nullable: true,
  })
  pageCount: number;

  @Field(() => [Abnormal], { name: 'abnormals', description: '이상감지 이력' })
  abnormals: Abnormal[];
}
@ObjectType()
export class AbnormalSummaryPaginationOutput {
  @Field(() => Int, {
    name: 'pageCount',
    description: '총 페이지 수',
    nullable: true,
  })
  pageCount: number;

  @Field(() => [AbnormalSummary], {
    name: 'abnormals',
    description: '이상감지 이력',
  })
  abnormals: AbnormalSummary[];
}
@ObjectType()
export class AbnormalDetailOutput {
  @Field(() => String, { description: 'CT 이상 발생 여부' })
  abnormalCt: string;
  @Field(() => Float, { description: 'CT 이상 값' })
  abnormalCtValue: number;
  @Field(() => Float, { description: 'CT 이상 임계치 (상한)' })
  abnormalCtThreshold: number;
  @Field(() => Float, { description: 'CT 이상 임계치 (하한)', nullable: true })
  abnormalMinCtThreshold?: number;

  @Field(() => String, { description: 'LoadSum 이상 발생 여부' })
  abnormalLoad: string;
  @Field(() => Float, { description: 'LoadSum 값' })
  abnormalLoadValue: number;
  @Field(() => Float, { description: 'LoadSum 임계치 (상한)' })
  abnormalLoadThreshold: number;
  @Field(() => Float, { description: 'LoadSum 임계치 (하한)', nullable: true })
  abnormalMinLoadThreshold?: number;

  @Field(() => [AbnormalAiOutput], {
    name: 'abnormalAi',
    description: '부하 이상감지 이력',
    nullable: true,
  })
  abnormalAi?: AbnormalAiOutput[];

  @Field(() => [RawOutput], {
    name: 'raws',
    description: '부하 데이터',
    nullable: true,
  })
  raws?: RawOutput[];
}
@ObjectType()
export class AbnormalAiOutput {
  @Field(() => Date, { description: '시작 일시', nullable: true })
  abnormalAiBeginDate: Date;

  @Field(() => Date, { description: '종료 일시', nullable: true })
  abnormalAiEndDate: Date;

  @Field(() => Float, { description: 'Loss', nullable: true })
  abnormalAiValue: number;
}

@ObjectType()
export class AbnormalSummaryMutationOutput extends PartialType(
  CommonMutationOutput,
) {
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

  @Field(() => String, { description: 'CT 이상 발생 여부', nullable: true })
  abnormalCt?: string;
  @Field(() => Float, { description: 'CT 이상 값', nullable: true })
  abnormalCtValue?: number;
  @Field(() => Float, { description: 'CT 이상 임계치', nullable: true })
  abnormalCtThreshold?: number;
  @Field(() => Float, { description: 'CT 이상 임계치', nullable: true })
  abnormalMinCtThreshold?: number;

  @Field(() => String, {
    description: 'LoadSum 이상 발생 여부',
    nullable: true,
  })
  abnormalLoad?: string;
  @Field(() => Float, { description: 'LoadSum 값', nullable: true })
  abnormalLoadValue?: number;
  @Field(() => Float, { description: 'LoadSum 임계치', nullable: true })
  abnormalLoadThreshold?: number;
  @Field(() => Float, { description: 'LoadSum 임계치', nullable: true })
  abnormalMinLoadThreshold?: number;

  @Field(() => String, { description: '부하 이상 발생 여부', nullable: true })
  abnormalAi?: string;
  @Field(() => Int, { description: '부하 이상 값', nullable: true })
  abnormalAiValue?: number;
  @Field(() => Float, { description: '부하 이상 임계치', nullable: true })
  abnormalAiThreshold?: number;
  @Field(() => Int, { description: '임계치 데이터 수', nullable: true })
  abnormalAiCount?: number;

  @Field(() => Date, { description: '시작 일시', nullable: true })
  abnormalBeginDate?: Date;
  @Field(() => Date, { description: '종료 일시', nullable: true })
  abnormalEndDate?: Date;

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

@ObjectType()
export class AbnormalReportOutput {
  @Field(() => String, { description: '이상 감지 구분' })
  abnormalCode: string;

  @Field(() => String, { description: '이상 감지 공구', nullable: true })
  abnormalTool?: string;

  @Field(() => Int, { description: '이상 발생 횟수' })
  abnormalCount: number;
}

@ObjectType()
export class AbnormalSubscriptionOutput {
  @Field(() => String, { description: '이상 감지 구분', nullable: true })
  abnormalCode: string;

  @Field(() => Date, { description: '시작 일시', nullable: true })
  abnormalBeginDate: Date;

  @Field(() => Date, { description: '종료 일시', nullable: true })
  abnormalEndDate: Date;

  @Field(() => String, { description: '이상 감지 공구', nullable: true })
  abnormalTool?: string;

  @Field(() => Float, { description: '이상 감지 값', nullable: true })
  abnormalValue: number;
}
