import { InputType, Int, Field, Float, PartialType } from '@nestjs/graphql';
import { IsEnum, IsNumber, IsString } from 'class-validator';
import { AbnormalMinMax } from 'src/common/dto/common.enum';
import { CreateCommonInput } from 'src/common/dto/create-common.input';

@InputType()
export class CreateAbnormalInput extends PartialType(CreateCommonInput) {
  @Field(() => String, { description: '생산 번호' })
  productId: string;

  @Field(() => String, { description: '이상 감지 구분' })
  abnormalCode: string;

  @IsEnum(AbnormalMinMax)
  @Field(() => AbnormalMinMax, { description: '상/하한 구분', nullable: true })
  abnormalDivision?: AbnormalMinMax;

  @Field(() => Date, { description: '시작 일시' })
  abnormalBeginDate: Date;

  @Field(() => Date, { description: '종료 일시' })
  abnormalEndDate: Date;

  @Field(() => String, { description: '이상 감지 공구', nullable: true })
  abnormalTool?: string;

  @Field(() => Float, { description: '이상 감지 값', nullable: true })
  abnormalValue: number;

  // CNC 정적 파라미터 추가
  @Field(() => String, {
    name: 'mainProgramNo',
    description: '메인 프로그램 번호',
  })
  mainProgramNo: string;

  @Field(() => String, {
    name: 'subProgramNo',
    description: '서브 프로그램 번호',
  })
  subProgramNo: string;

  @Field(() => String, { name: 'tCode', description: 'T Code' })
  tCode: string;

  @Field(() => String, { name: 'mCode', description: 'M Code' })
  mCode: string;

  @Field(() => Float, { name: 'fov', description: 'FOV(%)' })
  fov: number;

  @Field(() => Float, { name: 'sov', description: 'SOV(%)' })
  sov: number;

  @Field(() => Float, { name: 'offsetX', description: 'Tool Offset X Axis' })
  offsetX: number;

  @Field(() => Float, { name: 'offsetZ', description: 'Tool Offset Z Axis' })
  offsetZ: number;

  @Field(() => Float, { name: 'feed', description: 'Feedrate' })
  feed: number;

  @Field(() => Date, { description: '생성 일시' })
  createAt?: Date;

  @Field(() => Date, { description: '수정 일시', nullable: true })
  updateAt?: Date;
}

@InputType()
export class CreateLossAbnormalInput extends PartialType(CreateCommonInput) {
  @Field(() => String, { description: '생산 번호' })
  @IsString()
  productId: string;

  @Field(() => Int, { description: '시작 일시' })
  @IsNumber()
  startTime: number;

  @Field(() => Int, { description: '종료 일시' })
  @IsNumber()
  endTime: number;

  @Field(() => Float, { description: '오차율 (%)' })
  @IsNumber()
  loss: number;

  // CNC 정적 파라미터 추가
  @Field(() => String, {
    name: 'tCode',
    description: '공구 번호',
  })
  @IsString()
  tCode: string;

  @Field(() => Float, { description: '임계치' })
  @IsNumber()
  threshold: number;

  @Field(() => Int, { description: '임계치 데이터 수' })
  @IsNumber()
  lossCount: number;

  @Field(() => String, {
    name: 'mainProgramNo',
    description: '메인 프로그램 번호',
  })
  @IsString()
  mainProg: string;

  // @Field(() => String, {
  //   name: 'subProgramNo',
  //   description: '서브 프로그램 번호',
  // })
  // @IsString()
  // subProg: string;

  @Field(() => Float, { name: 'fov', description: 'FOV(%)' })
  @IsNumber()
  fov: number;

  @Field(() => Float, { name: 'sov', description: 'SOV(%)' })
  @IsNumber()
  sov: number;

  @Field(() => Float, { name: 'offsetX', description: 'Tool Offset X Axis' })
  @IsNumber()
  offsetX: number;

  @Field(() => Float, { name: 'offsetZ', description: 'Tool Offset Z Axis' })
  @IsNumber()
  offsetZ: number;
}

@InputType()
export class CreateAbnormalSummaryInput extends PartialType(CreateCommonInput) {
  @Field(() => String, { description: '생산 번호' })
  productId: string;

  @Field(() => Date, { description: '시작 일시' })
  abnormalBeginDate: Date;

  @Field(() => Date, { description: '종료 일시' })
  abnormalEndDate: Date;

  @Field(() => String, { description: 'CT 이상 발생 여부' })
  abnormalCt: string;
  @Field(() => Float, { description: 'CT 이상 값' })
  abnormalCtValue: number;
  @Field(() => Float, { description: 'CT 이상 임계치 (상한)' })
  abnormalCtThreshold: number;
  @Field(() => Float, { description: 'CT 이상 임계치 (하한)' })
  abnormalMinCtThreshold: number;

  @Field(() => String, { description: 'LoadSum 이상 발생 여부' })
  abnormalLoad: string;
  @Field(() => Float, { description: 'LoadSum 값' })
  abnormalLoadValue: number;
  @Field(() => Float, { description: 'LoadSum 임계치 (상한)' })
  abnormalLoadThreshold: number;
  @Field(() => Float, { description: 'LoadSum 임계치 (하한)' })
  abnormalMinLoadThreshold: number;

  @Field(() => String, { description: '부하 이상 발생 여부' })
  abnormalAi: string;
  @Field(() => String, { description: '부하 이상 값' })
  abnormalAiValue: number;
  @Field(() => Float, { description: '부하 이상 임계치' })
  abnormalAiThreshold: number;
  @Field(() => Int, { description: '임계치 데이터 수', nullable: true })
  abnormalAiCount?: number;

  // CNC 정적 파라미터 추가
  @Field(() => String, {
    name: 'mainProgramNo',
    description: '메인 프로그램 번호',
  })
  mainProgramNo: string;

  @Field(() => String, {
    name: 'subProgramNo',
    description: '서브 프로그램 번호',
  })
  subProgramNo: string;

  @Field(() => String, { name: 'tCode', description: 'T Code' })
  tCode: string;

  @Field(() => String, { name: 'mCode', description: 'M Code' })
  mCode: string;

  @Field(() => Float, { name: 'fov', description: 'FOV(%)' })
  fov: number;

  @Field(() => Float, { name: 'sov', description: 'SOV(%)' })
  sov: number;

  @Field(() => Float, { name: 'offsetX', description: 'Tool Offset X Axis' })
  offsetX: number;

  @Field(() => Float, { name: 'offsetZ', description: 'Tool Offset Z Axis' })
  offsetZ: number;

  @Field(() => Float, { name: 'feed', description: 'Feedrate' })
  feed: number;

  @Field(() => Date, { description: '생성 일시' })
  createAt?: Date;

  @Field(() => Date, { description: '수정 일시', nullable: true })
  updateAt?: Date;
}
