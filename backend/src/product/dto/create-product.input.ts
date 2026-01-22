import { InputType, Int, Field, Float } from '@nestjs/graphql';
import { PartialType } from '@nestjs/mapped-types';
import { IsNumber, IsString } from 'class-validator';
import { CreateCommonInput } from 'src/common/dto/create-common.input';

@InputType()
export class CreateProductInput extends PartialType(CreateCommonInput) {
  @Field(() => String, { description: '생산 번호' })
  @IsString()
  productId: string;

  @Field(() => Int, { description: '시작 일시' })
  @IsNumber()
  startTime: number;

  @Field(() => Int, { description: '종료 일시' })
  @IsNumber()
  endTime: number;

  @Field(() => Int, { description: '생산 수량' })
  @IsNumber()
  count: number;

  @Field(() => Float, { description: 'Cycle Time' })
  @IsNumber()
  ct: number;

  // @Field(() => String, { description: '생산 번호' })
  // @IsString()
  // ctResult: string;

  // @Field(() => Float, { description: 'AI 분석 결과' })
  // @IsNumber()
  // @IsOptional()
  // ai: number;

  // @Field(() => String, { description: '생산 번호' })
  // @IsString()
  // @IsOptional()
  // aiResult: string;

  @Field(() => Float, { description: '부하 Sum' })
  @IsNumber()
  loadSum: number;

  // @Field(() => String, { description: '생산 번호' })
  // @IsString()
  // loadSumResult: string;

  @Field(() => String, { description: '메인 프로그램 번호' })
  @IsString()
  mainProg: string;

  @Field(() => Int, { description: '생산 완료 상태' })
  @IsNumber()
  completeStatus: number;

  // @Field(() => String, { description: '서브 프로그램 번호' })
  // @IsString()
  // subProg: string;

  @Field(() => Float, { description: 'FOV (%)' })
  @IsNumber()
  fov: number;

  @Field(() => Float, { description: 'SOV (%)' })
  @IsNumber()
  sov: number;

  @Field(() => Float, { description: 'X Offset 값' })
  @IsNumber()
  offsetX: number;

  @Field(() => Float, { description: 'Z Offset 값' })
  @IsNumber()
  offsetZ: number;
}
