import { InputType, Int, Field, Float, PartialType } from '@nestjs/graphql';
import { IsString, IsNumber, IsOptional } from 'class-validator';
import { CreateCommonInput } from 'src/common/dto/create-common.input';

@InputType()
export class CreateToolHistoryInput extends PartialType(CreateCommonInput) {
  @Field(() => String, { description: '전송 타입' })
  @IsString()
  type: string;

  @Field(() => String, { description: '생산 번호' })
  @IsString()
  productId: string;

  @Field(() => String, { description: '공구 번호' })
  @IsString()
  code: string;

  @Field(() => Int, { description: '시작 일시' })
  @IsNumber()
  startTime: number;

  @Field(() => Int, { description: '종료 일시' })
  @IsNumber()
  @IsOptional()
  endTime?: number;

  @Field(() => Float, { description: '공구 CT' })
  @IsNumber()
  @IsOptional()
  ct?: number;

  @Field(() => Float, { description: '공구 LoadSum' })
  @IsNumber()
  @IsOptional()
  loadSum?: number;

  // @Field(() => [Float], { description: '공구 분석 결과', nullable: true })
  // @IsArray()
  // @IsOptional()
  // status?: number[];

  @Field(() => String, { description: '메인 프로그램 번호' })
  @IsString()
  @IsOptional()
  mainProg: string;

  // @Field(() => String, { description: '서브 프로그램 번호', nullable: true })
  // @IsString()
  // @IsOptional()
  // subProg?: string;

  @Field(() => Float, { description: 'FOV (%)' })
  @IsNumber()
  @IsOptional()
  fov: number;

  @Field(() => Float, { description: 'SOV (%)' })
  @IsNumber()
  @IsOptional()
  sov: number;

  @Field(() => Float, { description: 'X Offset 값' })
  @IsNumber()
  @IsOptional()
  offsetX: number;

  @Field(() => Float, { description: 'Z Offset 값' })
  @IsNumber()
  @IsOptional()
  offsetZ: number;
}
