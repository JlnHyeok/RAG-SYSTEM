import { InputType, Field, Float } from '@nestjs/graphql';
import { IsNumber, IsOptional, IsString } from 'class-validator';

@InputType()
export class CreateCommonInput {
  @Field(() => String, { description: '공장 코드' })
  @IsString()
  workshopId: string;

  @Field(() => String, { description: '라인 코드' })
  @IsString()
  lineId: string;

  @Field(() => String, { description: '공정 코드' })
  @IsString()
  opCode: string;

  @Field(() => String, {
    name: 'machineCode',
    description: '설비 코드',
    nullable: true,
  })
  @IsString()
  @IsOptional()
  machineId: string;
}

@InputType()
export class CreateCommonParameterInput {
  @Field(() => String, {
    name: 'mainProgramNo',
    description: '메인 프로그램 번호',
  })
  @IsString()
  mainProg: string;

  @Field(() => String, {
    name: 'subProgramNo',
    description: '서브 프로그램 번호',
  })
  @IsString()
  subProg: string;

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
