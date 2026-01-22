import { InputType, Field, PartialType, Int, Float } from '@nestjs/graphql';
import { IsNumber, IsString } from 'class-validator';
import { CreateCommonInput } from 'src/common/dto/create-common.input';

@InputType()
export class CreateToolChangeInput {
  @Field(() => String, { description: '공장 코드' })
  workshopCode: string;

  @Field(() => String, { description: '라인 코드' })
  lineCode: string;

  @Field(() => String, { description: '공정 코드' })
  opCode: string;

  @Field(() => String, { description: '설비 코드', nullable: true })
  machineCode?: string;

  @Field(() => String, { description: '공구 번호' })
  toolCode: string;

  @Field(() => String, { description: '공구 교체 사유 코드' })
  reasonCode: string;

  @Field(() => Float, { description: '공구 교체 일시', nullable: true })
  changeDate?: number;

  @Field(() => Float, { description: '사용 수량', nullable: true })
  useCount?: number;
}

@InputType()
export class CreateToolChangeAutoInput extends PartialType(CreateCommonInput) {
  @Field(() => String, { description: '공구 번호' })
  @IsString()
  code: string;

  @Field(() => Int, { description: '교체 일시 (ns)' })
  @IsNumber()
  time: number;

  @Field(() => Int, { description: '직전 사용 수량' })
  @IsNumber()
  useCount: number;
}
