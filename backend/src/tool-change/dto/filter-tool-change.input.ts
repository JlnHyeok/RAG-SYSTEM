import { InputType, PartialType, Field, Int } from '@nestjs/graphql';
import { FilterCommonInput } from 'src/common/dto/filter-common.input';

@InputType()
export class FilterToolChangeReportInput extends PartialType(
  FilterCommonInput,
) {
  @Field(() => Int, { description: '조회 시작 년도' })
  beginYear: number;

  @Field(() => Int, { description: '조회 시작 월' })
  beginMonth: number;

  @Field(() => Int, { description: '조회 종료 년도' })
  endYear: number;

  @Field(() => Int, { description: '조회 종료 월' })
  endMonth: number;
}

@InputType()
export class FilterToolChangeAvgReportInput extends PartialType(
  FilterCommonInput,
) {
  @Field(() => Date, {
    description: '조회 시작 일시',
    nullable: true,
  })
  rangeStart?: Date;

  @Field(() => Date, {
    description: '조회 종료 일시',
    nullable: true,
  })
  rangeEnd?: Date;
}

@InputType()
export class FilterToolInfoInput extends PartialType(FilterCommonInput) {
  @Field(() => String, { description: '공구 번호', nullable: true })
  toolCode?: string;
}
