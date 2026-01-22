import { InputType, Field, Int, PartialType, PickType } from '@nestjs/graphql';
import { IsDateString, IsEnum } from 'class-validator';
import { PeriodType } from 'src/common/dto/common.enum';
import {
  FilterCommonInput,
  SortCommonInput,
} from 'src/common/dto/filter-common.input';
import { FilterInfluxInput } from 'src/influx/dto/filter-influx.input';

// GraphQL - 생산 정보 조회 Param
@InputType()
export class FilterProductInput {
  @Field(() => FilterCommonInput, { description: '공통 필터' })
  commonFilter: FilterCommonInput;

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

  @Field(() => String, {
    description: '생산 번호',
    nullable: true,
  })
  productNo?: string;

  @Field(() => String, {
    description: '필터 문구',
    nullable: true,
  })
  filterKeyword?: string;
  @Field(() => String, {
    description: '필터 구분',
    nullable: true,
  })
  filterResult?: string;

  @Field(() => SortCommonInput, {
    description: '정렬 정보',
    nullable: true,
  })
  sortInput?: SortCommonInput;

  @Field(() => Int, {
    description: '조회 페이지',
    nullable: true,
  })
  page?: number;

  @Field(() => Int, {
    description: '조회 데이터 수',
    nullable: true,
  })
  count?: number;
}

@InputType()
export class FilterProductInfluxInput extends PartialType(FilterInfluxInput) {
  @Field(() => FilterCommonInput, { description: '공통 필터' })
  commonFilter: FilterCommonInput;
}
@InputType()
export class FilterProductSumReportInput extends PickType(FilterInfluxInput, [
  'rangeStart',
  'rangeStop',
]) {
  @Field(() => FilterCommonInput, { description: '공통 필터' })
  commonFilter: FilterCommonInput;

  @IsEnum(PeriodType)
  @Field(() => PeriodType, { description: '기간 구분' })
  periodType: PeriodType;
}

// REST - 당일 생산 수량 조회 Param
@InputType()
export class FilterProductCountInput extends PartialType(FilterCommonInput) {
  @Field(() => Date, {
    description: '조회 일자',
  })
  @IsDateString()
  filterDate: Date;
}
