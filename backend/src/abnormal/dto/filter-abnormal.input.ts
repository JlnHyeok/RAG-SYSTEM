import { Field, InputType, Int, PartialType } from '@nestjs/graphql';
import {
  FilterCommonInput,
  SortCommonInput,
} from 'src/common/dto/filter-common.input';

@InputType()
export class FilterAbnormalInput {
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
    description: '이상감지 구분',
    nullable: true,
  })
  abnormalCode?: string;

  @Field(() => String, {
    description: '필터 문구',
    nullable: true,
  })
  filterKeyword?: string;

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
export class FilterAbnormalDetailInput extends PartialType(FilterCommonInput) {
  @Field(() => String, { description: '제품 정보' })
  productNo: string;
}

@InputType()
export class FilterAbnormalReportInput {
  @Field(() => FilterCommonInput, { description: '공통 필터' })
  commonFilter: FilterCommonInput;

  @Field(() => Date, {
    description: '조회 시작 일시',
  })
  rangeStart: Date;

  @Field(() => Date, {
    description: '조회 종료 일시',
  })
  rangeEnd: Date;
}
