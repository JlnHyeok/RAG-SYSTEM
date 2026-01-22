import { InputType, PartialType, Field, PickType } from '@nestjs/graphql';
import { FilterCommonInput } from 'src/common/dto/filter-common.input';
import { FilterInfluxInput } from 'src/influx/dto/filter-influx.input';

@InputType()
export class FilterRawInput extends PartialType(FilterInfluxInput) {
  @Field(() => FilterCommonInput, { description: '공통 필터' })
  commonFilter: FilterCommonInput;
}

@InputType()
export class FilterRawTCodeInput extends PickType(FilterInfluxInput, [
  'aggregateInterval',
  'rangeStartString',
]) {
  @Field(() => FilterCommonInput, { description: '공통 필터' })
  commonFilter: FilterCommonInput;

  @Field(() => String, {
    description: '공구 번호',
  })
  TCode: string;

  @Field(() => [String], {
    description: '제품 코드',
    nullable: true,
  })
  productNo?: string[];
}

@InputType()
export class FilterOperateReportInput extends PartialType(FilterInfluxInput) {
  @Field(() => FilterCommonInput, { description: '공통 필터' })
  commonFilter: FilterCommonInput;

  @Field(() => String, {
    description: '집계 필드',
  })
  reportField: string;

  @Field(() => String, {
    description: '설비 상태',
  })
  status: string;
}
