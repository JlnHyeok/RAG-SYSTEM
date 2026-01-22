import { InputType, Field, PartialType } from '@nestjs/graphql';
import { FilterCommonInput } from 'src/common/dto/filter-common.input';
import { FilterInfluxInput } from 'src/influx/dto/filter-influx.input';

@InputType()
export class FilterToolHistoryInfluxInput extends PartialType(
  FilterInfluxInput,
) {
  @Field(() => FilterCommonInput, { description: '공통 필터' })
  commonFilter: FilterCommonInput;
}
