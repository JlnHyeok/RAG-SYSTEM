import { Resolver, Query, Args, Info } from '@nestjs/graphql';
import { RawService } from './raw.service';
import { Raw } from './entities/raw.entity';
import {
  RawOperationPeriodReportOutput,
  RawOutput,
  RawTCodeOutput,
} from './dto/raw.output';
import { FilterRawInput, FilterRawTCodeInput } from './dto/filter-raw.input';
import { FilterCommonInput } from 'src/common/dto/filter-common.input';
import { FilterProductSumReportInput } from 'src/product/dto/filter-product.input';
import { UseGuards } from '@nestjs/common';
import { AuthGuard } from 'src/auth/auth.guard';

@Resolver(() => Raw)
export class RawResolver {
  constructor(private readonly rawService: RawService) {}

  // * Query
  // 1. Raw 데이터 조회
  @UseGuards(...[AuthGuard])
  @Query(() => [RawOutput], { name: 'raws' })
  async find(
    @Args('filterRawInput')
    filterRawInput: FilterRawInput,
    @Info() info,
  ) {
    const selectedFields: Array<object> =
      info.fieldNodes[0].selectionSet.selections;
    const selectedNames = selectedFields.map((p) => p['name'].value);

    return await this.rawService.find(filterRawInput, selectedNames);
  }

  // 1. Raw 데이터 조회
  @UseGuards(...[AuthGuard])
  @Query(() => RawTCodeOutput, { name: 'rawTCodeRange' })
  async findTCodeRange(
    @Args('filterRawTCodeInput')
    filterRawTCodeInput: FilterRawTCodeInput,
  ) {
    return await this.rawService.findTCodeRange(filterRawTCodeInput);
  }

  // 2. Raw 데이터 검색 (시간 검색)
  @UseGuards(...[AuthGuard])
  @Query(() => RawOutput, { name: 'raw' })
  async findOne(
    @Args('filterDate')
    filterDate: Date,
  ) {
    return await this.rawService.findOne(filterDate);
  }

  // 3. 최근 Raw 데이터 조회
  @UseGuards(...[AuthGuard])
  @Query(() => RawOutput, { name: 'lastRaw' })
  async findLast(
    @Args('filterCommonInput')
    filterCommonInput: FilterCommonInput,
  ) {
    return await this.rawService.findLast(filterCommonInput);
  }

  @UseGuards(...[AuthGuard])
  @Query(() => [RawOperationPeriodReportOutput], {
    name: 'operationSumReports',
  })
  async aggregateSumTest(
    @Args('filterProductSumReportInput')
    filterProductSumReportInput: FilterProductSumReportInput,
    @Info() info,
  ) {
    const selectedFields: Array<object> =
      info.fieldNodes[0].selectionSet.selections;
    const selectedNames = selectedFields.map((p) => p['name'].value);

    return await this.rawService.aggregateOperationSum(
      filterProductSumReportInput,
      selectedNames,
    );
  }
}
