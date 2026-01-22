import { Resolver, Query, Subscription, Args, Info } from '@nestjs/graphql';
import { AbnormalService } from './abnormal.service';
import { Abnormal } from './entities/abnormal.entity';
import {
  AbnormalDetailOutput,
  AbnormalPaginationOutput,
  AbnormalReportOutput,
  AbnormalSubscriptionOutput,
  AbnormalSummaryPaginationOutput,
} from './dto/abnormal.output';
import { TOPIC_MONITOR_ABNORMAL } from 'src/pubsub/pubsub.constants';
import {
  FilterAbnormalDetailInput,
  FilterAbnormalInput,
  FilterAbnormalReportInput,
} from './dto/filter-abnormal.input';
import { FilterCommonInput } from 'src/common/dto/filter-common.input';
import { UseGuards } from '@nestjs/common';
import { AuthGuard } from 'src/auth/auth.guard';

@Resolver(() => Abnormal)
export class AbnormalResolver {
  constructor(private readonly abnormalService: AbnormalService) {}

  // * Query
  // 1. 이상 감지 이력 조회
  @UseGuards(...[AuthGuard])
  @Query(() => AbnormalPaginationOutput, { name: 'abnormals' })
  async find(
    @Args('filterAbnoramlInput', { nullable: true })
    filterAbnoramlInput?: FilterAbnormalInput,
  ) {
    const result = await this.abnormalService.find(filterAbnoramlInput);
    return result;
  }
  // 1. 이상 감지 이력 조회
  @UseGuards(...[AuthGuard])
  @Query(() => AbnormalSummaryPaginationOutput, { name: 'abnormalSummary' })
  async findSummary(
    @Args('filterAbnoramlInput', { nullable: true })
    filterAbnoramlInput?: FilterAbnormalInput,
  ) {
    const result = await this.abnormalService.findSummary(filterAbnoramlInput);
    return result;
  }
  @UseGuards(...[AuthGuard])
  @Query(() => AbnormalDetailOutput, { name: 'abnormalDetail' })
  async findDetail(
    @Args('filterAbnormalDetailInput')
    filterAbnormalDetailInput: FilterAbnormalDetailInput,
  ) {
    const result = await this.abnormalService.findDetail(
      filterAbnormalDetailInput,
    );
    return result;
  }

  // 2. 이상 감지 통계 조회
  @UseGuards(...[AuthGuard])
  @Query(() => [AbnormalReportOutput], { name: 'abnormalReports' })
  async aggregate(
    @Args('filterAbnoramlInput')
    filterAbnormalReportInput: FilterAbnormalReportInput,
  ) {
    return await this.abnormalService.aggregate(filterAbnormalReportInput);
  }

  // * Subscription
  // // 1. 이상 감지 Subscription
  // @Subscription(() => AbnormalSubscriptionOutput, {
  //   name: TOPIC_MONITOR_ABNORMAL,
  // })
  // async monitor() {
  //   return await this.abnormalService.monitor();
  // }
  // 2. 이상 감지 Subscription
  @Subscription(() => AbnormalSubscriptionOutput, {
    name: TOPIC_MONITOR_ABNORMAL,
  })
  async monitor(
    @Args('filterCommonInput')
    filterCommonInput: FilterCommonInput,
  ) {
    return await this.abnormalService.monitor(filterCommonInput);
  }
}
