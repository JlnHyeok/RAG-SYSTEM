import { Resolver, Query, Subscription, Args, Info } from '@nestjs/graphql';
import { ToolHistoryService } from './tool-history.service';
import { ToolHistory } from './entities/tool-history.entity';
import { TOPIC_TOOL_COUNT } from 'src/pubsub/pubsub.constants';
import {
  ToolCountLastOutput,
  ToolHistoryInfluxOutput,
} from './dto/tool-history.output';
import { FilterCommonInput } from 'src/common/dto/filter-common.input';
import { FilterToolHistoryInfluxInput } from './dto/filter-tool-history.input';
import { UseGuards } from '@nestjs/common';
import { AuthGuard } from 'src/auth/auth.guard';

@Resolver(() => ToolHistory)
export class ToolHistoryResolver {
  constructor(private readonly toolHistoryService: ToolHistoryService) {}

  @UseGuards(...[AuthGuard])
  @Query(() => [ToolHistory], { name: 'toolHistory' })
  find() {
    return this.toolHistoryService.find();
  }

  @UseGuards(...[AuthGuard])
  @Query(() => [ToolHistoryInfluxOutput], { name: 'toolHistoryReports' })
  async findInflux(
    @Args('filterProductInfluxInput')
    filterToolHistoryInfluxInput: FilterToolHistoryInfluxInput,
    @Info() info,
  ) {
    const selectedFields: Array<object> =
      info.fieldNodes[0].selectionSet.selections;
    const selectedNames = selectedFields.map((p) => p['name'].value);

    return await this.toolHistoryService.findInflux(
      filterToolHistoryInfluxInput,
      selectedNames,
    );
  }

  // * Subscription
  // 1. 공구 사용량 Subscription
  // @Subscription(() => [ToolCountLastOutput], {
  //   name: TOPIC_TOOL_COUNT,
  // })
  // monitor() {
  //   return this.toolHistoryService.monitor();
  // }

  @Subscription(() => [ToolCountLastOutput], {
    name: TOPIC_TOOL_COUNT,
  })
  async monitorTest(
    @Args('filterCommonInput')
    filterCommonInput: FilterCommonInput,
  ) {
    return await this.toolHistoryService.monitor(filterCommonInput);
  }
}
