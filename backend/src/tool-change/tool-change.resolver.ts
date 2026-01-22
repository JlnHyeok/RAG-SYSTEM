import { Resolver, Mutation, Args, Query } from '@nestjs/graphql';
import { ToolChangeService } from './tool-change.service';
import { ToolChange } from './entities/tool-change.entity';
import { CreateToolChangeInput } from './dto/create-tool-change.input';
import { CreateToolChangeOutput } from './dto/create-tool-change.output';
import { FilterCommonInput } from 'src/common/dto/filter-common.input';
import {
  ToolChangeAverageMonthlyReportOutput,
  ToolChangeAverageReportOutput,
  ToolChangeLastReportOutput,
  ToolChangeSumReportOutput,
} from './dto/tool-change.output';
import {
  FilterToolChangeAvgReportInput,
  FilterToolChangeReportInput,
} from './dto/filter-tool-change.input';
import { forwardRef, Inject, UseGuards } from '@nestjs/common';
import { MachineService } from 'src/master/machine/machine.service';
import { ToolService } from 'src/master/tool/tool.service';
import { TOPIC_TOOL_COUNT } from 'src/pubsub/pubsub.constants';
import { ToolHistoryService } from 'src/tool-history/tool-history.service';
import { PubSub } from 'graphql-subscriptions';
import { PUB_SUB } from 'src/app.provider';
import { AuthGuard } from 'src/auth/auth.guard';

@Resolver(() => ToolChange)
export class ToolChangeResolver {
  constructor(
    private readonly toolChangeService: ToolChangeService,
    @Inject(PUB_SUB)
    private readonly pubSub: PubSub,
    @Inject(forwardRef(() => MachineService))
    private readonly machineService: MachineService,
    @Inject(forwardRef(() => ToolService))
    private readonly toolService: ToolService,
    @Inject(forwardRef(() => ToolHistoryService))
    private readonly toolHistoryService: ToolHistoryService,
  ) {}

  @UseGuards(...[AuthGuard])
  @Mutation(() => CreateToolChangeOutput)
  async createToolChange(
    @Args('createToolChangeInput') createToolChangeInput: CreateToolChangeInput,
  ) {
    // 공정 코드를 이용하여 설비 기준 정보 조회
    const machineInfo = await this.machineService.find({
      opCode: createToolChangeInput.opCode,
    });

    const newToolChange = await this.toolChangeService.create({
      workshopCode: createToolChangeInput.workshopCode,
      lineCode: createToolChangeInput.lineCode,
      opCode: createToolChangeInput.opCode,
      machineCode: machineInfo[0].machineCode,
      toolCode: createToolChangeInput.toolCode,
      reasonCode: createToolChangeInput.reasonCode,
    });

    if (newToolChange.isSuccess) {
      const payloadObj = new Object();

      // 현재 전체 공구 사용 수량 조회
      const currentToolCount =
        await this.toolHistoryService.findCurrentToolCount({
          workshopId: createToolChangeInput.workshopCode,
          lineId: createToolChangeInput.lineCode,
          opCode: createToolChangeInput.opCode,
          machineId: machineInfo[0].machineCode,
          // workshopId: createToolChangeInput.workshopCode,
          // lineId: createToolChangeInput.lineId,
          // opCode: createToolChangeInput.opCode,
          // machineId: machineInfo[0].machineCode,
        });

      // 반복문을 이용하여 사용 수량 및 수명 상태 초기화 (비동기 처리를 위해 for~of 사용)
      for (const t of currentToolCount) {
        // // 현재 공구 기준 정보 조회
        // const tempToolInfo =
        //   machineInfo && machineInfo.length > 0
        //     ? await this.toolService.findOne({
        //         machineCode: machineInfo[0].machineCode,
        //         toolCode: t.code,
        //       })
        //     : null;
        // t.isUpdateTool = t.code == createToolChangeInput.toolCode;

        // // 조회된 공구 기준 정보가 있을 경우 수량 및 수명 상태 초기화
        // if (tempToolInfo) {
        //   t.useCount = t.useCount;
        //   t.toolStatusCount = this.toolHistoryService.parseToolStatus(
        //     t.useCount,
        //     tempToolInfo,
        //   );
        // }

        if (t.isUpdateTool) {
          t.useStartTime =
            newToolChange.changeDate.getUTCMilliseconds() * 1000000;
        }
      }

      payloadObj[TOPIC_TOOL_COUNT] = currentToolCount;
      const topic = `${createToolChangeInput.workshopCode}/${createToolChangeInput.lineCode}/${createToolChangeInput.opCode}/${TOPIC_TOOL_COUNT}`;

      await this.pubSub.publish(topic, payloadObj);
    }

    return newToolChange;
  }

  @UseGuards(...[AuthGuard])
  @Query(() => [ToolChangeLastReportOutput], {
    name: 'toolChangeLastReports',
  })
  async aggregateLast(
    @Args('filterCommonInput')
    filterCommonInput: FilterCommonInput,
  ) {
    return await this.toolChangeService.aggregateLast(filterCommonInput);
  }

  @UseGuards(...[AuthGuard])
  @Query(() => [ToolChangeAverageReportOutput], {
    name: 'toolChangeAverageReports',
  })
  async aggregateAverage(
    @Args('filterToolAvgReportInput')
    filterToolAvgReportInput: FilterToolChangeAvgReportInput,
  ) {
    return await this.toolChangeService.aggregateAverage(
      filterToolAvgReportInput,
    );
  }

  @UseGuards(...[AuthGuard])
  @Query(() => [ToolChangeAverageMonthlyReportOutput], {
    name: 'toolChangeAverageMonthlyReports',
  })
  async aggregateAverageMonthly(
    @Args('filterToolReportInput')
    filterToolReportInput: FilterToolChangeReportInput,
  ) {
    return await this.toolChangeService.aggregateAverageMonthly(
      filterToolReportInput,
    );
  }

  @UseGuards(...[AuthGuard])
  @Query(() => [ToolChangeSumReportOutput], {
    name: 'toolChangeSumReports',
  })
  async aggregateSum(
    @Args('filterToolReportInput')
    filterToolReportInput: FilterToolChangeReportInput,
  ) {
    return await this.toolChangeService.aggregateSum(filterToolReportInput);
  }
}
