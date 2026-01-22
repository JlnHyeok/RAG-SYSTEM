import {
  Body,
  Controller,
  forwardRef,
  Get,
  Inject,
  Post,
  Query,
} from '@nestjs/common';
import { PubSub } from 'graphql-subscriptions';
import { PUB_SUB } from 'src/app.provider';
import { FilterCommonInput } from 'src/common/dto/filter-common.input';
import { ToolHistoryService } from './tool-history.service';
import { CreateToolHistoryInput } from './dto/create-tool-history.input';
import { CommonMutationOutput } from 'src/common/dto/common.output';
import { TOPIC_TOOL_COUNT } from 'src/pubsub/pubsub.constants';
import { ToolService } from 'src/master/tool/tool.service';
import { MachineService } from 'src/master/machine/machine.service';

@Controller('tool-history')
export class ToolHistoryController {
  constructor(
    @Inject(PUB_SUB)
    private readonly pubSub: PubSub,
    @Inject(forwardRef(() => ToolHistoryService))
    private readonly toolHistoryService: ToolHistoryService,
    @Inject(forwardRef(() => MachineService))
    private readonly machineService: MachineService,
    @Inject(forwardRef(() => ToolService))
    private readonly toolService: ToolService,
  ) {}

  // * GET
  // 1. 공구별 최근 사용량 조회
  @Get('count')
  async getRecentCount(@Query() filterCommonInput: FilterCommonInput) {
    // TODO: 공구 교체 모듈 개발 후 수정
    const result =
      await this.toolHistoryService.findCurrentToolCount(filterCommonInput);

    return result.map((t) => {
      return { code: t.code, useCount: t.useCount };
    });
  }

  // * POST
  // 1. 공구별 최근 사용량 조회
  @Post()
  async createToolHistory(
    @Body() createToolHistoryInput: CreateToolHistoryInput,
  ) {
    const output = new CommonMutationOutput();
    output.isSuccess = false;

    const machineInfo = await this.machineService.find({
      opCode: createToolHistoryInput.opCode,
    });
    const currentToolInfo =
      machineInfo && machineInfo.length > 0
        ? await this.toolService.findOne({
            machineCode: machineInfo[0].machineCode,
            toolCode: createToolHistoryInput.code,
          })
        : null;

    // 입력한 공구 번호가 기준 정보에 없을 경우 함수 종료
    if (!currentToolInfo) {
      output.isSuccess = false;
      return output;
    }

    // 공구 사용 시작 시 Publish
    if (createToolHistoryInput.type == 'S') {
      const payloadObj = new Object();

      // 공정 코드를 이용하여 설비 기준 정보 조회
      const machineInfo = await this.machineService.find({
        opCode: createToolHistoryInput.opCode,
      });

      // 현재 전체 공구 사용 수량 조회
      const currentToolCount =
        await this.toolHistoryService.findCurrentToolCount({
          workshopId: createToolHistoryInput.workshopId,
          lineId: createToolHistoryInput.lineId,
          opCode: createToolHistoryInput.opCode,
          machineId: createToolHistoryInput.machineId,
        });

      // 반복문을 이용하여 사용 수량 및 수명 상태 초기화 (비동기 처리를 위해 for~of 사용)
      for (const t of currentToolCount) {
        // 현재 공구 기준 정보 조회
        const tempToolInfo =
          machineInfo && machineInfo.length > 0
            ? await this.toolService.findOne({
                machineCode: machineInfo[0].machineCode,
                toolCode: t.code,
              })
            : null;
        t.isUpdateTool = t.code == createToolHistoryInput.code;

        // 조회된 공구 기준 정보가 있을 경우 수량 및 수명 상태 초기화
        if (tempToolInfo) {
          t.useCount = t.useCount;
          t.toolStatusCount = this.toolHistoryService.parseToolStatus(
            t.useCount,
            tempToolInfo,
          );
        }

        if (t.isUpdateTool) {
          t.useStartTime = createToolHistoryInput.startTime;
        }
      }

      // currentToolCount.forEach((t) => {
      //   t.isUpdateTool = t.code == createToolHistoryInput.code;
      //   t.useCount = t.useCount + 1;

      //   if (t.isUpdateTool) {
      //     t.useStartTime = createToolHistoryInput.startTime;
      //   }
      // });

      payloadObj[TOPIC_TOOL_COUNT] = currentToolCount;
      const topic = `${createToolHistoryInput.workshopId}/${createToolHistoryInput.lineId}/${createToolHistoryInput.opCode}/${TOPIC_TOOL_COUNT}`;

      await this.pubSub.publish(topic, payloadObj);

      output.isSuccess = true;
      return output;
    }

    // 공구 사용 종료 시 RDB 저장
    const newToolHistory = await this.toolHistoryService.create(
      createToolHistoryInput,
    );

    if (newToolHistory) {
      // const payloadObj = new Object();

      // currentToolCount.forEach((t) => {
      //   t.isUpdateTool = t.code == createToolHistoryInput.code;

      //   if (t.isUpdateTool) {
      //     t.useStartTime = createToolHistoryInput.startTime;
      //   }
      // });

      // payloadObj[TOPIC_TOOL_COUNT] = currentToolCount;
      // await this.pubSub.publish(TOPIC_TOOL_COUNT, payloadObj);

      output.isSuccess = true;
      return output;
    }

    return output;
  }
}
