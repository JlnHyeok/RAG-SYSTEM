import { Body, Controller, forwardRef, Inject, Post } from '@nestjs/common';
import { CreateToolChangeAutoInput } from './dto/create-tool-change.input';
import { CommonMutationOutput } from 'src/common/dto/common.output';
import { PubSub } from 'graphql-subscriptions';
import { ToolChangeService } from './tool-change.service';
import { PUB_SUB } from 'src/app.provider';
import { TOPIC_TOOL_COUNT } from 'src/pubsub/pubsub.constants';
import { ToolHistoryService } from 'src/tool-history/tool-history.service';

@Controller('tool-exchange')
export class ToolChangeController {
  constructor(
    private readonly toolChangeService: ToolChangeService,
    @Inject(PUB_SUB)
    private readonly pubSub: PubSub,
    @Inject(forwardRef(() => ToolHistoryService))
    private readonly toolHistoryService: ToolHistoryService,
  ) {}
  // * POST
  // 1. 공구별 최근 사용량 조회
  @Post()
  async createToolChange(
    @Body() createToolChangeInput: CreateToolChangeAutoInput,
  ) {
    const output = new CommonMutationOutput();
    output.isSuccess = false;

    const newToolChange = await this.toolChangeService.create({
      workshopCode: createToolChangeInput.workshopId,
      lineCode: createToolChangeInput.lineId,
      opCode: createToolChangeInput.opCode,
      machineCode: createToolChangeInput.machineId,
      toolCode: createToolChangeInput.code,
      reasonCode: '정기교체(자동)',
      changeDate: createToolChangeInput.time,
      useCount:
        createToolChangeInput.useCount < 0 ? 0 : createToolChangeInput.useCount,
    });

    if (newToolChange.isSuccess) {
      const payloadObj = new Object();

      // 현재 전체 공구 사용 수량 조회
      const currentToolCount =
        await this.toolHistoryService.findCurrentToolCount({
          workshopId: createToolChangeInput.workshopId,
          lineId: createToolChangeInput.lineId,
          opCode: createToolChangeInput.opCode,
          machineId: createToolChangeInput.machineId,
        });

      // 반복문을 이용하여 사용 수량 및 수명 상태 초기화 (비동기 처리를 위해 for~of 사용)
      for (const t of currentToolCount) {
        if (t.isUpdateTool) {
          t.useStartTime =
            newToolChange.changeDate.getUTCMilliseconds() * 1000000;
        }
      }

      payloadObj[TOPIC_TOOL_COUNT] = currentToolCount;
      const topic = `${createToolChangeInput.workshopId}/${createToolChangeInput.lineId}/${createToolChangeInput.opCode}/${TOPIC_TOOL_COUNT}`;

      await this.pubSub.publish(topic, payloadObj);

      output.isSuccess = true;
    }

    return output;
  }
}
