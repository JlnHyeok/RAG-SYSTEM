import { forwardRef, Inject, Injectable } from '@nestjs/common';
import { CreateToolHistoryInput } from './dto/create-tool-history.input';
import { InjectModel } from '@nestjs/mongoose';
import { PubSub } from 'graphql-subscriptions';
import { Model } from 'mongoose';
import { PUB_SUB, TOOL_ENTITY } from 'src/app.provider';
import {
  convertInfluxFilter,
  FilterCommonInput,
} from 'src/common/dto/filter-common.input';
import { ToolHistory } from './entities/tool-history.entity';
import { ToolService } from 'src/master/tool/tool.service';
import { ToolChangeService } from 'src/tool-change/tool-change.service';
import {
  ToolCountLastOutput,
  ToolHistoryInfluxOutput,
} from './dto/tool-history.output';
import { Tool } from 'src/master/tool/entities/tool.entity';
import { TOPIC_TOOL_COUNT } from 'src/pubsub/pubsub.constants';
import { MachineService } from 'src/master/machine/machine.service';
import { IInfluxModel } from 'src/influx/interface/influx.interface';
import { FilterToolHistoryInfluxInput } from './dto/filter-tool-history.input';
import { FilterInfluxTagInput } from 'src/influx/dto/filter-influx.input';
import { InfluxService } from 'src/influx/influx.service';
import { ToolHistoryInflux } from './entities/tool-history-influx.entity';

@Injectable()
export class ToolHistoryService {
  constructor(
    @Inject(PUB_SUB)
    private readonly pubSub: PubSub,
    @InjectModel(ToolHistory.name)
    private readonly toolHistoryModel: Model<ToolHistory>,
    @Inject(forwardRef(() => ToolService))
    private readonly toolService: ToolService,
    @Inject(forwardRef(() => ToolChangeService))
    private readonly toolChangeService: ToolChangeService,
    @Inject(forwardRef(() => MachineService))
    private readonly machineService: MachineService,
    @Inject(TOOL_ENTITY)
    private readonly toolInfluxModel: IInfluxModel,
    private readonly influxService: InfluxService,
  ) {}

  async create(createToolHistoryInput: CreateToolHistoryInput) {
    return await this.toolHistoryModel.create({
      workshopCode: createToolHistoryInput.workshopId,
      lineCode: createToolHistoryInput.lineId,
      opCode: createToolHistoryInput.opCode,
      machineCode: createToolHistoryInput.machineId,
      toolCode: createToolHistoryInput.code,
      toolCt: createToolHistoryInput.ct,
      toolLoadSum: createToolHistoryInput.loadSum,
      toolUseCount: 1,
      // toolUseDate: new Date(),
      toolUseStartDate: new Date(createToolHistoryInput.startTime / 1000000),
      toolUseEndDate: new Date(createToolHistoryInput.endTime / 1000000),
    });
  }

  find() {
    return `This action returns all toolHistory`;
  }

  // Influx Method
  async findInflux(
    filterToolHistoryInfluxInput: FilterToolHistoryInfluxInput,
    fields?: string[],
  ) {
    if (
      !filterToolHistoryInfluxInput.rangeStart &&
      !filterToolHistoryInfluxInput.rangeStartString
    ) {
      return [];
    }

    let outputs: ToolHistoryInfluxOutput[] = [];

    // 동적 필드 할당을 위해 필드명 취득
    const outputProps = Object.getOwnPropertyNames(
      new ToolHistoryInfluxOutput(),
    );
    const tagArray = convertInfluxFilter(
      filterToolHistoryInfluxInput.commonFilter,
    );

    if (filterToolHistoryInfluxInput.tags) {
      const filterTagArray = filterToolHistoryInfluxInput.tags.map((t) => {
        const temp: FilterInfluxTagInput = new FilterInfluxTagInput();
        temp.tagName = t.tagName;
        temp.tagValue =
          t.tagName == 'TCode' ? t.tagValue.replaceAll('T', '') : t.tagValue;

        return temp;
      });

      tagArray.push(...filterTagArray);
    }

    // TSDB 데이터 조회
    const queryData: ToolHistoryInflux[] = await this.toolInfluxModel.find(
      this.influxService,
      filterToolHistoryInfluxInput.rangeStart
        ? new Date(filterToolHistoryInfluxInput.rangeStart)
        : null,
      filterToolHistoryInfluxInput.rangeStop
        ? new Date(filterToolHistoryInfluxInput.rangeStop)
        : null,
      filterToolHistoryInfluxInput.rangeStartString,
      tagArray
        ? {
            operator: 'and',
            values: tagArray.map((t) => {
              return t.getInfluxFilter();
            }),
          }
        : null,
      !fields || fields.length != outputProps.length
        ? {
            operator: 'or',
            values: fields.map((f) => {
              return {
                property: '_field',
                operator: '==',
                value: f,
              };
            }),
          }
        : null,
      filterToolHistoryInfluxInput.aggregateInterval
        ? {
            aggregation: 'mean',
            interval: filterToolHistoryInfluxInput.aggregateInterval,
            dropColumns: [
              'host',
              'ProductId',
              'StartTime',
              'EndTime',
              'MainProgram',
            ],
            createEmpty: false,
          }
        : null,
    );

    outputs = queryData.map((d) => {
      const operationInfo = this.didToOperationInfo(d.did);
      const tempData = new ToolHistoryInfluxOutput();
      tempData.time = new Date(d._time);
      tempData.WorkshopCode = operationInfo[0];
      tempData.LineCode = operationInfo[1];
      tempData.OpCode = operationInfo[2];
      tempData.MachineCode = operationInfo[3];
      tempData.TCode = d.TCode;
      // tempData.startTime = new Date(parseInt(d.StartTime) / 1000000);
      // tempData.endTime = new Date(parseInt(d.EndTime) / 1000000);
      tempData.CT = d.CT ? d.CT / 1000000000 : 0;
      tempData.LoadSum = d.LoadSum ? d.LoadSum : 0;
      tempData.Loss = d.Loss ? d.Loss : 0;
      tempData.Count = d.Count ? d.Count : 0;

      return tempData;
    });

    return outputs;
  }

  // * Subscription Method
  // 1. 공구 사용량 모니터링 등록
  // monitor() {
  //   // Pub-Sub 토픽 등록
  //   return this.pubSub.asyncIterator(TOPIC_TOOL_COUNT);
  // }
  async monitor(filterCommonInput: FilterCommonInput) {
    // Pub-Sub 토픽 등록
    const topic = `${filterCommonInput.workshopId}/${filterCommonInput.lineId}/${filterCommonInput.opCode}/${TOPIC_TOOL_COUNT}`;

    return this.pubSub.asyncIterator(topic);
  }

  async findCurrentToolCount(filterCommonInput: FilterCommonInput) {
    const output: ToolCountLastOutput[] = [];
    const machineInfo = await this.machineService.find({
      opCode: filterCommonInput.opCode,
    });

    if (!machineInfo || machineInfo.length == 0) {
      return [];
    }

    // TODO: 공구 교체 모듈 개발 후 수정
    const toolInfo = await this.toolService.find({
      machineCode: filterCommonInput.machineId
        ? filterCommonInput.machineId
        : machineInfo[0].machineCode,
    });
    const lastHistory =
      await this.toolChangeService.aggregateLast(filterCommonInput);

    for (const t of toolInfo) {
      const currentCount = lastHistory.find((c) => c.toolCode == t.toolCode);
      const result: ToolCountLastOutput = {
        code: t.toolCode,
        no: t.toolOrder,
        useCount: currentCount ? currentCount.toolUseCount : 0,
        maxCount: t.maxCount,
        toolStatusCount: currentCount
          ? this.parseToolStatus(currentCount.toolUseCount, t)
          : 'OK',
        isUpdateTool: false,
        useStartTime: null,
      };

      output.push(result);
    }

    return output;

    return toolInfo.map((t) => {
      const currentCount = lastHistory.find((c) => c.toolCode == t.toolCode);
      const result: ToolCountLastOutput = {
        code: t.toolCode,
        no: t.toolOrder,
        useCount: currentCount ? currentCount.toolUseCount : 0,
        maxCount: t.maxCount,
        toolStatusCount: currentCount
          ? this.parseToolStatus(currentCount.toolUseCount, t)
          : 'OK',
        isUpdateTool: false,
        useStartTime: null,
      };

      return result;
    });
  }

  async getCurrentToolUseCount(
    filterCommonInput: FilterCommonInput,
    filterToolCode: string,
  ) {
    // const machineInfo = await this.machineService.find({
    //   opCode: filterCommonInput.opCode,
    // });

    // if (!machineInfo || machineInfo.length == 0) {
    //   return 0;
    // }

    let filterBeginDate = new Date(0);
    // 직전 교체 일시 취득
    const lastToolChange = await this.toolChangeService.findLast(
      {
        workshopId: filterCommonInput.workshopId,
        lineId: filterCommonInput.lineId,
        opCode: filterCommonInput.opCode,
        machineId: filterCommonInput.machineId,
      },
      filterToolCode,
    );
    if (lastToolChange.length > 0) {
      filterBeginDate = lastToolChange[0].changeDate;
    }

    const toolHistory = await this.toolHistoryModel.find({
      workshopCode: filterCommonInput.workshopId,
      lineCode: filterCommonInput.lineId,
      opCode: filterCommonInput.opCode,
      machineCode: filterCommonInput.machineId,
      toolCode: filterToolCode,
      toolUseStartDate: {
        $gte: filterBeginDate,
      },
    });
    return toolHistory.length;
  }

  parseToolStatus(useCount: number, toolInfo: Tool) {
    return useCount / toolInfo.maxCount >= 1
      ? 'ERROR'
      : useCount / toolInfo.maxCount >= toolInfo.warnRate / 100
        ? 'WARN'
        : 'OK';
  }
  private didToOperationInfo(did: string) {
    const result = ['', '', '', ''];
    const operationInfo = did.split('_');

    if (operationInfo.length == 4) {
      return operationInfo;
    }

    return result;
  }
}
