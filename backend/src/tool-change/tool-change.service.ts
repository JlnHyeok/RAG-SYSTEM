import { forwardRef, Inject, Injectable } from '@nestjs/common';
import { CreateToolChangeInput } from './dto/create-tool-change.input';
import { InjectModel } from '@nestjs/mongoose';
import { Model } from 'mongoose';
import { ToolChange } from './entities/tool-change.entity';
import { FilterCommonInput } from 'src/common/dto/filter-common.input';
import { ToolService } from 'src/master/tool/tool.service';
import {
  ToolChangeAverageMonthlyReportOutput,
  ToolChangeAverageReportOutput,
  ToolChangeLastReportOutput,
  ToolChangeSumReportOutput,
} from './dto/tool-change.output';
import { ToolHistoryService } from 'src/tool-history/tool-history.service';
import {
  FilterToolChangeAvgReportInput,
  FilterToolChangeReportInput,
} from './dto/filter-tool-change.input';
import { CreateToolChangeOutput } from './dto/create-tool-change.output';
import { MachineService } from 'src/master/machine/machine.service';
import { Tool } from 'src/master/tool/entities/tool.entity';

@Injectable()
export class ToolChangeService {
  constructor(
    @InjectModel(ToolChange.name)
    private readonly toolChangeModel: Model<ToolChange>,
    @Inject(forwardRef(() => MachineService))
    private readonly machineService: MachineService,
    @Inject(forwardRef(() => ToolService))
    private readonly toolService: ToolService,
    @Inject(forwardRef(() => ToolHistoryService))
    private readonly toolHistoryService: ToolHistoryService,
  ) {}

  async create(
    createToolChangeInput: CreateToolChangeInput,
  ): Promise<CreateToolChangeOutput> {
    // const machineInfo = await this.machineService.find({
    //   opCode: createToolChangeInput.opCode,
    // });

    // if (!machineInfo || machineInfo.length == 0) {
    //   const output = new CreateToolChangeOutput();
    //   output.isSuccess = false;

    //   return output;
    // }

    // 교체 이전 공구 사용량 계산
    let toolUseCount = 0;

    if (createToolChangeInput.useCount || createToolChangeInput.useCount == 0) {
      toolUseCount = createToolChangeInput.useCount;
    } else {
      toolUseCount = await this.toolHistoryService.getCurrentToolUseCount(
        {
          workshopId: createToolChangeInput.workshopCode,
          lineId: createToolChangeInput.lineCode,
          opCode: createToolChangeInput.opCode,
          machineId: createToolChangeInput.machineCode,
        },
        createToolChangeInput.toolCode,
      );
    }
    const newToolChange = await this.toolChangeModel.create({
      ...createToolChangeInput,
      // 공구 교체 시간이 있을 경우 (엔진 데이터) 데이터 변환 (없을 경우 현재 시간 사용)
      changeDate: createToolChangeInput.changeDate
        ? new Date(createToolChangeInput.changeDate / 1000000)
        : new Date(),
      toolUseCount,
    });

    if (!newToolChange) {
      const output = new CreateToolChangeOutput();
      output.isSuccess = false;

      return output;
    }

    // 데이터 저장
    return {
      isSuccess: true,
      toolCode: newToolChange.toolCode,
      reasonCode: newToolChange.reasonCode,
      toolUseCount: newToolChange.toolUseCount,
      changeDate: newToolChange.changeDate,
    };
  }

  async findLast(filterCommonInput: FilterCommonInput, filterToolCode: string) {
    return await this.toolChangeModel
      .find({
        workshopCode: filterCommonInput.workshopId,
        lineCode: filterCommonInput.lineId,
        opCode: filterCommonInput.opCode,
        machineCode: filterCommonInput.machineId,
        toolCode: filterToolCode,
      })
      .sort({
        changeDate: -1,
      })
      .limit(1);
  }

  async aggregateLast(
    filterCommonInput: FilterCommonInput,
  ): Promise<ToolChangeLastReportOutput[]> {
    const machineInfo = await this.machineService.find({
      opCode: filterCommonInput.opCode,
    });

    if (!machineInfo || machineInfo.length == 0) {
      return [];
    }

    const toolMaster = await this.toolService.find({
      machineCode: machineInfo[0].machineCode,
    });

    const output: ToolChangeLastReportOutput[] = [];
    for (const t of toolMaster) {
      const toolUseCount = await this.toolHistoryService.getCurrentToolUseCount(
        filterCommonInput,
        t.toolCode,
      );

      output.push({
        toolCode: t.toolCode,
        toolName: t.toolName,
        reasonCode: '교체이력 없음',
        changeDate: null,
        toolUseCount: toolUseCount,
        toolUseCountAvg: 0,
      });
    }
    // const output: ToolChangeLastReportOutput[] = await Promise.all(
    //   toolMaster.map(async (t) => {
    //     const toolUseCount =
    //       await this.toolHistoryService.getCurrentToolUseCount(
    //         filterCommonInput,
    //         t.toolCode,
    //       );

    //     return {
    //       toolCode: t.toolCode,
    //       reasonCode: '교체이력 없음',
    //       changeDate: null,
    //       toolUseCount: toolUseCount,
    //       toolUseCountAvg: 0,
    //     };
    //   }),
    // );
    const aggregateResult = await this.toolChangeModel
      .aggregate([
        {
          $match: {
            workshopCode: filterCommonInput.workshopId,
            lineCode: filterCommonInput.lineId,
            opCode: filterCommonInput.opCode,
            machineCode: machineInfo[0].machineCode,
            toolUseCount: {
              $gte: 0,
            },
          },
        },
        {
          $group: {
            _id: '$toolCode',
            reasonCode: {
              $last: '$reasonCode',
            },
            changeDate: {
              $last: '$changeDate',
            },
            toolUseCountAvg: {
              $avg: '$toolUseCount',
            },
          },
        },
      ])
      .sort({
        _id: 1,
      });

    for (const r of output) {
      const current = r;
      const tempResult = aggregateResult.filter(
        (a) => a._id == current.toolCode,
      );

      if (tempResult.length > 0) {
        current.reasonCode = tempResult[0].reasonCode;
        current.changeDate = tempResult[0].changeDate;
        current.toolUseCountAvg = tempResult[0].toolUseCountAvg;
      }
    }
    // output.forEach(async (r) => {
    //   const current = await r;
    //   const tempResult = aggregateResult.filter(
    //     (a) => a._id == current.toolCode,
    //   );

    //   if (tempResult.length > 0) {
    //     current.reasonCode = tempResult[0].reasonCode;
    //     current.changeDate = tempResult[0].changeDate;
    //     current.toolUseCountAvg = tempResult[0].toolUseCountAvg;
    //   }
    // });

    return output;
  }
  async aggregateAverage(
    filterToolAvgReportInput: FilterToolChangeAvgReportInput,
  ): Promise<ToolChangeAverageReportOutput[]> {
    const machineInfo = await this.machineService.find({
      opCode: filterToolAvgReportInput.opCode,
    });

    if (!machineInfo || machineInfo.length == 0) {
      return [];
    }

    const toolMaster = await this.toolService.find({
      machineCode: machineInfo[0].machineCode,
    });

    const output: ToolChangeAverageReportOutput[] = toolMaster.map((t) => {
      return {
        toolCode: t.toolCode,
        toolName: t.toolName,
        changeCount: 0,
        toolUseCountAvg: 0,
      };
    });

    // 기간별 통계를 위해 필터 항목 추가
    const aggregateResult = await this.toolChangeModel
      .aggregate([
        {
          $match: {
            workshopCode: filterToolAvgReportInput.workshopId,
            lineCode: filterToolAvgReportInput.lineId,
            opCode: filterToolAvgReportInput.opCode,
            machineCode: machineInfo[0].machineCode,
            changeDate: {
              $gte: filterToolAvgReportInput.rangeStart,
              $lte: filterToolAvgReportInput.rangeEnd,
            },
            toolUseCount: {
              $gte: 0,
            },
          },
        },
        {
          $group: {
            _id: '$toolCode',
            changeCount: {
              $count: {},
            },
            toolUseCountAvg: {
              $avg: '$toolUseCount',
            },
          },
        },
      ])
      .sort({
        _id: 1,
      });

    output.forEach(async (r) => {
      const current = await r;
      const tempResult = aggregateResult.filter(
        (a) => a._id == current.toolCode,
      );

      if (tempResult.length > 0) {
        current.changeCount = tempResult[0].changeCount;
        current.toolUseCountAvg = tempResult[0].toolUseCountAvg;
      }
    });

    return output;
  }
  async aggregateSum(
    filterToolChangeReportInput: FilterToolChangeReportInput,
  ): Promise<ToolChangeSumReportOutput[]> {
    const output: ToolChangeSumReportOutput[] = [];
    const today = new Date(Date.now());

    const machineInfo = await this.machineService.find({
      opCode: filterToolChangeReportInput.opCode,
    });
    let toolMaster = [];

    if (machineInfo && machineInfo.length > 0) {
      toolMaster = await this.toolService.find({
        machineCode: machineInfo[0].machineCode,
      });
    }

    // 조회 일자 기준 6개월 데이터에 대한 통계 (향후 기간 변경 가능성 있음)
    const filterBeginDate = new Date(
      filterToolChangeReportInput.beginYear,
      filterToolChangeReportInput.beginMonth - 1,
      1,
    );
    const filterEndDate = new Date(
      filterToolChangeReportInput.endYear,
      filterToolChangeReportInput.endMonth - 1,
      1,
    );
    let tempDate = new Date(filterBeginDate);

    const loopCount = 1;
    while (tempDate <= filterEndDate) {
      output.push({
        reportDate: `${tempDate.getFullYear()}-${(tempDate.getMonth() + 1).toString().padStart(2, '0')}`,
        toolUseCount: await Promise.all(
          toolMaster.map(async (t) => {
            return {
              toolCode: t.toolCode,
              toolName: t.toolName,
              toolUseCountSum:
                // 당월의 경우 현재 사용 수량 합산
                tempDate.getMonth() == today.getMonth()
                  ? await this.toolHistoryService.getCurrentToolUseCount(
                      {
                        workshopId: filterToolChangeReportInput.workshopId,
                        lineId: filterToolChangeReportInput.lineId,
                        opCode: filterToolChangeReportInput.opCode,
                        machineId: filterToolChangeReportInput.machineId,
                      },
                      t.toolCode,
                    )
                  : 0,
            };
          }),
        ),
      });

      tempDate = new Date(tempDate.setMonth(tempDate.getMonth() + loopCount));
    }

    const aggregateResult = await this.toolChangeModel
      .aggregate([
        {
          $match: {
            workshopCode: filterToolChangeReportInput.workshopId,
            lineCode: filterToolChangeReportInput.lineId,
            opCode: filterToolChangeReportInput.opCode,
            machineCode: filterToolChangeReportInput.machineId,
          },
        },
        {
          $group: {
            _id: [
              {
                $dateToString: {
                  format: '%Y-%m',
                  date: '$changeDate',
                },
              },
              '$toolCode',
            ],
            toolUseCount: {
              $sum: '$toolUseCount',
            },
          },
        },
        {
          $project: {
            reportDate: { $arrayElemAt: ['$_id', 0] },
            toolCode: { $arrayElemAt: ['$_id', 1] },
            toolUseCountSum: '$toolUseCount',
          },
        },
      ])
      .sort({
        reportDate: 1,
        toolCode: 1,
      });

    output.forEach((p) => {
      const tempResult = aggregateResult.filter((q) => {
        if (p.reportDate == q.reportDate) {
          return true;
        }

        return false;
      });
      const tempToolCount = tempResult.map((r) => {
        return {
          toolCode: r.toolCode,
          toolUseCountSum: r.toolUseCountSum,
        };
      });

      p.toolUseCount.forEach(async (t) => {
        if (tempToolCount.filter((s) => s.toolCode == t.toolCode).length > 0) {
          t.toolUseCountSum += tempToolCount.filter(
            (s) => s.toolCode == t.toolCode,
          )[0].toolUseCountSum;
        }
      });
    });

    return output;
  }
  async aggregateAverageMonthly(
    filterToolChangeReportInput: FilterToolChangeReportInput,
  ): Promise<ToolChangeAverageMonthlyReportOutput[]> {
    const output: ToolChangeAverageMonthlyReportOutput[] = [];
    const today = new Date(Date.now());

    const machineInfo = await this.machineService.find({
      opCode: filterToolChangeReportInput.opCode,
    });
    let toolMaster: Tool[] = [];

    if (machineInfo && machineInfo.length > 0) {
      toolMaster = await this.toolService.find({
        machineCode: machineInfo[0].machineCode,
      });
    }

    // 조회 일자 기준 6개월 데이터에 대한 통계 (향후 기간 변경 가능성 있음)
    const filterBeginDate = new Date(
      filterToolChangeReportInput.beginYear,
      filterToolChangeReportInput.beginMonth - 1,
      1,
    );
    const filterEndDate = new Date(
      filterToolChangeReportInput.endYear,
      filterToolChangeReportInput.endMonth - 1,
      1,
    );
    let tempDate = new Date(filterBeginDate);

    const loopCount = 1;
    while (tempDate <= filterEndDate) {
      const initOutput = new ToolChangeAverageMonthlyReportOutput();
      initOutput.reportDate = `${tempDate.getFullYear()}-${(tempDate.getMonth() + 1).toString().padStart(2, '0')}`;
      initOutput.toolUseCount = await Promise.all(
        toolMaster.map(async (t) => {
          const tempUseCount = new ToolChangeAverageReportOutput();
          tempUseCount.toolCode = t.toolCode;
          tempUseCount.toolName = t.toolName;
          tempUseCount.changeCount = 0;
          tempUseCount.toolUseCountAvg = 0;

          return tempUseCount;
        }),
      );

      output.push(initOutput);

      tempDate = new Date(tempDate.setMonth(tempDate.getMonth() + loopCount));
    }

    const aggregateResult = await this.toolChangeModel
      .aggregate([
        {
          $match: {
            workshopCode: filterToolChangeReportInput.workshopId,
            lineCode: filterToolChangeReportInput.lineId,
            opCode: filterToolChangeReportInput.opCode,
            machineCode: filterToolChangeReportInput.machineId,
          },
        },
        {
          $group: {
            _id: [
              {
                $dateToString: {
                  format: '%Y-%m',
                  date: '$changeDate',
                },
              },
              '$toolCode',
            ],
            changeCount: {
              $count: {},
            },
            toolUseCountAvg: {
              $avg: '$toolUseCount',
            },
          },
        },
        {
          $project: {
            reportDate: { $arrayElemAt: ['$_id', 0] },
            toolCode: { $arrayElemAt: ['$_id', 1] },
            changeCount: '$changeCount',
            toolUseCountAvg: '$toolUseCountAvg',
          },
        },
      ])
      .sort({
        reportDate: 1,
        toolCode: 1,
      });

    output.forEach((p) => {
      const tempResult = aggregateResult.filter((q) => {
        if (p.reportDate == q.reportDate) {
          return true;
        }

        return false;
      });
      const tempToolCount = tempResult.map((r) => {
        return {
          toolCode: r.toolCode,
          changeCount: r.changeCount,
          toolUseCountAvg: r.toolUseCountAvg,
        };
      });

      p.toolUseCount.forEach(async (t) => {
        if (tempToolCount.filter((s) => s.toolCode == t.toolCode).length > 0) {
          const currentToolCount = tempToolCount.filter(
            (s) => s.toolCode == t.toolCode,
          )[0];
          t.changeCount += currentToolCount.changeCount;
          t.toolUseCountAvg += currentToolCount.toolUseCountAvg;
        }
      });
    });

    return output;
  }
}
