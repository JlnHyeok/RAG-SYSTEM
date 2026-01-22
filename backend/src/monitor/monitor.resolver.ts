import { Resolver, Query, Args } from '@nestjs/graphql';
import { MonitorService } from './monitor.service';
// import { AbnormalService } from 'src/rams_bak/abnormal/abnormal.service';
// import { ProductService } from 'src/rams_bak/product/product.service';
import { MonitorOutput } from './dto/monitor.output';
// import { FilterAbnormalInput } from 'src/rams_bak/abnormal/dto/filter-abnormal.input';
// import { RawService } from 'src/rams_bak/raw/raw.service';
import { UseGuards } from '@nestjs/common';
import { AuthGuard } from 'src/auth/auth.guard';
import { ROLE_USER } from 'src/role/role.constants';
import { Role } from 'src/role/role.decorator';
import { RoleGuard } from 'src/role/role.guard';
// import { ToolService } from 'src/tool/tool.service';
import { FilterCommonInput } from 'src/common/dto/filter-common.input';
import { RawService } from 'src/raw/raw.service';
import { FilterAbnormalInput } from 'src/abnormal/dto/filter-abnormal.input';
import { ProductService } from 'src/product/product.service';
import { AbnormalService } from 'src/abnormal/abnormal.service';
import { ToolHistoryService } from 'src/tool-history/tool-history.service';
import { WorkshopMonitorOutput } from './dto/monitor.output';
import { LineMonitorOutput } from './dto/monitor.output';
import { Operation } from 'src/master/operation/entities/operation.entity';
import { Line } from 'src/master/line/entities/line.entity';
import { LineService } from 'src/master/line/line.service';
import { OperationService } from 'src/master/operation/operation.service';
import { FilterOperationInput } from 'src/master/operation/dto/filter-operation.input';
import { WorkshopService } from 'src/master/workshop/workshop.service';
import { MachineService } from 'src/master/machine/machine.service';

@Resolver(() => MonitorOutput)
export class MonitorResolver {
  constructor(
    private readonly monitorService: MonitorService,
    private readonly abnormalService: AbnormalService,
    private readonly productService: ProductService,
    private readonly rawService: RawService,
    private readonly toolHistoryService: ToolHistoryService,
    private readonly operationService: OperationService,
    private readonly workshopService: WorkshopService,
    private readonly lineService: LineService,
    private readonly machineService: MachineService,
  ) {}

  // * Query
  // 1. CNC 현재 데이터 조회
  @UseGuards(...[AuthGuard])
  @Query(() => MonitorOutput, { name: 'monitorCnc' })
  async find(
    @Args('filterCommonInput')
    filterCommonInput: FilterCommonInput,
  ) {
    const opInfo = await this.operationService.findOne(
      filterCommonInput.opCode,
    );
    // * 당일 이상 감지 이력 조회
    const today = new Date(Date.now());
    // // 조회 필터 객체 초기화 (당일)
    const abnormalFilter = new FilterAbnormalInput();
    abnormalFilter.commonFilter = filterCommonInput;
    abnormalFilter.rangeStart = new Date(
      today.getFullYear(),
      today.getMonth(),
      today.getDate(),
    );
    abnormalFilter.rangeEnd = new Date(
      today.getFullYear(),
      today.getMonth(),
      today.getDate() + 1,
    );
    abnormalFilter.count = 10;

    // DB 데이터 조회
    // const lastAbnormal = (await this.abnormalService.find(abnormalFilter))
    //   .abnormals;

    // * 최근 생산 이력 조회
    const lastProduct = await this.productService.findLast(filterCommonInput);

    // * 최근 CNC 파라미터 조회
    const lastRaw = await this.rawService.findLast(filterCommonInput);

    // * 최근 공구 사용량 조회
    const lastToolCount =
      await this.toolHistoryService.findCurrentToolCount(filterCommonInput);

    // * 당일 가동 시간 조회
    const todayOperation = await this.rawService.aggregateOperation({
      reportField: 'Run',
      rangeStart: abnormalFilter.rangeStart,
      rangeStop: abnormalFilter.rangeEnd,
      status: '3',
      aggregateInterval: '1d',
      commonFilter: filterCommonInput,
    });

    // Output 객체 초기화
    const monitorOutput = new MonitorOutput();
    monitorOutput.opCode = opInfo.opCode;
    monitorOutput.opName = opInfo.opName;
    monitorOutput.operationTime =
      todayOperation.length == 0 ? 0 : todayOperation[0].operationTime;
    monitorOutput.product = lastProduct.productNo ? lastProduct : null;
    monitorOutput.toolCount = lastToolCount;
    monitorOutput.parameter = lastRaw;

    return monitorOutput;
  }

  // 라인 코드로 하위 공정 정보 조회
  @UseGuards(...[AuthGuard])
  @Query(() => LineMonitorOutput, { name: 'monitorLine' })
  async findOperations(
    @Args('lineCode')
    lineCode: string,
  ) {
    const currentLine = await this.lineService.findOne(lineCode);
    const monitorOperationList = await this.operationService.find(null);
    const lineMonitorOutput: LineMonitorOutput = new LineMonitorOutput();
    lineMonitorOutput.lineCode = lineCode;
    lineMonitorOutput.lineName = currentLine ? currentLine.lineName : '';

    const filterCommonInput: FilterCommonInput = new FilterCommonInput();
    const filerOperationList: Operation[] = monitorOperationList.filter(
      (o) => o.lineCode == lineCode,
    );
    lineMonitorOutput.operationMonitors =
      filerOperationList.length == 0
        ? null
        : new Array(filerOperationList.length);

    for (let oIdx = 0; oIdx < filerOperationList.length; oIdx++) {
      const o = filerOperationList[oIdx];
      const machines = await this.machineService.find({
        workshopCode: o.workshopCode,
        lineCode: o.lineCode,
        opCode: o.opCode,
      });

      filterCommonInput.workshopId = o.workshopCode;
      filterCommonInput.lineId = o.lineCode;
      filterCommonInput.opCode = o.opCode;

      if (machines && machines.length > 0) {
        filterCommonInput.machineId = machines[0].machineCode;
      }

      lineMonitorOutput.operationMonitors[oIdx] =
        await this.find(filterCommonInput);
    }

    return lineMonitorOutput;
  }

  // 공장 코드로 하위 라인 정보 조회
  @UseGuards(...[AuthGuard])
  @Query(() => WorkshopMonitorOutput, { name: 'monitorWorkshop' })
  async findLines(
    @Args('workshopCode')
    workshopCode: string,
  ) {
    const currentWorkshop = await this.workshopService.findOne(workshopCode);
    const monitorLineList = await this.lineService.find(null);
    const workshopMonitorOutput: WorkshopMonitorOutput =
      new WorkshopMonitorOutput();
    workshopMonitorOutput.workshopCode = workshopCode;
    workshopMonitorOutput.workshopName = currentWorkshop
      ? currentWorkshop.workshopName
      : '';

    const filerLineList: Line[] = monitorLineList.filter(
      (l) => l.workshopCode == workshopCode,
    );

    workshopMonitorOutput.lineMonitors =
      filerLineList.length == 0 ? null : new Array(filerLineList.length);

    for (let lIdx = 0; lIdx < filerLineList.length; lIdx++) {
      const l = filerLineList[lIdx];
      workshopMonitorOutput.lineMonitors[lIdx] = await this.findOperations(
        l.lineCode,
      );
    }

    return workshopMonitorOutput;
  }
}
