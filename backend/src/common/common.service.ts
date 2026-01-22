import { Injectable } from '@nestjs/common';
import { LineService } from 'src/master/line/line.service';
import { MachineService } from 'src/master/machine/machine.service';
import { OperationService } from 'src/master/operation/operation.service';
import { WorkshopService } from 'src/master/workshop/workshop.service';
import {
  WorkshopListOutput,
  LineListOutput,
  OperationListOutput,
  MachineListOutput,
} from './dto/operation-info.output';

@Injectable()
export class CommonService {
  constructor(
    private readonly workshopService: WorkshopService,
    private readonly lineService: LineService,
    private readonly operationService: OperationService,
    private readonly machineService: MachineService,
  ) {}

  async findDashboardMenus() {
    const workshopList = await this.workshopService.find(null);
    const lineList = await this.lineService.find(null);
    const operationList = await this.operationService.find(null);
    const machineList = await this.machineService.find(null);

    // const lines: Line[] = [];
    // const lines: Line[] = new Array(test2.length)
    // const operation: Operation[] = new Array(test3.length);
    const menus: WorkshopListOutput[] = new Array(workshopList.length);

    workshopList.forEach((w, wIdx) => {
      const result = new WorkshopListOutput();
      result.workshopCode = w.workshopCode;
      result.workshopTitle = w.workshopName;

      const filterLines = lineList.filter(
        (l) => l.workshopCode == w.workshopCode,
      );

      result.lineList =
        filterLines.length == 0 ? null : new Array(filterLines.length);

      filterLines.forEach((e, lidx) => {
        result.lineList[lidx] = new LineListOutput();
        result.lineList[lidx].lineCode = e.lineCode;
        result.lineList[lidx].lineTitle = e.lineName;

        const fileterOperation = operationList.filter(
          (o) => o.lineCode == e.lineCode,
        );
        result.lineList[lidx].operationList = fileterOperation.map((fop) => {
          const temp = new OperationListOutput();
          temp.operationCode = fop.opCode;
          temp.operationTitle = fop.opName;

          const filterMachines = machineList.filter(
            (m) => m.opCode == temp.operationCode,
          );

          temp.machineList = filterMachines.map((mop) => {
            const temp2 = new MachineListOutput();
            temp2.machineCode = mop.machineCode;
            temp2.machineTitle = mop.machineName;

            return temp2;
          });

          return temp;
        });
      });

      menus[wIdx] = result;
    });

    return menus;
  }
}
