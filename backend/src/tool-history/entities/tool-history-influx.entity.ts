import { ObjectType } from '@nestjs/graphql';
import { InfluxField } from 'src/influx/entities/influx.entity';
import { InfluxService } from 'src/influx/influx.service';
import {
  IInfluxAggregate,
  IInfluxFilter,
  IInfluxModel,
} from 'src/influx/interface/influx.interface';

const MEASUREMENT_NAME = 'cnc_tool';

// Payload:
// cnc_product,
// WorkshopId=w001,LineId=l001,OpCode=op10,MachineId=mc001,ProductId=hmt-20241025=0001,StartTime=1729663350361000000,EndTime=1729663350361000000
// CT=98.90,LoadSum=2798000,Count=1 1729663350361000000
@ObjectType()
export class ToolHistoryInflux extends InfluxField implements IInfluxModel {
  // Tags
  did: string = ''; // WorkshopId_LineId_OpCode_MachineId
  ProductId: string = '';
  TCode: string = '';
  // MainProgram: string = '';
  StartTime: string = '0';
  EndTime: string = '0';

  // Fields
  CT: number = 0;
  LoadSum: number = 0;
  Loss: number = 0;
  Threshold: number = 0;
  Fov: number = 0;
  Sov: number = 0;
  SV_X_Offset: number = 0;
  SV_Z_Offset: number = 0;
  Count: number = 0;

  async find(
    influxService: InfluxService,
    rangeStart?: Date,
    rangeEnd?: Date,
    rangeStartString?: string,
    tags?: IInfluxFilter,
    fields?: IInfluxFilter,
    aggregateInterval?: IInfluxAggregate,
  ): Promise<any[]> {
    const influxQueryApi = influxService.getQueryApi();
    const query = influxService.initQuery(
      MEASUREMENT_NAME,
      rangeStart,
      rangeEnd,
      rangeStartString,
      tags,
      fields,
      aggregateInterval,
      null,
      true,
    );

    return (await influxQueryApi.collectRows<ToolHistoryInflux>(query)).sort(
      (a, b) => {
        const strATime = `${a._time}`;
        const strBTime = `${b._time}`;

        const compareATime =
          strATime.replace('T', ' ').replace('Z', '').length == 19
            ? `${strATime.replace('T', ' ').replace('Z', '')}.000`
            : strATime.replace('T', ' ').replace('Z', '').padEnd(23, '0');
        const compareBTime =
          strBTime.replace('T', ' ').replace('Z', '').length == 19
            ? `${strBTime.replace('T', ' ').replace('Z', '')}.000`
            : strBTime.replace('T', ' ').replace('Z', '').padEnd(23, '0');

        if (compareATime > compareBTime) {
          return 1;
        } else {
          return -1;
        }
      },
    );
  }
  findOne(influxService: InfluxService, date: Date) {
    const influxQueryApi = influxService.getQueryApi();
    const query = influxService.initQuery(
      MEASUREMENT_NAME,
      date,
      new Date(date.getTime() + 300),
    );

    return influxQueryApi.collectRows<ToolHistoryInflux>(query);
  }
  findLast(influxService: InfluxService, tags?: IInfluxFilter) {
    const now = new Date(Date.now());
    const influxQueryApi = influxService.getQueryApi();
    const query = influxService.initQuery(
      MEASUREMENT_NAME,
      // new Date(0),
      new Date(now.getFullYear(), now.getMonth(), now.getDate() - 6),
      null,
      null,
      tags,
      null,
      null,
      true,
    );

    return influxQueryApi.collectRows<ToolHistoryInflux>(query);
  }
}
