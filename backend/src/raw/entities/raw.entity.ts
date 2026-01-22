import { ObjectType } from '@nestjs/graphql';
import { InfluxField } from 'src/influx/entities/influx.entity';
import { InfluxService } from 'src/influx/influx.service';
import {
  IInfluxAggregate,
  IInfluxFilter,
  IInfluxModel,
} from 'src/influx/interface/influx.interface';

const MEASUREMENT_NAME = 'cnc_analyze';

@ObjectType()
export class Raw extends InfluxField implements IInfluxModel {
  // 2024.11.07 TSDB 조회 성능 이슈로 인한 스키마 변경
  // Tags
  did: string = ''; // WorkshopId_LineId_OpCode_MachineId
  TCode: string = '';
  ProductId: string = '';

  // Fields
  Run: number = 0;
  MainProgram: number = 0;
  SubProgram: number = 0;
  MCode: number = 0;
  Feed: number = 0;
  Fov: number = 0;
  Sov: number = 0;
  SV_X_Offset: number = 0;
  SV_Z_Offset: number = 0;
  SV_X_Pos: number = 0;
  SV_Z_Pos: number = 0;
  TCount1: number = 0;
  TCount2: number = 0;
  TCount3: number = 0;
  TCount4: number = 0;
  Interval: number = 0;

  // AI Engine Data
  Load: number = 0;
  Predict: number = 0;
  Loss: number = 0;

  // // Tags
  // WorkshopId: string = '';
  // LineId: string = '';
  // OpCode: string = '';
  // MachineId: string = '';
  // // Aut: string = '';
  // Run: string = '';
  // MainProgram: string = '';
  // SubProgram: string = '';
  // MCode: string = '';
  // TCode: string = '';
  // ProductId: string = '';

  // // Fields
  // Feed: number = 0;
  // Fov: number = 0;
  // Sov: number = 0;
  // SV_X_Offset: number = 0;
  // SV_X_Pos: number = 0;
  // SV_Z_Offset: number = 0;
  // SV_Z_Pos: number = 0;
  // Load: number = 0;
  // Interval: number = 0;

  // // AI Engine Data
  // Predict: number = 0;
  // Loss: number = 0;
  // PredictFlag: number = 0;

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

    return (await influxQueryApi.collectRows<Raw>(query)).sort((a, b) => {
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
    });
  }
  findOne(influxService: InfluxService, date: Date) {
    const influxQueryApi = influxService.getQueryApi();
    const query = influxService.initQuery(
      MEASUREMENT_NAME,
      date,
      new Date(date.getTime() + 300),
    );

    return influxQueryApi.collectRows<Raw>(query);
  }
  findLast(influxService: InfluxService, tags?: IInfluxFilter) {
    const now = new Date(Date.now());
    const influxQueryApi = influxService.getQueryApi();
    const query = influxService.initQuery(
      MEASUREMENT_NAME,
      // new Date(0),
      // new Date(now.getFullYear(), now.getMonth(), now.getDate() - 6),
      null,
      null,
      '1h',
      tags,
      null,
      null,
      true,
    );

    return influxQueryApi.collectRows<Raw>(query);
  }
}
