import { Injectable } from '@nestjs/common';

import { InfluxDB, QueryApi } from '@influxdata/influxdb-client';
import {
  IInfluxAggregate,
  IInfluxFilter,
  IInfluxRelationFilter,
} from './interface/influx.interface';

const INFLUX_RECORD_KEYWORD = 'r';

@Injectable()
export class InfluxService {
  private influx: InfluxDB;
  private influxQueryApi: QueryApi;

  /*
  INFLUX_URL=http://127.0.0.1:8086
  INFLUX_TOKEN=SN9zSb2mOaQetu7qAlIOYmiZWPGIUoDa2BnkDszdnsfWWUMS3g3JdEjwfTU1BV1vsa8BlvJn1Wwc73Lwn1iy3A==
  INFLUX_ORG=roylabs
  */
  private influxUrl: string;
  private influxToken: string;
  private influxOrg: string;
  private influxBucket: string;

  constructor() {
    this.influxUrl = process.env.INFLUX_URL;
    this.influxToken = process.env.INFLUX_TOKEN;
    this.influxOrg = process.env.INFLUX_ORG;
    this.influxBucket = process.env.INFLUX_BUCKET;

    this.createInflux();
  }

  private createInflux() {
    this.influx = new InfluxDB({
      url: this.influxUrl,
      token: this.influxToken,
      timeout: 60000,
    });
  }
  public getQueryApi() {
    this.influxQueryApi = this.influx.getQueryApi(this.influxOrg);

    return this.influxQueryApi;
  }

  public initQuery(
    measurement: string,
    rangeStart?: Date,
    rangeStop?: Date,
    rangeStartString?: string,
    tags?: IInfluxFilter,
    fields?: IInfluxFilter,
    aggregation?: IInfluxAggregate,
    isLast?: boolean,
    isPivot?: boolean,
  ): string {
    let query = '';

    query += `${this.initBucket(this.influxBucket)}`;
    query += ` ${this.initRange(rangeStart, rangeStop, rangeStartString)}`;
    query += ` |> filter(fn: (${INFLUX_RECORD_KEYWORD}) => ${INFLUX_RECORD_KEYWORD}._measurement == "${measurement}")`;
    query += !tags
      ? ''
      : ` |> filter(fn: (${INFLUX_RECORD_KEYWORD}) => ${this.initFilter(tags)})`;
    query += !fields
      ? ''
      : ` |> filter(fn: (${INFLUX_RECORD_KEYWORD}) => ${this.initFilter(fields)})`;
    //query += ` |> filter(fn: (r) => r["_field"] == "load" or r["_field"] == "loss" or r["_field"] == "predict")`;
    query += !aggregation ? '' : `${this.initAggregate(aggregation, !tags)}`;
    query += ` ${this.initDropQuery()}`;
    // 데이터 축소를 위해 Pivot 처리
    query +=
      (isLast == null || isLast == true) &&
      (isPivot == null || isPivot == false)
        ? ''
        : ` |> pivot(rowKey: ["_time"], columnKey:["_field"], valueColumn: "_value")`;
    query += isLast == null ? '' : `${this.initSelectorQuery(isLast)}`;

    return query;
  }

  //* Private Method
  // 1. Bucket Query 초기화
  initBucket(bucket: string) {
    return `from(bucket:"${bucket}")`;
  }
  // 2. Range Query 초기화
  initRange(rangeStart?: Date, rangeEnd?: Date, rangeStartString?: string) {
    if (rangeStartString) {
      return this.initRelativeRange(rangeStartString);
    }

    return this.initAbsoluteRange(rangeStart, rangeEnd);
  }
  // 2. Range Query 초기화 (Timestamp 형식)
  initAbsoluteRange(rangeStart: Date, rangeEnd?: Date) {
    if (rangeEnd) {
      return ` |> range(start: ${rangeStart.toISOString()}, stop: ${rangeEnd.toISOString()})`;
    }

    return ` |> range(start: ${rangeStart.toISOString()})`;
  }
  // 3.Range Query 초기화 (Relative TimeString 형식)
  initRelativeRange(rangeStartString: string) {
    return ` |> range(start: -${rangeStartString})`;
  }
  // 4. Filter Query 초기화
  initFilter(filter: IInfluxFilter) {
    if (filter.values.length < 1) {
      return '';
    }

    const queryArray = filter.values.map((f) => {
      const currentFilter = f as IInfluxRelationFilter;

      if (currentFilter.property) {
        return this.initRelationFilter(currentFilter);
      } else {
        const tempFilter = f as IInfluxFilter;

        return `(${this.initFilter(tempFilter)})`;
      }
    });

    return queryArray.join(` ${filter.operator} `);
  }
  // 4.1. Filter Query 초기화 (관계 연산)
  initRelationFilter(filter: IInfluxRelationFilter) {
    return `${INFLUX_RECORD_KEYWORD}.${filter.property} ${filter.operator} "${filter.value}"`;
  }
  // 5. Aggregate Query 초기화
  initAggregate(aggregate: IInfluxAggregate, isTagDrop: boolean) {
    if (isTagDrop) {
      return ` |> drop(columns: ["Aut", "Run", "MainProgram", "SubProgram", "MCode"]) |> aggregateWindow(every: ${aggregate.interval}, fn: ${aggregate.aggregation}, createEmpty: false)`;
    }

    let strCreateEmpty = 'false';
    if (aggregate.createEmpty) {
      strCreateEmpty = 'true';
    }

    if (aggregate.dropColumns) {
      let strDropColumns = '';

      aggregate.dropColumns.forEach((c) => {
        strDropColumns += `"${c}",`;
      });
      strDropColumns = strDropColumns.substring(0, strDropColumns.length - 1);

      if (strDropColumns != '') {
        if (aggregate.interval == '1w') {
          return ` |> drop(columns: [${strDropColumns}]) |> aggregateWindow(every: ${aggregate.interval}, fn: ${aggregate.aggregation}, createEmpty: ${strCreateEmpty}, offset: -3d, location: { zone: "Asia/Seoul", offset: 0s})`;
        }

        return ` |> drop(columns: [${strDropColumns}]) |> aggregateWindow(every: ${aggregate.interval}, fn: ${aggregate.aggregation}, createEmpty: ${strCreateEmpty}, location: { zone: "Asia/Seoul", offset: 0s})`;
      }

      return ` |> aggregateWindow(every: ${aggregate.interval}, fn: ${aggregate.aggregation}, createEmpty: ${strCreateEmpty}, location: { zone: "Asia/Seoul", offset: 0s})`;
    }

    return ` |> aggregateWindow(every: ${aggregate.interval}, fn: ${aggregate.aggregation}, createEmpty: ${strCreateEmpty}, location: { zone: "Asia/Seoul", offset: 0s})`;
  }
  // 6. Selector Query 초기화 (first/last)
  initSelectorQuery(isLast: boolean) {
    // InfluxDB 스키마 변경(태그 추가)에 따른 'group' 함수 추가
    if (isLast) {
      return ' |> group(columns: ["_field"]) |> last()';
    }

    return ' |> group(columns: ["_field"]) |> first()';
  }
  // 7. Drop Column Query 초기화
  initDropQuery() {
    return ' |> drop(columns: ["_measurement", "_start", "_stop", "host", "topic", "table"])';
  }
}
