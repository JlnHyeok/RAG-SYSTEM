import { InfluxService } from '../influx.service';

export interface IInfluxModel {
  find(
    influxService: InfluxService,
    rangeStart?: Date,
    rangeEnd?: Date,
    rangeStartString?: string,
    tags?: IInfluxFilter,
    fields?: IInfluxFilter,
    aggregateInterval?: IInfluxAggregate,
  ): Promise<any[]>;

  findOne(influxService: InfluxService, date: Date);

  findLast(influxService: InfluxService, tags?: IInfluxFilter);
}

export interface IInfluxAggregate {
  aggregation: 'min' | 'max' | 'mean' | 'sum' | 'count';
  interval: string;
  dropColumns?: string[];
  createEmpty?: boolean;
}

export interface IInfluxFilter {
  operator: 'and' | 'or';
  values: (IInfluxFilter | IInfluxRelationFilter)[];
}

export interface IInfluxRelationFilter {
  property: string;
  operator: '==' | '!=';
  value: string;
}
