import { InputType, Field } from '@nestjs/graphql';
import { IsOptional, IsString } from 'class-validator';
import { FilterInfluxTagInput } from 'src/influx/dto/filter-influx.input';

@InputType()
export class FilterCommonInput {
  @Field(() => String, { description: '공장 코드' })
  @IsString()
  workshopId: string;

  @Field(() => String, { description: '라인 코드' })
  @IsString()
  lineId: string;

  @Field(() => String, { description: '공정 코드' })
  @IsString()
  opCode: string;

  @Field(() => String, {
    name: 'machineId',
    description: '설비 코드',
    nullable: true,
  })
  @IsString()
  @IsOptional()
  machineId?: string;
}

@InputType()
export class SortCommonInput {
  @Field(() => String, { description: '정렬 대상 항목' })
  sortColumn: string;

  @Field(() => String, { description: '정렬 방향' })
  sortDirection: string;
}

export function convertInfluxFilter(origin: FilterCommonInput) {
  // 2024.11.07 TSDB 조회 성능 이슈로 인한 태그 포맷 변경
  const didFilter = new FilterInfluxTagInput();
  didFilter.tagName = 'did';
  didFilter.tagValue = `${origin.workshopId}_${origin.lineId}_${origin.opCode}_${origin.machineId}`;

  return [didFilter];

  // const workshopFilter = new FilterInfluxTagInput();
  // const lineFilter = new FilterInfluxTagInput();
  // const operationFilter = new FilterInfluxTagInput();
  // const machineFilter = new FilterInfluxTagInput();

  // workshopFilter.tagName = 'WorkshopId';
  // workshopFilter.tagValue = origin.workshopId;
  // lineFilter.tagName = 'LineId';
  // lineFilter.tagValue = origin.lineId;
  // operationFilter.tagName = 'OpCode';
  // operationFilter.tagValue = origin.opCode;

  // return [workshopFilter, lineFilter, operationFilter];
  // // return [machineFilter];
}
