import { Inject, Injectable, Logger, LoggerService } from '@nestjs/common';
import { InfluxService } from 'src/influx/influx.service';
import { RAW_ENTITY } from 'src/app.provider';
import { IInfluxModel } from 'src/influx/interface/influx.interface';
import {
  FilterOperateReportInput,
  FilterRawInput,
  FilterRawTCodeInput,
} from './dto/filter-raw.input';
import {
  RawOperationPeriodReportOutput,
  RawOperationReportOutput,
  RawOutput,
  RawTCodeOutput,
  RawTCodeSingleOutput,
} from './dto/raw.output';
import {
  convertInfluxFilter,
  FilterCommonInput,
} from 'src/common/dto/filter-common.input';
import { Raw } from './entities/raw.entity';
import { ProductService } from 'src/product/product.service';
import { FilterInfluxTagInput } from 'src/influx/dto/filter-influx.input';
import { FilterProductSumReportInput } from 'src/product/dto/filter-product.input';
import { PeriodType } from 'src/common/dto/common.enum';

const RUN_TAG_NAME = 'Run';

@Injectable()
export class RawService {
  constructor(
    private readonly influxService: InfluxService,
    @Inject(RAW_ENTITY)
    private readonly rawModel: IInfluxModel,
    @Inject(Logger)
    private readonly logger: LoggerService,
    private readonly productService: ProductService,
  ) {}

  // * Query Method
  // 1. Raw 데이터 조회
  // TODO: 데이터 조회 시 Join하는 방안이 나을지 검토 필요
  // TODO: CNC 기준 정보 사용 유무 확인 필요
  async find(filterRawInput: FilterRawInput, fields?: string[]) {
    if (!filterRawInput.rangeStart && !filterRawInput.rangeStartString) {
      return [];
    }

    let outputs: RawOutput[] = [];

    // 동적 필드 할당을 위해 필드명 취득
    const outputProps = Object.getOwnPropertyNames(new RawOutput());
    const tagArray = convertInfluxFilter(filterRawInput.commonFilter);

    if (filterRawInput.tags) {
      const filterTagArray = filterRawInput.tags.map((t) => {
        const temp: FilterInfluxTagInput = new FilterInfluxTagInput();
        temp.tagName = t.tagName;
        temp.tagValue = t.tagValue;

        return temp;
      });

      tagArray.push(...filterTagArray);
    }

    // TSDB 데이터 조회
    const queryData: Raw[] = await this.rawModel.find(
      this.influxService,
      filterRawInput.rangeStart ? new Date(filterRawInput.rangeStart) : null,
      filterRawInput.rangeStop ? new Date(filterRawInput.rangeStop) : null,
      filterRawInput.rangeStartString,
      tagArray
        ? {
            operator: 'and',
            values: tagArray.map((t) => {
              return t.getInfluxFilter();
            }),
          }
        : null,
      fields
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
      filterRawInput.aggregateInterval
        ? {
            aggregation: 'mean',
            interval: filterRawInput.aggregateInterval,
            dropColumns:
              tagArray.findIndex((t) => t.tagName == 'TCode') >= 0
                ? ['host', 'ProductId']
                : ['host', 'TCode', 'ProductId'],
            createEmpty:
              tagArray.findIndex((t) => t.tagName == 'TCode') >= 0
                ? false
                : true,
          }
        : null,
    );

    outputs = queryData.map((d, index) => {
      const tempData = new RawOutput();
      const operationInfo = this.didToOperationInfo(d.did);

      tempData.Idx = index;
      tempData.time = new Date(d._time);

      tempData.WorkshopCode = operationInfo[0];
      tempData.LineCode = operationInfo[1];
      tempData.OpCode = operationInfo[2];
      tempData.MachineCode = operationInfo[3];
      tempData.ProductId = d.ProductId ? d.ProductId : '';

      tempData.Run = !d.Run ? '0' : d.Run != 0 ? '3' : '0';
      tempData.MainProgram = d.MainProgram ? `${d.MainProgram}` : '0';
      tempData.SubProgram = d.SubProgram ? `${d.SubProgram}` : '0';
      tempData.MCode = d.MCode ? `${d.MCode}` : '0';
      tempData.TCode = d.TCode ? d.TCode : '0';

      tempData.Feed = d.Feed ? d.Feed : 0;
      tempData.Fov = d.Fov ? d.Fov : 0;
      tempData.Sov = d.Sov ? d.Sov : 0;
      tempData.SV_X_Offset = d.SV_X_Offset ? d.SV_X_Offset : 0;
      tempData.SV_Z_Offset = d.SV_Z_Offset ? d.SV_Z_Offset : 0;
      tempData.SV_X_Pos = d.SV_X_Pos ? d.SV_X_Pos : 0;
      tempData.SV_Z_Pos = d.SV_Z_Pos ? d.SV_Z_Pos : 0;
      tempData.TCount1 = d.TCount1 ? d.TCount1 : 0;
      tempData.TCount2 = d.TCount2 ? d.TCount2 : 0;
      tempData.TCount3 = d.TCount3 ? d.TCount3 : 0;
      tempData.TCount4 = d.TCount4 ? d.TCount4 : 0;

      tempData.Load = d.Load ? d.Load : 0;
      tempData.Loss = d.Loss ? d.Loss : 0;
      tempData.Predict = d.Predict ? d.Predict : 0;

      return tempData;
    });

    return outputs;
  }

  async findTCodeRange(filterRawTCodeInput: FilterRawTCodeInput) {
    const output = new RawTCodeOutput();
    output.TCode = filterRawTCodeInput.TCode;
    output.TCodeRange = [];

    if (filterRawTCodeInput.productNo) {
      // 제품 정보가 있을 경우 입력 받은 제품 정보 배열을 이용하여 제품의 생산/종료 일시 취득
      for (const p of filterRawTCodeInput.productNo) {
        const singleOutput = new RawTCodeSingleOutput();
        singleOutput.productNo = p;
        singleOutput.beginIdx = null;
        singleOutput.endIdx = null;

        const currentProduct = await this.productService.findOne(p);
        if (!currentProduct) {
          output.TCodeRange.push(singleOutput);

          // 다음 반복문 수행
          continue;
        }

        const currentRaws = await this.find(
          {
            commonFilter: filterRawTCodeInput.commonFilter,
            rangeStart: currentProduct.startTime,
            rangeStop: currentProduct.endTime,
            tags: [],
          },
          ['Load'],
        );
        if (!currentRaws || currentRaws.length == 0) {
          output.TCodeRange.push(singleOutput);

          // 다음 반복문 수행
          continue;
        }

        const currentTCodeRaws = currentRaws.filter(
          (r) => r.TCode == filterRawTCodeInput.TCode.replaceAll('T', ''),
        );
        if (!currentTCodeRaws || currentTCodeRaws.length == 0) {
          output.TCodeRange.push(singleOutput);

          // 다음 반복문 수행
          continue;
        }

        singleOutput.beginIdx = currentTCodeRaws[0].Idx;
        singleOutput.beginTime = currentTCodeRaws[0].time;
        singleOutput.endIdx = currentTCodeRaws[currentTCodeRaws.length - 1].Idx;
        singleOutput.endTime =
          currentTCodeRaws[currentTCodeRaws.length - 1].time;

        output.TCodeRange.push(singleOutput);
      }
    } else if (filterRawTCodeInput.rangeStartString) {
      // 제품 정보가 없을 경우 입력 받은 시간 문자열을 이용
      const interval = this.periodStringToMs(
        filterRawTCodeInput.aggregateInterval,
      );

      const filterTag = new FilterInfluxTagInput();
      filterTag.tagName = 'TCode';
      filterTag.tagValue = filterRawTCodeInput.TCode;

      const currentRaws = await this.find(
        {
          commonFilter: filterRawTCodeInput.commonFilter,
          rangeStartString: filterRawTCodeInput.rangeStartString,
          aggregateInterval: filterRawTCodeInput.aggregateInterval,
          tags: [filterTag],
        },
        ['Load'],
      );
      if (!currentRaws || currentRaws.length == 0) {
        return output;
      }

      //
      currentRaws.sort((a, b) => {
        if (a.TCode > b.TCode) {
          return 1;
        } else if (a.TCode == b.TCode) {
          if (a.time > b.time) {
            return 1;
          }

          return -1;
        }

        return -1;
      });

      // currentRaws.forEach((a, idx) => {
      //   console.log(a.time, a.TCode);
      // });

      const convertRaws: RawOutput[] = [];
      currentRaws.reduce((acc, curr) => {
        const { TCode } = curr;

        if (acc[TCode]) {
          const prev = acc[TCode][acc[TCode].length - 1];

          if (curr.time.getTime() - prev.time.getTime() <= interval) {
            acc[TCode].push({ ...curr, Idx: prev.Idx });
            convertRaws.push({ ...curr, Idx: prev.Idx });
          } else {
            acc[TCode].push({ ...curr, Idx: prev.Idx + 1 });
            convertRaws.push({ ...curr, Idx: prev.Idx + 1 });
          }
        } else {
          acc[TCode] = [{ ...curr, Idx: 0 }];
          convertRaws.push({ ...curr, Idx: 0 });
        }

        return acc;
      });

      convertRaws.map((c, idx, raws) => {
        if (idx == 0) {
          const temp = new RawTCodeSingleOutput();
          temp.beginIdx = c.Idx;
          temp.beginTime = c.time;
          temp.endIdx = c.Idx;
          temp.endTime = c.time;
          temp.productNo = '';

          output.TCodeRange.push(temp);
        } else {
          if (c.Idx != raws[idx - 1].Idx) {
            const temp = new RawTCodeSingleOutput();
            temp.beginIdx = c.Idx;
            temp.beginTime = c.time;
            temp.endIdx = c.Idx;
            temp.endTime = c.time;
            temp.productNo = '';

            output.TCodeRange.push(temp);
          } else {
            output.TCodeRange[output.TCodeRange.length - 1].endIdx = c.Idx;
            output.TCodeRange[output.TCodeRange.length - 1].endTime = c.time;
          }
        }
      });
    }

    return output;
  }

  // 2. Raw 데이터 검색 (시간 검색)
  // TODO: 데이터 조회 시 Join하는 방안이 나을지 검토 필요
  async findOne(filterTime: Date) {
    const output = new RawOutput();
    const outputProps = Object.getOwnPropertyNames(output);

    const queryData: Raw[] = await this.rawModel.findOne(
      this.influxService,
      new Date(filterTime),
    );

    outputProps.forEach((p) => {
      const tempArray = queryData.filter((value) => value['_field'] == p);

      tempArray.forEach((t) => {
        if (output.WorkshopCode == '') {
          const operationInfo = this.didToOperationInfo(t.did);

          output.Idx = 0;
          output.time = new Date(t._time);
          output.WorkshopCode = operationInfo[0];
          output.LineCode = operationInfo[1];
          output.OpCode = operationInfo[2];
          output.MachineCode = operationInfo[3];
          output.ProductId = t.ProductId;
          output.TCode = t.TCode;
        }

        output[p] = t._value;
      });
    });

    return output;
  }

  // 3. 최근 Raw 데이터 조회
  // TODO: 데이터 조회 시 Join하는 방안이 나을지 검토 필요
  async findLast(filterCommonInput: FilterCommonInput) {
    const output = new RawOutput();
    const outputProps = Object.getOwnPropertyNames(output);
    const tagArray = convertInfluxFilter(filterCommonInput);

    const queryData: Raw[] = await this.rawModel.findLast(
      this.influxService,
      tagArray
        ? {
            operator: 'and',
            values: tagArray.map((t) => {
              return t.getInfluxFilter();
            }),
          }
        : null,
    );

    outputProps.forEach((p) => {
      const tempArray = queryData
        .filter((value) => value['_field'] == p)
        .sort((a, b) => (a._time > b._time ? -1 : 1));

      if (tempArray.length > 0) {
        if (output.WorkshopCode == '') {
          const operationInfo = this.didToOperationInfo(tempArray[0].did);

          output.Idx = 0;
          output.time = new Date(tempArray[0]._time);
          output.WorkshopCode = operationInfo[0];
          output.LineCode = operationInfo[1];
          output.OpCode = operationInfo[2];
          output.MachineCode = operationInfo[3];
          output.ProductId = tempArray[0].ProductId;
          output.TCode = tempArray[0].TCode;
        }

        output[p] = tempArray[0]._value;
      }
    });

    return output;
  }

  // 4. 가동 시간 집계
  // TODO: 데이터 조회 시 Join하는 방안이 나을지 검토 필요
  // TODO: CNC 기준 정보 사용 유무 확인 필요
  async aggregateOperation(filterOperateReportInput: FilterOperateReportInput) {
    const tagArray = convertInfluxFilter(filterOperateReportInput.commonFilter);

    // TSDB 데이터 조회
    let queryData: Raw[] = await this.rawModel.find(
      this.influxService,
      filterOperateReportInput.rangeStart,
      filterOperateReportInput.rangeStop,
      null,
      {
        operator: 'and',
        values: [
          ...tagArray.map((t) => {
            return t.getInfluxFilter();
          }),
          // {
          //   property: RUN_TAG_NAME,
          //   operator: '==',
          //   value: filterOperateReportInput.status,
          // },
          {
            property: '_field',
            operator: '==',
            value: filterOperateReportInput.reportField,
          },
        ],
      },
      null,
      filterOperateReportInput.aggregateInterval
        ? {
            aggregation: 'sum',
            interval: filterOperateReportInput.aggregateInterval,
            dropColumns: [
              // 'Aut',
              // 'Run',
              // 'MainProgram',
              // 'SubProgram',
              'TCode',
              // 'MCode',
              'ProductId',
              'host',
            ],
          }
        : null,
    );

    // Null 데이터 삭제
    queryData = queryData.filter((d) => d.Run != null);

    // 반복문을 통해 Output 객체 초기화
    const outputs = queryData.map((t) => {
      const tempData = new RawOperationReportOutput();
      const operationInfo = this.didToOperationInfo(t.did);

      tempData.reportDate = new Date(t._time);
      tempData.WorkshopCode = operationInfo[0];
      tempData.LineCode = operationInfo[1];
      tempData.OpCode = operationInfo[2];
      tempData.MachineCode = operationInfo[3];

      // tempData.Run = t.Run;
      // tempData.MainProgram = t.MainProgram;
      // tempData.SubProgram = t.SubProgram;
      // tempData.MCode = t.MCode;
      // tempData.TCode = t.TCode;

      // tempData.operationTime = t.Run ? t.Run * 200 : 0;
      tempData.operationTime = t.Run ? t.Run : 0;

      return tempData;
    });

    return outputs;
  }

  // 5. 가동 시간 통계
  async aggregateOperationSum(
    filterProductSumReportInput: FilterProductSumReportInput,
    fields?: string[],
  ) {
    if (
      !filterProductSumReportInput.rangeStart ||
      !filterProductSumReportInput.rangeStop
    ) {
      return [];
    }

    let outputs: RawOperationPeriodReportOutput[] = [];

    // 동적 필드 할당을 위해 필드명 취득
    const outputProps = Object.getOwnPropertyNames(new Raw());
    const tagArray = convertInfluxFilter(
      filterProductSumReportInput.commonFilter,
    );

    // if (tagArray) {
    //   const statusFilter = new FilterInfluxTagInput();
    //   statusFilter.tagName = RUN_TAG_NAME;
    //   statusFilter.tagValue = '3';

    //   tagArray.push(statusFilter);
    // }

    let currentAggInterval = '';
    if (filterProductSumReportInput.periodType == PeriodType.Yearly) {
      currentAggInterval = '1mo';
    } else if (filterProductSumReportInput.periodType == PeriodType.Monthly) {
      currentAggInterval = '1w';
    } else if (filterProductSumReportInput.periodType == PeriodType.Weekly) {
      currentAggInterval = '1d';
    } else {
      currentAggInterval = '1h';
    }

    // TSDB 데이터 조회
    const queryData: Raw[] = await this.rawModel.find(
      this.influxService,
      new Date(filterProductSumReportInput.rangeStart),
      new Date(filterProductSumReportInput.rangeStop),
      null,
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
      {
        aggregation: 'sum',
        interval: currentAggInterval,
        dropColumns: [
          // 'Aut',
          // 'Run',
          // 'MainProgram',
          // 'SubProgram',
          'TCode',
          // 'MCode',
          'ProductId',
          // 'PredictFlag',
          'host',
        ],
        createEmpty: true,
      },
    );

    outputs = queryData.map((d, idx) => {
      const currentDate = new Date(d._time);

      const convertDate =
        currentAggInterval != '1w'
          ? new Date(currentDate)
          : // : new Date(currentDate);
            idx == 0 && currentDate.getDate() <= 7
            ? new Date(new Date(currentDate).setDate(1))
            : idx == queryData.length - 1 && currentDate.getDay() != 1
              ? currentDate.getDay() == 0
                ? new Date(
                    new Date(currentDate).setDate(currentDate.getDate() - 6),
                  )
                : new Date(
                    new Date(currentDate).setDate(
                      currentDate.getDate() - (currentDate.getDay() - 1),
                    ),
                  )
              : new Date(
                  new Date(currentDate).setDate(
                    currentDate.getDate() - (7 - currentDate.getDay() + 1),
                  ),
                );

      const tempData = new RawOperationPeriodReportOutput();
      const operationInfo = this.didToOperationInfo(d.did);

      // tempData.time = new Date(d._time);
      tempData.time = new Date(convertDate);
      tempData.WorkshopCode = operationInfo[0];
      tempData.LineCode = operationInfo[1];
      tempData.OpCode = operationInfo[2];
      tempData.MachineCode = operationInfo[3];

      // tempData.Run = d.Run ? d.Run * 200 : 0;
      tempData.Run = d.Run ? d.Run : 0;

      return tempData;
    });

    return outputs;
  }

  private periodStringToMs(period: string) {
    const periodChar = period.substring(period.length - 1, period.length);
    const periodNum = parseInt(period.replaceAll(periodChar, ''));
    let periodValue = 0;

    switch (periodChar) {
      case 'ms':
        periodValue = 1;
        break;
      case 's':
        periodValue = 1000;
        break;
      case 'm':
        periodValue = 60 * 1000;
        break;
      case 'h':
        periodValue = 60 * 60 * 1000;
        break;
      default:
        periodValue = 1;
        break;
    }

    return periodNum * periodValue;
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
