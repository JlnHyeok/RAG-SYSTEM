import { forwardRef, Inject, Injectable } from '@nestjs/common';
import { CreateProductInput } from './dto/create-product.input';
import { InjectModel } from '@nestjs/mongoose';
import { Product } from './entities/product.entity';
import { FilterQuery, Model } from 'mongoose';
import {
  ProductInfluxOutput,
  ProductLastOutput,
  ProductMutationOutput,
  ProductPaginationOutput,
  ProductSubscriptionOutput,
  ProductSumReportOutput,
} from './dto/product.output';
import {
  FilterProductCountInput,
  FilterProductInfluxInput,
  FilterProductInput,
  FilterProductSumReportInput,
} from './dto/filter-product.input';
import {
  convertInfluxFilter,
  FilterCommonInput,
} from 'src/common/dto/filter-common.input';
import { PRODUCT_ENTITY, PUB_SUB } from 'src/app.provider';
import { PubSub } from 'graphql-subscriptions';
import { TOPIC_MONITOR_PRODUCT } from 'src/pubsub/pubsub.constants';
import { AbnormalSubscriptionOutput } from 'src/abnormal/dto/abnormal.output';
import { ThresholdService } from 'src/master/threshold/threshold.service';
import { MachineService } from 'src/master/machine/machine.service';
import { FilterInfluxTagInput } from 'src/influx/dto/filter-influx.input';
import { ProductInflux } from './entities/product-influx.entity';
import { IInfluxModel } from 'src/influx/interface/influx.interface';
import { InfluxService } from 'src/influx/influx.service';
import { PeriodType } from 'src/common/dto/common.enum';
import { AbnormalService } from 'src/abnormal/abnormal.service';
import { UpdateProductInput } from './dto/update-product.input';

@Injectable()
export class ProductService {
  constructor(
    @InjectModel(Product.name)
    private readonly productModel: Model<Product>,
    @Inject(PRODUCT_ENTITY)
    private readonly productInfluxModel: IInfluxModel,
    @Inject(PUB_SUB)
    private readonly pubSub: PubSub,
    @Inject(forwardRef(() => MachineService))
    private readonly machineService: MachineService,
    @Inject(forwardRef(() => ThresholdService))
    private readonly thresholdService: ThresholdService,
    private readonly influxService: InfluxService,
    @Inject(forwardRef(() => AbnormalService))
    private readonly abnormalService: AbnormalService,
  ) {}

  async create(createProductInput: CreateProductInput) {
    // 공정 코드를 이용하여 설비 정보 조회
    const currentMachine = await this.machineService.find({
      opCode: createProductInput.opCode,
    });
    if (!currentMachine || currentMachine.length == 0) {
      return null;
    }
    // 현재 등록된 임계치 기준 정보 조회
    const currentThreshold = await this.thresholdService.findOne(
      currentMachine[0].machineCode,
    );
    if (!currentThreshold) {
      return null;
    }

    // 임계치를 이용하여 이상 유무 확인
    const isCtAbnormal =
      currentThreshold.maxThresholdCt < createProductInput.ct / 1000000000 ||
      currentThreshold.minThresholdCt > createProductInput.ct / 1000000000; // 초 단위로 변경하여 비교
    const isLoadSumAbnormal =
      currentThreshold.maxThresholdLoad < createProductInput.loadSum ||
      currentThreshold.minThresholdLoad > createProductInput.loadSum;

    // AI 이상 발생 여부 확인
    const currentAbnormals = await this.abnormalService.find({
      commonFilter: {
        workshopId: createProductInput.workshopId,
        lineId: createProductInput.lineId,
        opCode: createProductInput.opCode,
      },
      productNo: createProductInput.productId,
      abnormalCode: 'AI',
    });

    // 생산 이력 RDB 저장
    return await this.productModel.create({
      workshopCode: createProductInput.workshopId,
      lineCode: createProductInput.lineId,
      opCode: createProductInput.opCode,
      machineCode: createProductInput.machineId,
      productNo: createProductInput.productId,
      startTime: new Date(createProductInput.startTime / 1000000),
      endTime: new Date(createProductInput.endTime / 1000000),
      // TODO: 양품/불량 결과 확인 필요
      productResult: isCtAbnormal
        ? 'N'
        : isLoadSumAbnormal
          ? 'N'
          : currentAbnormals.abnormals.length > 0
            ? 'W'
            : 'Y',
      completeStatus: createProductInput.completeStatus,
      count: createProductInput.count,
      ct: createProductInput.ct,
      ctResult: isCtAbnormal ? 'N' : 'Y',
      loadSum: createProductInput.loadSum,
      loadSumResult: isLoadSumAbnormal ? 'N' : 'Y',
      ai: currentAbnormals.abnormals.length,
      aiResult: currentAbnormals.abnormals.length > 0 ? 'N' : 'Y',

      mainProgramNo: createProductInput.mainProg,
      tCode: '',
      mCode: '',
      fov: createProductInput.fov,
      sov: createProductInput.sov,
      offsetX: createProductInput.offsetX,
      offsetZ: createProductInput.offsetZ,
      feed: 0,

      // 빈 값 입력 (이후 삭제 필요)
      subProgramNo: '',
    });
  }
  async find(filterProductInput: FilterProductInput) {
    if (!filterProductInput) {
      return await this.productModel.find().sort({
        startTime: -1,
      });
    }

    // 생산 번호가 있을 경우 생산 번호로만 조회 (Unique 값)
    if (filterProductInput.productNo) {
      return await this.productModel.find({
        workshopCode: filterProductInput.commonFilter.workshopId,
        lineCode: filterProductInput.commonFilter.lineId,
        opCode: filterProductInput.commonFilter.opCode,
        machineCode: filterProductInput.commonFilter.machineId,
        productNo: filterProductInput.productNo,
      });
    }
    // 조회 데이터 수가 없을 경우 조회 일자를 이용하여 조회
    if (!filterProductInput.count) {
      return await this.productModel
        .find({
          workshopCode: filterProductInput.commonFilter.workshopId,
          lineCode: filterProductInput.commonFilter.lineId,
          opCode: filterProductInput.commonFilter.opCode,
          machineCode: filterProductInput.commonFilter.machineId,
          startTime: {
            $gte: filterProductInput.rangeStart,
            $lte: filterProductInput.rangeEnd,
          },
        })
        .sort({ startTime: -1 });
    }
    // 조회 데이터 수가 있는 경우 기간 내 최근 N건의 데이터를 조회
    return await this.productModel
      .find({
        workshopCode: filterProductInput.commonFilter.workshopId,
        lineCode: filterProductInput.commonFilter.lineId,
        opCode: filterProductInput.commonFilter.opCode,
        machineCode: filterProductInput.commonFilter.machineId,
        startTime: {
          $gt: filterProductInput.rangeStart,
          $lt: filterProductInput.rangeEnd,
        },
      })
      .sort({ startTime: -1 })
      .limit(filterProductInput.count);
  }
  async findOne(productCode: string) {
    const result = await this.productModel.findOne({
      productNo: productCode,
    });

    return result;
  }
  async findPagination(
    filterProductInput: FilterProductInput,
  ): Promise<ProductPaginationOutput> {
    // Filter가 없을 경우 전체 조회
    if (!filterProductInput) {
      const products = await this.productModel.find().sort({ startTime: -1 });
      return {
        pageCount: null,
        products,
      };
    }

    const DEFAULT_SORT_KEY = 'startTime';
    const countPerPage = filterProductInput.count
      ? filterProductInput.count
      : null;
    const skipCount =
      countPerPage && filterProductInput.page
        ? countPerPage * (filterProductInput.page - 1)
        : null;
    let productFilter: FilterQuery<Product> = {};
    const productSort = {};
    if (filterProductInput.sortInput) {
      productSort[filterProductInput.sortInput.sortColumn] =
        filterProductInput.sortInput.sortDirection == 'asc' ? 1 : -1;

      if (filterProductInput.sortInput.sortColumn != DEFAULT_SORT_KEY) {
        productSort[DEFAULT_SORT_KEY] = -1;
      }
    }

    productFilter = {
      workshopCode: filterProductInput.commonFilter.workshopId,
      lineCode: filterProductInput.commonFilter.lineId,
      opCode: filterProductInput.commonFilter.opCode,
      machineCode: filterProductInput.commonFilter.machineId,
      // productNo: filterAbnormalInput.productNo,

      startTime: {
        $gte: filterProductInput.rangeStart,
        $lte: filterProductInput.rangeEnd,
      },
      $and: this.initRegexFilter(
        filterProductInput.filterKeyword,
        filterProductInput.filterResult,
      ),
    };

    if (countPerPage && (skipCount || skipCount == 0)) {
      const countResult = await this.productModel.countDocuments(productFilter);
      const products = await this.productModel
        .find(productFilter)
        .sort(productSort)
        .skip(skipCount)
        .limit(countPerPage);

      return {
        pageCount:
          countResult % countPerPage == 0
            ? countResult / countPerPage
            : Math.floor(countResult / countPerPage + 1),
        products,
      };
    }

    const products = await this.productModel
      .find(productFilter)
      .sort(productSort);

    return {
      pageCount: null,
      products,
    };
  }
  async findLast(filterCommonInput: FilterCommonInput) {
    const lastProduct = await this.productModel
      .find({
        workshopCode: filterCommonInput.workshopId,
        lineCode: filterCommonInput.lineId,
        opCode: filterCommonInput.opCode,
        machineCode: filterCommonInput.machineId,
      })
      .sort({ startTime: -1 })
      .limit(1);

    if (lastProduct.length == 0) {
      return new ProductLastOutput();
    }

    return await this.initProductLastOutput(lastProduct[0]);
  }
  async update(
    productNo: string,
    updateProductInput: UpdateProductInput,
  ): Promise<ProductMutationOutput> {
    const updateResult = await this.productModel.findOneAndUpdate(
      {
        productNo: productNo,
      },
      {
        ai: updateProductInput.ai,
        aiResult: updateProductInput.aiResult,
        productResult: updateProductInput.productResult,
        updateAt: new Date(),
      },
      {
        returnDocument: 'after',
      },
    );

    if (updateResult) {
      return {
        isSuccess: true,
        productResult: updateResult.productResult,
        ai: updateResult.ai,
        aiResult: updateResult.aiResult,
      };
    } else {
      return {
        isSuccess: false,
      };
    }
  }

  // Influx Method
  async findInflux(
    filterProductInfluxInput: FilterProductInfluxInput,
    fields?: string[],
  ) {
    if (
      !filterProductInfluxInput.rangeStart &&
      !filterProductInfluxInput.rangeStartString
    ) {
      return [];
    }

    let outputs: ProductInfluxOutput[] = [];

    // 동적 필드 할당을 위해 필드명 취득
    const outputProps = Object.getOwnPropertyNames(new ProductInfluxOutput());
    const tagArray = convertInfluxFilter(filterProductInfluxInput.commonFilter);

    if (filterProductInfluxInput.tags) {
      const filterTagArray = filterProductInfluxInput.tags.map((t) => {
        const temp: FilterInfluxTagInput = new FilterInfluxTagInput();
        temp.tagName = t.tagName;
        temp.tagValue =
          t.tagName == 'TCode' ? t.tagValue.replaceAll('T', '') : t.tagValue;

        return temp;
      });

      tagArray.push(...filterTagArray);
    }

    // TSDB 데이터 조회
    const queryData: ProductInflux[] = await this.productInfluxModel.find(
      this.influxService,
      filterProductInfluxInput.rangeStart
        ? new Date(filterProductInfluxInput.rangeStart)
        : null,
      filterProductInfluxInput.rangeStop
        ? new Date(filterProductInfluxInput.rangeStop)
        : null,
      filterProductInfluxInput.rangeStartString,
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
      filterProductInfluxInput.aggregateInterval
        ? {
            aggregation: 'mean',
            interval: filterProductInfluxInput.aggregateInterval,
            dropColumns: ['host', 'ProductId', 'StartTime', 'EndTime'],
            createEmpty: false,
          }
        : null,
    );

    outputs = queryData.map((d) => {
      const operationInfo = this.didToOperationInfo(d.did);
      const tempData = new ProductInfluxOutput();
      tempData.time = new Date(d._time);
      tempData.WorkshopCode = operationInfo[0];
      tempData.LineCode = operationInfo[1];
      tempData.OpCode = operationInfo[2];
      tempData.MachineCode = operationInfo[3];
      // tempData.startTime = new Date(parseInt(d.StartTime) / 1000000);
      // tempData.endTime = new Date(parseInt(d.EndTime) / 1000000);
      tempData.CT = d.CT ? d.CT / 1000000000 : 0;
      tempData.LoadSum = d.LoadSum ? d.LoadSum : 0;
      tempData.Count = d.Count ? d.Count : 0;

      return tempData;
    });

    return outputs;
  }
  async aggregateSum(
    filterProductSumReportInput: FilterProductSumReportInput,
    fields?: string[],
  ) {
    if (
      !filterProductSumReportInput.rangeStart ||
      !filterProductSumReportInput.rangeStop
    ) {
      return [];
    }

    let outputs: ProductSumReportOutput[] = [];

    // 동적 필드 할당을 위해 필드명 취득
    const outputProps = Object.getOwnPropertyNames(new ProductInfluxOutput());
    const tagArray = convertInfluxFilter(
      filterProductSumReportInput.commonFilter,
    );

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
    const queryData: ProductInflux[] = await this.productInfluxModel.find(
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
        dropColumns: ['host', 'ProductId', 'StartTime', 'EndTime'],
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

      const operationInfo = this.didToOperationInfo(d.did);
      const tempData = new ProductInfluxOutput();
      // const tempData = new ProductSumReportOutput();
      tempData.time = new Date(convertDate);
      tempData.WorkshopCode = operationInfo[0];
      tempData.LineCode = operationInfo[1];
      tempData.OpCode = operationInfo[2];
      tempData.MachineCode = operationInfo[3];
      // tempData.startTime = new Date(parseInt(d.StartTime) / 1000000);
      // tempData.endTime = new Date(parseInt(d.EndTime) / 1000000);
      // tempData.CT = d.CT ? d.CT / 1000000000 : 0;
      // tempData.LoadSum = d.LoadSum ? d.LoadSum : 0;
      tempData.Count = d.Count ? d.Count : 0;

      return tempData;
    });

    return outputs;
  }

  async monitor(filterCommonInput: FilterCommonInput) {
    // Pub-Sub 토픽 등록
    const topic = `${filterCommonInput.workshopId}/${filterCommonInput.lineId}/${filterCommonInput.opCode}/${TOPIC_MONITOR_PRODUCT}`;

    return this.pubSub.asyncIterator(topic);
  }

  // 특정 일자의 생산 수량 집계
  async getProductCount(
    filterProductCountInput: FilterProductCountInput,
  ): Promise<any[]> {
    const sYear = filterProductCountInput.filterDate.getFullYear(); // 2023
    const sMonth = (filterProductCountInput.filterDate.getMonth() + 1)
      .toString()
      .padStart(2, '0'); // 06
    const sDate = filterProductCountInput.filterDate
      .getDate()
      .toString()
      .padStart(2, '0'); // 18

    // const sYear = String(filterProductCountInput.filterDate).substring(0, 4);
    // const sMonth = String(filterProductCountInput.filterDate).substring(4, 6);
    // const sDate = String(filterProductCountInput.filterDate).substring(6, 8);

    const filterBeginDate: Date = new Date(
      Number(sYear),
      Number(sMonth) - 1,
      Number(sDate),
    );
    const filterEndDate = new Date(
      filterBeginDate.getFullYear(),
      filterBeginDate.getMonth(),
      filterBeginDate.getDate() + 1,
    );

    return await this.productModel.aggregate([
      {
        $match: {
          workshopCode: filterProductCountInput.workshopId,
          lineCode: filterProductCountInput.lineId,
          opCode: filterProductCountInput.opCode,
          machineCode: filterProductCountInput.machineId,
          endTime: {
            $gte: filterBeginDate,
            $lt: filterEndDate,
          },
        },
      },
      {
        $count: 'productCount',
      },
    ]);
  }

  // 당일 생산 수량 계산 함수
  async getTodayProductCount(
    filterCommonInput: FilterCommonInput,
  ): Promise<any[]> {
    const today = new Date(Date.now());
    const filterBeginDate = new Date(
      today.getFullYear(),
      today.getMonth(),
      today.getDate(),
    );
    const filterEndDate = new Date(
      today.getFullYear(),
      today.getMonth(),
      today.getDate() + 1,
    );
    return await this.aggregateSum(
      {
        commonFilter: filterCommonInput,
        periodType: PeriodType.Weekly,
        rangeStart: filterBeginDate,
        rangeStop: filterEndDate,
      },
      ['Count'],
    );

    // return await this.getProductCount({
    //   workshopId: filterCommonInput.workshopId,
    //   opCode: filterCommonInput.opCode,
    //   lineId: filterCommonInput.lineId,
    //   machineId: filterCommonInput.machineId,
    //   filterDate: filterBeginDate,
    // });
  }

  // 평균 C/T 계산 함수 (현재 시간 기준 20분 이내)
  async getProductAvgCt(): Promise<any[]> {
    const filterEndDate = new Date(Date.now());
    const filterBeginDate = new Date(
      filterEndDate.getFullYear(),
      filterEndDate.getMonth(),
      filterEndDate.getDate(),
      filterEndDate.getHours(),
      filterEndDate.getMinutes() - 20,
      filterEndDate.getSeconds(),
    );

    return await this.productModel.aggregate([
      {
        $match: {
          startTime: {
            $gte: filterBeginDate,
            $lte: filterEndDate,
          },
        },
      },
      {
        $group: {
          _id: null,
          productAvgCt: {
            $avg: '$ct',
          },
        },
      },
    ]);
  }
  // 평균 C/T 계산 함수 (현재 시간 기준 20분 이내)
  async getProductAvgLoadSum(): Promise<any[]> {
    const filterEndDate = new Date(Date.now());
    const filterBeginDate = new Date(
      filterEndDate.getFullYear(),
      filterEndDate.getMonth(),
      filterEndDate.getDate(),
      filterEndDate.getHours(),
      filterEndDate.getMinutes() - 20,
      filterEndDate.getSeconds(),
    );

    return await this.productModel.aggregate([
      {
        $match: {
          startTime: {
            $gte: filterBeginDate,
            $lte: filterEndDate,
          },
        },
      },
      {
        $group: {
          _id: null,
          productAvgLoadSum: {
            $avg: '$loadSum',
          },
        },
      },
    ]);
  }

  // 최근 생산 이력 초기화 함수
  async initProductLastOutput(product: Product): Promise<ProductLastOutput> {
    const output = new ProductLastOutput();

    // * 당일 생산 수량 집계
    const productDailyCount = await this.getTodayProductCount({
      workshopId: product.workshopCode,
      lineId: product.lineCode,
      opCode: product.opCode,
      machineId: product.machineCode,
    });
    // * 당일 평균 C/T 집계
    const productAvgCt = await this.getProductAvgCt();
    // * 당일 생산 수량 집계
    const productLoadSum = await this.getProductAvgLoadSum();

    output.productNo = product.productNo;
    output.productBeginDate = product.startTime;
    output.productEndDate = product.endTime;
    output.productAi = product.ai;
    output.productCt = product.ct;
    output.productLoadSum = product.loadSum;
    output.productDailyCount =
      productDailyCount.length == 0 ? 0 : productDailyCount[0]['Count'];
    output.productAvgCt =
      productAvgCt.length == 0 ? 0 : productAvgCt[0]['productAvgCt'];
    output.productAvgLoadSum =
      productLoadSum.length == 0 ? 0 : productLoadSum[0]['productAvgLoadSum'];

    return output;
  }
  // 생산 이력 페이로드 초기화 함수
  async initProductPublishPayload(
    product: Product,
  ): Promise<ProductSubscriptionOutput> {
    const output = new ProductSubscriptionOutput();

    // * 당일 생산 수량 집계
    const productCount = await this.getTodayProductCount({
      workshopId: product.workshopCode,
      lineId: product.lineCode,
      opCode: product.opCode,
      machineId: product.machineCode,
    });
    // * 당일 평균 C/T 집계
    const productAvgCt = await this.getProductAvgCt();
    // * 당일 생산 수량 집계
    const productLoadSum = await this.getProductAvgLoadSum();

    output.productNo = product.productNo;
    output.productBeginDate = product.startTime;
    output.productEndDate = product.endTime;
    output.productResult = product.productResult;
    output.productCount =
      productCount.length == 0 ? 0 : productCount[0]['Count'];
    output.productCt = product.ct;
    output.productCtResult = product.ctResult;
    // TODO: 평균 LoadSum인지 확인 필요
    output.productLoadSum = product.loadSum;
    output.productLoadSumResult = product.loadSumResult;
    output.productAvgCt =
      productAvgCt.length == 0 ? 0 : productAvgCt[0]['productAvgCt'];
    output.productAvgLoadSum =
      productLoadSum.length == 0 ? 0 : productLoadSum[0]['productAvgLoadSum']; //product.loadSum;

    output.productAi = product.ai;
    output.productAiResult = product.aiResult;

    return output;
  }
  // 이상감지 이력 페이로드 초기화 함수
  async initAbnormalPayload(
    product: Product,
  ): Promise<AbnormalSubscriptionOutput> {
    const output = new AbnormalSubscriptionOutput();

    // TODO: AI 엔진 데이터 포맷 확인 필요
    output.abnormalCode = 'AI';
    // output.abnormalLevel =
    //   maxStatusIdx == 0 ? '' : maxStatusIdx > 6 ? 'BROKEN' : 'WEAR';
    output.abnormalBeginDate = product.startTime;
    output.abnormalEndDate = product.endTime;
    // output.abnormalTool =
    //   maxStatusIdx == 0
    //     ? ''
    //     : maxStatusIdx > 6
    //       ? TOOL_ORDER[maxStatusIdx - 7]
    //       : TOOL_ORDER[maxStatusIdx - 1];
    output.abnormalTool = '';
    output.abnormalValue = product.ai;

    return output;
  }

  initRegexFilter(
    keyword: string,
    resultKeyword: string,
  ): FilterQuery<Product>[] {
    const result: FilterQuery<Product>[] = [];

    result.push({
      productNo: keyword ? { $regex: `.*${keyword}.*` } : undefined,
    });
    result.push({
      productResult: resultKeyword
        ? { $regex: `.*${resultKeyword}.*` }
        : undefined,
    });

    return result;
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
