import {
  Body,
  Controller,
  forwardRef,
  Get,
  Inject,
  Post,
  Query,
} from '@nestjs/common';
import { PubSub } from 'graphql-subscriptions';
import { PUB_SUB } from 'src/app.provider';
import { ProductService } from './product.service';
import { FilterProductCountInput } from './dto/filter-product.input';
import { ProductCountOutput } from './dto/product.output';
import { CreateProductInput } from './dto/create-product.input';
import { CommonMutationOutput } from 'src/common/dto/common.output';
import { TOPIC_MONITOR_PRODUCT } from 'src/pubsub/pubsub.constants';
import { AbnormalService } from 'src/abnormal/abnormal.service';
import { ThresholdService } from 'src/master/threshold/threshold.service';
import { AbnormalMinMax } from 'src/common/dto/common.enum';

@Controller('product-history')
export class ProductController {
  constructor(
    @Inject(PUB_SUB)
    private readonly pubSub: PubSub,
    private readonly productService: ProductService,
    private readonly abnormalService: AbnormalService,
    @Inject(forwardRef(() => ThresholdService))
    private readonly thresholdService: ThresholdService,
  ) {}

  // 1. 생산 수량 조회
  @Get('count')
  async getDailyProductCount(
    @Query() filterProductCountInput: FilterProductCountInput,
  ): Promise<ProductCountOutput> {
    const sYear = String(filterProductCountInput.filterDate).substring(0, 4);
    const sMonth = String(filterProductCountInput.filterDate).substring(4, 6);
    const sDate = String(filterProductCountInput.filterDate).substring(6, 8);
    const filterBeginDate: Date = new Date(
      Number(sYear),
      Number(sMonth) - 1,
      Number(sDate),
    );
    filterProductCountInput.filterDate = filterBeginDate;

    const productCountResult = await this.productService.getProductCount(
      filterProductCountInput,
    );
    const productCount =
      productCountResult.length == 0
        ? 0
        : productCountResult[0]['productCount'];

    return {
      productCount,
    };
  }

  // 2. 생산 결과 등록
  @Post()
  async createProduct(@Body() createProductInput: CreateProductInput) {
    const output = new CommonMutationOutput();

    // 생산 결과 데이터 저장
    const newProduct = await this.productService.create(createProductInput);

    // 생산 이력 데이터 초기화
    const productOutput =
      await this.productService.initProductPublishPayload(newProduct);

    const currentThreshold = await this.thresholdService.findOne(
      createProductInput.machineId,
    );

    const currentAbnormals = await this.abnormalService.find({
      commonFilter: {
        workshopId: createProductInput.workshopId,
        lineId: createProductInput.lineId,
        opCode: createProductInput.opCode,
      },
      productNo: createProductInput.productId,
      abnormalCode: 'AI',
    });

    // 이상 감지가 되었을 때 이상 데이터 저장
    if (productOutput.productResult == 'N') {
      if (newProduct.ctResult == 'N') {
        await this.abnormalService.create({
          workshopId: newProduct.workshopCode,
          lineId: newProduct.lineCode,
          opCode: newProduct.opCode,
          machineId: newProduct.machineCode,
          productId: newProduct.productNo,
          abnormalCode: 'CT',
          //TODO: 하한 값 생성 시 비교 로직 추가 필요
          abnormalDivision:
            currentThreshold.minThresholdCt > createProductInput.ct / 1000000000
              ? AbnormalMinMax.Min
              : AbnormalMinMax.Max,
          abnormalBeginDate: newProduct.startTime,
          abnormalEndDate: newProduct.endTime,
          abnormalValue: newProduct.ct,

          mainProgramNo: newProduct.mainProgramNo,
          subProgramNo: newProduct.subProgramNo,
          mCode: newProduct.mCode,
          tCode: newProduct.tCode,
          sov: newProduct.sov,
          fov: newProduct.fov,
          offsetX: newProduct.offsetX,
          offsetZ: newProduct.offsetZ,
          feed: newProduct.feed,
        });
      }

      if (newProduct.loadSumResult == 'N') {
        await this.abnormalService.create({
          workshopId: newProduct.workshopCode,
          lineId: newProduct.lineCode,
          opCode: newProduct.opCode,
          machineId: newProduct.machineCode,
          productId: newProduct.productNo,
          abnormalCode: 'LOAD',
          //TODO: 하한 값 생성 시 비교 로직 추가 필요
          abnormalDivision:
            currentThreshold.minThresholdLoad > createProductInput.loadSum
              ? AbnormalMinMax.Min
              : AbnormalMinMax.Max,
          abnormalBeginDate: newProduct.startTime,
          abnormalEndDate: newProduct.endTime,
          abnormalValue: newProduct.loadSum,

          mainProgramNo: newProduct.mainProgramNo,
          subProgramNo: newProduct.subProgramNo,
          mCode: newProduct.mCode,
          tCode: newProduct.tCode,
          sov: newProduct.sov,
          fov: newProduct.fov,
          offsetX: newProduct.offsetX,
          offsetZ: newProduct.offsetZ,
          feed: newProduct.feed,
        });
      }
    }

    if (
      productOutput.productResult == 'N' ||
      productOutput.productResult == 'W'
    ) {
      const abnormalSummary = await this.abnormalService.createSummary({
        workshopId: newProduct.workshopCode,
        lineId: newProduct.lineCode,
        opCode: newProduct.opCode,
        machineId: newProduct.machineCode,
        productId: newProduct.productNo,

        abnormalCt: newProduct.ctResult,
        abnormalCtValue: newProduct.ct,
        abnormalCtThreshold: currentThreshold.maxThresholdCt * 1000000000,
        abnormalMinCtThreshold: currentThreshold.minThresholdCt * 1000000000,

        abnormalLoad: newProduct.loadSumResult,
        abnormalLoadValue: newProduct.loadSum,
        abnormalLoadThreshold: currentThreshold.maxThresholdLoad,
        abnormalMinLoadThreshold: currentThreshold.minThresholdLoad,

        abnormalAi: currentAbnormals.abnormals.length == 0 ? 'Y' : 'N',
        abnormalAiValue: currentAbnormals.abnormals.length,
        abnormalAiThreshold: currentThreshold.thresholdLoss
          ? currentThreshold.thresholdLoss
          : 0,
        abnormalAiCount: currentThreshold.predictPeriod
          ? currentThreshold.predictPeriod
          : 0,

        abnormalBeginDate: newProduct.startTime,
        abnormalEndDate: newProduct.endTime,

        mainProgramNo: newProduct.mainProgramNo,
        subProgramNo: newProduct.subProgramNo,
        mCode: newProduct.mCode,
        tCode: newProduct.tCode,
        sov: newProduct.sov,
        fov: newProduct.fov,
        offsetX: newProduct.offsetX,
        offsetZ: newProduct.offsetZ,
        feed: newProduct.feed,
      });
    }

    const productPayloadObj = new Object();
    const topic = `${createProductInput.workshopId}/${createProductInput.lineId}/${createProductInput.opCode}/${TOPIC_MONITOR_PRODUCT}`;
    productPayloadObj[TOPIC_MONITOR_PRODUCT] = productOutput;

    await this.pubSub.publish(topic, productPayloadObj);

    output.isSuccess = true;

    return output;
  }
}
