import {
  Body,
  Controller,
  Inject,
  Logger,
  LoggerService,
  Post,
} from '@nestjs/common';
import { PubSub } from 'graphql-subscriptions';
import { PUB_SUB } from 'src/app.provider';
import { AbnormalService } from './abnormal.service';
import { CreateLossAbnormalInput } from './dto/create-abnormal.input';
import { CommonMutationOutput } from 'src/common/dto/common.output';
import { AbnormalSubscriptionOutput } from './dto/abnormal.output';
import { TOPIC_MONITOR_ABNORMAL } from 'src/pubsub/pubsub.constants';
import { ProductService } from 'src/product/product.service';
import { ThresholdService } from 'src/master/threshold/threshold.service';

@Controller('abnormal-history')
export class AbnormalController {
  constructor(
    @Inject(PUB_SUB)
    private readonly pubSub: PubSub,
    private readonly abnormalService: AbnormalService,
    private readonly productService: ProductService,
    private readonly thresholdService: ThresholdService,
  ) {}

  // 1. 부하 이상 감지 이력 등록
  @Post()
  async createAbnormal(
    @Body() createLossAbnormalInput: CreateLossAbnormalInput,
  ) {
    const output = new CommonMutationOutput();

    // 비가동일 경우 이상감지 이력 저장 X
    if (createLossAbnormalInput.productId.includes('-00000', 0)) {
      output.isSuccess = true;

      return output;
    }
    // MongoDB 데이터 저장
    const newAbnormal = await this.abnormalService.createLossAbnormal(
      createLossAbnormalInput,
    );

    // 이상감지, 생산 이력 업데이트
    const currentSummary = await this.abnormalService.findSummary({
      commonFilter: {
        workshopId: newAbnormal.workshopCode,
        lineId: newAbnormal.lineCode,
        opCode: newAbnormal.opCode,
        machineId: newAbnormal.machineCode,
      },
      productNo: newAbnormal.productNo,
      page: 1,
      count: 1,
    });
    const currentProduct = await this.productService.findOne(
      newAbnormal.productNo,
    );

    // 이상감지 이력이 없을 경우 (생산 완료보다 이상감지가 늦게 들어왔을 경우)
    if (currentSummary.abnormals.length == 0) {
      Logger.error(`Insert Abnormal Summary`);

      const currentThreshold = await this.thresholdService.findOne(
        createLossAbnormalInput.machineId,
      );

      if (currentProduct) {
        const abnormalSummary = await this.abnormalService.createSummary({
          workshopId: currentProduct.workshopCode,
          lineId: currentProduct.lineCode,
          opCode: currentProduct.opCode,
          machineId: currentProduct.machineCode,
          productId: currentProduct.productNo,

          // CT: abnormalSummary에 데이터가 없기 때문에 정상으로 입력
          abnormalCt: 'Y',
          abnormalCtValue: currentProduct.ct,
          abnormalCtThreshold: currentThreshold.maxThresholdCt * 1000000000,
          abnormalMinCtThreshold: currentThreshold.minThresholdCt * 1000000000,

          // LoadSum: abnormalSummary에 데이터가 없기 때문에 정상으로 입력
          abnormalLoad: 'Y',
          abnormalLoadValue: currentProduct.loadSum,
          abnormalLoadThreshold: currentThreshold.maxThresholdLoad,
          abnormalMinLoadThreshold: currentThreshold.minThresholdLoad,

          abnormalAi: 'N',
          abnormalAiValue: 1,
          abnormalAiThreshold: currentThreshold.thresholdLoss
            ? currentThreshold.thresholdLoss
            : 0,
          abnormalAiCount: currentThreshold.predictPeriod
            ? currentThreshold.predictPeriod
            : 0,

          abnormalBeginDate: currentProduct.startTime,
          abnormalEndDate: currentProduct.endTime,

          mainProgramNo: currentProduct.mainProgramNo,
          subProgramNo: currentProduct.subProgramNo,
          mCode: currentProduct.mCode,
          tCode: currentProduct.tCode,
          sov: currentProduct.sov,
          fov: currentProduct.fov,
          offsetX: currentProduct.offsetX,
          offsetZ: currentProduct.offsetZ,
          feed: currentProduct.feed,
        });

        // CT, LoadSum은 정상이므로 경고 상태로 등록
        const updateResult = await this.productService.update(
          currentProduct.productNo,
          {
            productResult: 'W',
            ai: 1,
            aiResult: 'N',
          },
        );
      } else {
        Logger.error(`Insert Abnormal Summary: Product Not Complete`);
      }
    } else {
      Logger.error(
        `Update Abnormal Summary: ${currentSummary.abnormals[0].productNo} ${currentSummary.abnormals[0].abnormalAi} ${currentSummary.abnormals[0].abnormalAiValue}`,
      );

      // 이상감지 업데이트
      const updateResult = await this.abnormalService.updateSummary(
        currentSummary.abnormals[0].productNo,
        {
          abnormalAi: 'N',
          abnormalAiValue: currentSummary.abnormals[0].abnormalAiValue,
        },
      );
      if (updateResult.isSuccess) {
        Logger.error(
          `Update Abnormal Summary Success: ${currentSummary.abnormals[0].productNo} ${updateResult.abnormalAi} ${updateResult.abnormalAiValue}`,
        );
      } else {
        Logger.error(
          `Update Abnormal Summary Fail: ${currentSummary.abnormals[0].productNo} ${updateResult.abnormalAi} ${updateResult.abnormalAiValue}`,
        );
      }

      // 생산이력 업데이트
      if (currentProduct) {
        // CT, LoadSum은 정상이므로 경고 상태로 등록
        const productUpdateResult = await this.productService.update(
          currentProduct.productNo,
          {
            ai: currentSummary.abnormals[0].abnormalAiValue,
            aiResult: 'N',
          },
        );
        if (productUpdateResult.isSuccess) {
          Logger.error(
            `Update Product History Success: ${currentProduct.productNo} ${productUpdateResult.aiResult} ${productUpdateResult.ai}`,
          );
        } else {
          Logger.error(
            `Update Product History Fail: ${currentProduct.productNo}`,
          );
        }
      }
    }

    const abnormalPayload = new AbnormalSubscriptionOutput();
    abnormalPayload.abnormalCode = newAbnormal.abnormalCode;
    abnormalPayload.abnormalBeginDate = newAbnormal.abnormalBeginDate;
    abnormalPayload.abnormalEndDate = newAbnormal.abnormalEndDate;
    abnormalPayload.abnormalValue = newAbnormal.abnormalValue;

    const payloadObj = new Object();
    payloadObj[TOPIC_MONITOR_ABNORMAL] = abnormalPayload;

    // await this.pubSub.publish(TOPIC_MONITOR_ABNORMAL, payloadObj);
    const topic = `${createLossAbnormalInput.workshopId}/${createLossAbnormalInput.lineId}/${createLossAbnormalInput.opCode}/${TOPIC_MONITOR_ABNORMAL}`;
    await this.pubSub.publish(topic, payloadObj);

    output.isSuccess = true;

    return output;
  }
}
