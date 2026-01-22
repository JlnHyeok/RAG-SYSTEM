import { Controller, forwardRef, Get, Inject, Query } from '@nestjs/common';
import { FilterCommonInput } from 'src/common/dto/filter-common.input';
import { MachineService } from '../machine/machine.service';
import { ThresholdService } from './threshold.service';
import { ThresholdPredictOutput } from './dto/threshold.output';

@Controller('threshold')
export class ThresholdController {
  constructor(
    private readonly thresholdService: ThresholdService,
    @Inject(forwardRef(() => MachineService))
    private readonly machineService: MachineService,
  ) {}

  // * GET
  // 1. 공구별 최근 사용량 조회
  @Get('load-predict')
  async getPredictThreshold(
    @Query() filterCommonInput: FilterCommonInput,
  ): Promise<ThresholdPredictOutput> {
    // 공정 코드를 이용하여 설비 정보 조회
    const currentMachine = await this.machineService.find({
      opCode: filterCommonInput.opCode,
    });
    const result = await this.thresholdService.findOne(
      currentMachine[0].machineCode,
    );

    if (result) {
      return {
        tool1Threshold: result.tool1Threshold,
        tool2Threshold: result.tool2Threshold,
        tool3Threshold: result.tool3Threshold,
        tool4Threshold: result.tool4Threshold,
      };
    }

    return {
      tool1Threshold: 0,
      tool2Threshold: 0,
      tool3Threshold: 0,
      tool4Threshold: 0,
    };
  }
}
