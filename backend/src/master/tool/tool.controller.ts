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
import { ToolService } from 'src/master/tool/tool.service';
import { ToolInfoOutput } from './dto/info-tool.output';
import { FilterToolInput } from './dto/filter-tool.input';
import { FilterCommonInput } from 'src/common/dto/filter-common.input';

@Controller('tool')
export class ToolController {
  constructor(
    @Inject(forwardRef(() => ToolService))
    private readonly toolService: ToolService,
  ) {}

  // * GET
  // 1. 공구별 최근 사용량 조회
  @Get()
  async getToolInfo(@Query() filterCommonInput: FilterCommonInput) {
    const result: ToolInfoOutput[] = await this.toolService.find({
      machineCode: filterCommonInput.machineId,
    });

    return result.map((t) => {
      return {
        code: t.toolCode,
        order: t.toolOrder,
        maxCount: t.maxCount,
        warnRate: t.warnRate,
      };
    });
  }
}
