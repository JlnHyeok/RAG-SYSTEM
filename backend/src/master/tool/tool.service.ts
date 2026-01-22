import { Injectable } from '@nestjs/common';
import { CreateToolInput } from './dto/create-tool.input';
import { UpdateToolInput } from './dto/update-tool.input';
import { FilterToolCodeInput, FilterToolInput } from './dto/filter-tool.input';
import { Tool } from './entities/tool.entity';
import { InjectModel } from '@nestjs/mongoose';
import { Model } from 'mongoose';
import { ToolMutationOutput } from './dto/tool.output';
import { ErrorCode, GetErrorMsg } from 'src/common/dto/common.enum';

const DEFAULT_TOOL: CreateToolInput[] = [
  {
    machineCode: '',
    toolCode: '505',
    toolName: 'CNMG',
    toolOrder: 1,
    maxCount: 80,
    warnRate: 90,
  },
  {
    machineCode: '',
    toolCode: '303',
    toolName: 'TNMG',
    toolOrder: 2,
    maxCount: 80,
    warnRate: 90,
  },
  {
    machineCode: '',
    toolCode: '707',
    toolName: 'TNMG(M)',
    toolOrder: 3,
    maxCount: 300,
    warnRate: 90,
  },
  {
    machineCode: '',
    toolCode: '101',
    toolName: 'DNMG(F)',
    toolOrder: 4,
    maxCount: 120,
    warnRate: 80,
  },
];

@Injectable()
export class ToolService {
  constructor(
    @InjectModel(Tool.name)
    private readonly toolModel: Model<Tool>,
  ) {}

  async create(createToolInput: CreateToolInput) {
    const output = new ToolMutationOutput();
    const currentTool = await this.findOne({
      machineCode: createToolInput.machineCode,
      toolCode: createToolInput.toolCode,
    });

    if (currentTool) {
      output.isSuccess = false;
      output.errorCode = ErrorCode.DUPLICATED;
      output.errorMsg = GetErrorMsg(ErrorCode.DUPLICATED);
      return output;
    }

    // Sub ToolCode 초기화
    const toolCodeNum = parseInt(createToolInput.toolCode.replace('T', ''));
    const subToolCodeNum = Math.floor(toolCodeNum / 100) * 100;

    const newTool = await this.toolModel.create({
      machineCode: createToolInput.machineCode,
      toolCode: createToolInput.toolCode,
      subToolCode: `T${subToolCodeNum}`,
      toolName: createToolInput.toolName,
      toolOrder: createToolInput.toolOrder,
      maxCount: createToolInput.maxCount,
      warnRate: createToolInput.warnRate,
      createAt: Date.now(),
      updateAt: Date.now(),
    });

    if (newTool) {
      output.isSuccess = true;
      output.machineCode = newTool.machineCode;
      output.toolCode = newTool.toolCode;
      output.toolName = newTool.toolName;
      output.toolOrder = newTool.toolOrder;
      output.maxCount = newTool.maxCount;
      output.warnRate = newTool.warnRate;
      output.createAt = newTool.createAt;
      output.updateAt = newTool.updateAt;

      return output;
    }

    output.isSuccess = false;
    output.errorCode = ErrorCode.ETC;
    output.errorMsg = GetErrorMsg(ErrorCode.ETC);

    return output;
  }

  async find(filterToolInput: FilterToolInput) {
    if (filterToolInput) {
      return this.toolModel
        .find({
          machineCode: filterToolInput.machineCode,
        })
        .sort({
          toolOrder: 1,
          toolCode: 1,
        });
    }

    return this.toolModel.find().sort({
      toolOrder: 1,
      toolCode: 1,
    });
  }

  async findOne(filterToolCodeInput: FilterToolCodeInput) {
    return await this.toolModel.findOne({
      machineCode: filterToolCodeInput.machineCode,
      toolCode: filterToolCodeInput.toolCode,
    });
  }

  async update(
    filterToolCodeInput: FilterToolCodeInput,
    updateToolInput: UpdateToolInput,
  ): Promise<ToolMutationOutput> {
    const updateResult = await this.toolModel.findOneAndUpdate(
      {
        machineCode: filterToolCodeInput.machineCode,
        toolCode: filterToolCodeInput.toolCode,
      },
      {
        ...updateToolInput,
        updateAt: new Date(),
      },
      {
        returnDocument: 'after',
      },
    );

    if (updateResult) {
      return {
        isSuccess: true,
        machineCode: updateResult.machineCode,
        toolCode: updateResult.toolCode,
        toolName: updateResult.toolName,
        toolOrder: updateResult.toolOrder,
        maxCount: updateResult.maxCount,
        warnRate: updateResult.warnRate,
        createAt: updateResult.createAt,
        updateAt: updateResult.updateAt,
      };
    }

    return {
      isSuccess: false,
      errorCode: ErrorCode.ETC,
      errorMsg: GetErrorMsg(ErrorCode.ETC),
    };
  }

  async delete(filterToolCodeInput: FilterToolCodeInput) {
    const deleteResult = await this.toolModel.findOneAndDelete(
      {
        ...filterToolCodeInput,
      },
      {
        returnDocument: 'before',
      },
    );

    if (deleteResult) {
      return {
        isSuccess: true,
        machineCode: deleteResult.machineCode,
        toolCode: deleteResult.toolCode,
        toolName: deleteResult.toolName,
        toolOrder: deleteResult.toolOrder,
        maxCount: deleteResult.maxCount,
        warnRate: deleteResult.warnRate,
        createAt: deleteResult.createAt,
        updateAt: deleteResult.updateAt,
      };
    }

    return {
      isSuccess: false,
      errorCode: ErrorCode.ETC,
      errorMsg: GetErrorMsg(ErrorCode.ETC),
    };
  }

  async createDefaultTools(machineCode: string) {
    for (let i = 0; i < DEFAULT_TOOL.length; i++) {
      const currentTool = DEFAULT_TOOL[i];
      currentTool.machineCode = machineCode;

      await this.create(currentTool);
    }
  }
}
