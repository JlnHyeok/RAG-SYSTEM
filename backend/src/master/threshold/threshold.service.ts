import { Injectable } from '@nestjs/common';
import {
  UpdateMultiThresholdInput,
  UpdateThresholdInput,
} from './dto/update-threshold.input';
import { CreateThresholdInput } from './dto/create-threshold.input';
import { InjectModel } from '@nestjs/mongoose';
import { Threshold } from './entities/threshold.entity';
import { Model, PipelineStage, Types } from 'mongoose';
import {
  ThresholdMutationOutput,
  ThresholdQueryOutput,
} from './dto/threshold.output';
import { FilterThresholdInput } from './dto/filter-threshold.input';
import { ErrorCode, GetErrorMsg } from 'src/common/dto/common.enum';
import { ToolService } from '../tool/tool.service';

@Injectable()
export class ThresholdService {
  constructor(
    @InjectModel(Threshold.name)
    private readonly thresholdModel: Model<Threshold>,
    private readonly toolService: ToolService,
  ) {}

  async create(createThresholdInput: CreateThresholdInput) {
    // const thresholdList = await this.find(null);
    // let resultCode = this.generateThresholdId();
    // const filterThresholds = thresholdList.filter(
    //  (t) => t.machineCode == createThresholdInput.machineCode,
    // );

    // const thresholdCodes = new Array(filterThresholds.length);
    // filterThresholds.forEach((f, fIdx) => {
    // thresholdCodes[fIdx] = f.thresholdCode;
    // });

    const output = new ThresholdMutationOutput();
    const currentThreshold = await this.findOne(
      createThresholdInput.machineCode,
    );

    if (currentThreshold) {
      output.isSuccess = false;
      output.errorCode = ErrorCode.DUPLICATED;
      output.errorMsg = GetErrorMsg(ErrorCode.DUPLICATED);

      return output;
    }

    const temp_ct =
      createThresholdInput.minThresholdCt == null
        ? 0
        : createThresholdInput.minThresholdCt;

    const temp_load =
      createThresholdInput.minThresholdLoad == null
        ? 0
        : createThresholdInput.minThresholdLoad;

    const newThreshold = await this.thresholdModel.create({
      machineCode: createThresholdInput.machineCode,
      maxThresholdCt: createThresholdInput.maxThresholdCt,
      maxThresholdLoad: createThresholdInput.maxThresholdLoad,
      thresholdLoss: createThresholdInput.thresholdLoss,
      predictPeriod: createThresholdInput.predictPeriod,
      minThresholdCt: temp_ct,
      minThresholdLoad: temp_load,
      tool1Threshold: createThresholdInput.tool1Threshold,
      tool2Threshold: createThresholdInput.tool2Threshold,
      tool3Threshold: createThresholdInput.tool3Threshold,
      tool4Threshold: createThresholdInput.tool4Threshold,
      remark: createThresholdInput.remark,
      selected: createThresholdInput.selected,
      createAt: Date.now(),
      updateAt: Date.now(),

      // thresholdCode: resultCode,
    });

    if (newThreshold) {
      output.isSuccess = true;
      output.machineCode = newThreshold.machineCode;
      output.maxThresholdCt = newThreshold.maxThresholdCt;
      output.maxThresholdLoad = newThreshold.maxThresholdLoad;
      output.thresholdLoss = newThreshold.thresholdLoss;
      output.predictPeriod = newThreshold.predictPeriod;
      output.createAt = newThreshold.createAt;
      output.updateAt = newThreshold.updateAt;
      output.minThresholdCt = temp_ct;
      output.minThresholdLoad = temp_load;
      output.tool1Threshold = newThreshold.tool1Threshold;
      output.tool2Threshold = newThreshold.tool2Threshold;
      output.tool3Threshold = newThreshold.tool3Threshold;
      output.tool4Threshold = newThreshold.tool4Threshold;
      output.remark = newThreshold.remark;
      output.selected = newThreshold.selected;
      // output.thresholdCode = newThreshold.thresholdCode;
      output.thresholdId = newThreshold._id.toHexString();

      return output;
    }

    output.isSuccess = false;
    output.errorCode = ErrorCode.ETC;
    output.errorMsg = GetErrorMsg(ErrorCode.ETC);

    return output;
  }

  async find(filterThresholdInput: FilterThresholdInput) {
    const currentThresholds =
      await this.aggregateMachineMaster(filterThresholdInput);
    const currentTools = await this.toolService.find({
      machineCode: filterThresholdInput.machineCode,
    });

    if (
      currentThresholds &&
      currentThresholds.length > 0 &&
      currentTools.length >= 4
    ) {
      currentThresholds.forEach((t) => {
        t.tool1Name = currentTools[0].toolName;
        t.tool2Name = currentTools[1].toolName;
        t.tool3Name = currentTools[2].toolName;
        t.tool4Name = currentTools[3].toolName;
      });
    }

    return currentThresholds;

    if (filterThresholdInput) {
      const thresholds = await this.thresholdModel.find({
        workshopCode: filterThresholdInput.workshopCode,
        lineCode: filterThresholdInput.lineCode,
        opCode: filterThresholdInput.opCode,
      });

      return thresholds;
    }

    return await this.thresholdModel.find().sort({
      machineCode: 1,
    });
  }

  async findOne(machineCode: string) {
    const aggregated = await this.aggregateMachineMaster(null);

    return aggregated.find(
      (p) => p.machineCode == machineCode && p.selected == 'Y',
    );
  }

  async update(
    updateMultiThresholdInput: UpdateMultiThresholdInput[],
  ): Promise<ThresholdMutationOutput> {
    const result: ThresholdMutationOutput = {
      isSuccess: true,
    };

    for (const input of updateMultiThresholdInput) {
      const updateThresholdInput = new UpdateThresholdInput();
      updateThresholdInput.maxThresholdCt = input.maxThresholdCt;
      updateThresholdInput.maxThresholdLoad = input.maxThresholdLoad;
      updateThresholdInput.minThresholdCt = input.minThresholdCt;
      updateThresholdInput.minThresholdLoad = input.minThresholdLoad;
      updateThresholdInput.predictPeriod = input.predictPeriod;
      updateThresholdInput.remark = input.remark;
      updateThresholdInput.selected = input.selected;
      updateThresholdInput.thresholdLoss = input.thresholdLoss;
      updateThresholdInput.tool1Threshold = input.tool1Threshold;
      updateThresholdInput.tool2Threshold = input.tool2Threshold;
      updateThresholdInput.tool3Threshold = input.tool3Threshold;
      updateThresholdInput.tool4Threshold = input.tool4Threshold;

      const temp = await this.updateOne(
        input.thresholdId,
        updateThresholdInput,
      );

      if (!temp.isSuccess) {
        return temp;
      }
    }

    return result;
  }

  async delete(machineCode: string): Promise<ThresholdMutationOutput> {
    const result: ThresholdMutationOutput = {
      isSuccess: true,
    };

    const thresholdList = await this.find({
      machineCode: machineCode,
    });
    const entityList = thresholdList as (Threshold & { _id: Types.ObjectId })[];

    if (entityList) {
      for (const t of entityList) {
        const temp = await this.deleteOne(t._id.toHexString());

        if (!temp.isSuccess) {
          return temp;
        }
      }
    } else {
      return {
        isSuccess: false,
        errorCode: ErrorCode.ETC,
      };
    }

    return result;

    const deleteResult = await this.thresholdModel.findOneAndDelete(
      {
        machineCode,
      },
      {
        returnDocument: 'before',
      },
    );

    const temp_ct = !deleteResult.minThresholdCt
      ? 0
      : deleteResult.minThresholdCt;

    const temp_load = !deleteResult.minThresholdLoad
      ? 0
      : deleteResult.minThresholdLoad;

    if (deleteResult) {
      return {
        isSuccess: true,
        machineCode: deleteResult.machineCode,
        maxThresholdCt: deleteResult.maxThresholdCt
          ? deleteResult.maxThresholdCt / 1000000000
          : deleteResult.maxThresholdCt,
        maxThresholdLoad: deleteResult.maxThresholdLoad,
        thresholdLoss: deleteResult.thresholdLoss,
        predictPeriod: deleteResult.predictPeriod,
        createAt: deleteResult.createAt,
        updateAt: deleteResult.updateAt,
        minThresholdCt: temp_ct ? temp_ct / 1000000000 : temp_ct,
        minThresholdLoad: temp_load,
        tool1Threshold: deleteResult.tool1Threshold,
        tool2Threshold: deleteResult.tool2Threshold,
        tool3Threshold: deleteResult.tool3Threshold,
        tool4Threshold: deleteResult.tool4Threshold,
        remark: deleteResult.remark,
        selected: deleteResult.selected,
        // thresholdCode: deleteResult.thresholdCode,
      };
    }

    return {
      isSuccess: false,
      errorCode: ErrorCode.ETC,
      errorMsg: GetErrorMsg(ErrorCode.ETC),
    };
  }

  async createThreshold(createThresholdInput: CreateThresholdInput) {
    //const thresholdList = await this.find(null);
    //let resultCode = this.generateThresholdId();
    //const filterThresholds = thresholdList.filter(
    //  (t) => t.machineCode == createThresholdInput.machineCode,
    //);

    // const thresholdCodes = new Array(filterThresholds.length);
    // filterThresholds.forEach((f, fIdx) => {
    //  thresholdCodes[fIdx] = f.thresholdCode;
    //});

    const output = new ThresholdMutationOutput();

    const newThreshold = await this.thresholdModel.create({
      machineCode: createThresholdInput.machineCode,
      maxThresholdCt: createThresholdInput.maxThresholdCt
        ? createThresholdInput.maxThresholdCt * 1000000000
        : 0,
      maxThresholdLoad: createThresholdInput.maxThresholdLoad,
      thresholdLoss: createThresholdInput.thresholdLoss,
      predictPeriod: createThresholdInput.predictPeriod,
      minThresholdCt: createThresholdInput.minThresholdCt
        ? createThresholdInput.minThresholdCt * 1000000000
        : 0,
      minThresholdLoad: createThresholdInput.minThresholdLoad,
      tool1Threshold: createThresholdInput.tool1Threshold,
      tool2Threshold: createThresholdInput.tool2Threshold,
      tool3Threshold: createThresholdInput.tool3Threshold,
      tool4Threshold: createThresholdInput.tool4Threshold,
      remark: createThresholdInput.remark,
      selected: 'N',
      // thresholdCode: resultCode,
    });

    if (newThreshold) {
      output.isSuccess = true;
      output.machineCode = newThreshold.machineCode;
      output.maxThresholdCt = newThreshold.maxThresholdCt;
      output.maxThresholdLoad = newThreshold.maxThresholdLoad;
      output.thresholdLoss = newThreshold.thresholdLoss;
      output.predictPeriod = newThreshold.predictPeriod;
      output.createAt = newThreshold.createAt;
      output.updateAt = newThreshold.updateAt;
      output.minThresholdCt = newThreshold.minThresholdCt;
      output.minThresholdLoad = newThreshold.minThresholdLoad;
      output.tool1Threshold = newThreshold.tool1Threshold;
      output.tool2Threshold = newThreshold.tool2Threshold;
      output.tool3Threshold = newThreshold.tool3Threshold;
      output.tool4Threshold = newThreshold.tool4Threshold;
      output.remark = newThreshold.remark;
      output.selected = newThreshold.selected;
      // output.thresholdCode = newThreshold.thresholdCode;
      output.thresholdId = newThreshold._id.toHexString();
      return output;
    }

    output.isSuccess = false;
    output.errorCode = ErrorCode.ETC;
    output.errorMsg = GetErrorMsg(ErrorCode.ETC);

    return output;
  }

  async deleteOne(treshold_Id: string): Promise<ThresholdMutationOutput> {
    const deleteResult = await this.thresholdModel.findOneAndDelete(
      {
        _id: treshold_Id,
      },
      {
        returnDocument: 'before',
      },
    );

    if (deleteResult) {
      const temp_ct = !deleteResult.minThresholdCt
        ? 0
        : deleteResult.minThresholdCt;

      const temp_load = !deleteResult.minThresholdLoad
        ? 0
        : deleteResult.minThresholdLoad;

      return {
        isSuccess: true,
        machineCode: deleteResult.machineCode,
        maxThresholdCt: deleteResult.maxThresholdCt
          ? deleteResult.maxThresholdCt / 1000000000
          : deleteResult.maxThresholdCt,
        maxThresholdLoad: deleteResult.maxThresholdLoad,
        thresholdLoss: deleteResult.thresholdLoss,
        predictPeriod: deleteResult.predictPeriod,
        createAt: deleteResult.createAt,
        updateAt: deleteResult.updateAt,
        minThresholdCt: temp_ct ? temp_ct / 1000000000 : temp_ct,
        minThresholdLoad: temp_load,
        tool1Threshold: deleteResult.tool1Threshold,
        tool2Threshold: deleteResult.tool2Threshold,
        tool3Threshold: deleteResult.tool3Threshold,
        tool4Threshold: deleteResult.tool4Threshold,
        remark: deleteResult.remark,
        selected: deleteResult.selected,
        // thresholdCode: deleteResult.thresholdCode,
      };
    }

    return {
      isSuccess: false,
      errorCode: ErrorCode.ETC,
      errorMsg: GetErrorMsg(ErrorCode.ETC),
    };
  }

  async updateOne(
    treshold_Id: string,
    updateThresholdInput: UpdateThresholdInput,
  ): Promise<ThresholdMutationOutput> {
    const updateResult = await this.thresholdModel.findOneAndUpdate(
      {
        _id: treshold_Id,
      },
      {
        ...updateThresholdInput,
        maxThresholdCt: updateThresholdInput.maxThresholdCt
          ? updateThresholdInput.maxThresholdCt * 1000000000
          : updateThresholdInput.maxThresholdCt,

        minThresholdCt: updateThresholdInput.minThresholdCt
          ? updateThresholdInput.minThresholdCt * 1000000000
          : updateThresholdInput.minThresholdCt,

        updateAt: new Date(),
      },
      {
        returnDocument: 'after',
      },
    );

    const temp_ct = !updateResult.minThresholdCt
      ? 1
      : updateResult.minThresholdCt;

    const temp_load = !updateResult.minThresholdLoad
      ? 1
      : updateResult.minThresholdLoad;

    const temp_tool1Threshold = !updateResult.tool1Threshold
      ? 1
      : updateResult.tool1Threshold;

    const temp_tool2Threshold = !updateResult.tool2Threshold
      ? 1
      : updateResult.tool2Threshold;

    const temp_tool3Threshold = !updateResult.tool3Threshold
      ? 1
      : updateResult.tool3Threshold;

    const temp_tool4Threshold = !updateResult.tool4Threshold
      ? 1
      : updateResult.tool4Threshold;

    if (updateResult) {
      return {
        isSuccess: true,
        machineCode: updateResult.machineCode,

        maxThresholdCt: updateResult.maxThresholdCt
          ? updateResult.maxThresholdCt / 1000000000
          : updateResult.maxThresholdCt, // 초 단위를 ns 단위로 변경하여 저장

        maxThresholdLoad: updateResult.maxThresholdLoad,
        thresholdLoss: updateResult.thresholdLoss,
        predictPeriod: updateResult.predictPeriod,
        createAt: updateResult.createAt,
        updateAt: updateResult.updateAt,
        minThresholdCt: temp_ct ? temp_ct / 1000000000 : temp_ct,
        minThresholdLoad: temp_load,
        tool1Threshold: temp_tool1Threshold,
        tool2Threshold: temp_tool2Threshold,
        tool3Threshold: temp_tool3Threshold,
        tool4Threshold: temp_tool4Threshold,
        remark: updateResult.remark,
        selected: updateResult.selected,
        // thresholdCode: updateResult.thresholdCode,
      };
    }

    return {
      isSuccess: false,
      errorCode: ErrorCode.ETC,
      errorMsg: GetErrorMsg(ErrorCode.ETC),
    };
  }

  private async aggregateMachineMaster(
    filterThresholdInput: FilterThresholdInput,
  ): Promise<ThresholdQueryOutput[]> {
    const stages: PipelineStage[] = [
      {
        $lookup: {
          from: 'machineMaster',
          localField: 'machineCode',
          foreignField: 'machineCode',
          as: 'machineMaster',
        },
      },
      {
        $unwind: {
          path: '$machineMaster',
        },
      },
      {
        $lookup: {
          from: 'operationMaster',
          localField: 'machineMaster.opCode',
          foreignField: 'opCode',
          as: 'operationMaster',
        },
      },
      {
        $unwind: {
          path: '$operationMaster',
        },
      },
    ];

    if (filterThresholdInput) {
      stages.push({
        $match: {
          'machineMaster.workshopCode': filterThresholdInput.workshopCode,
          'machineMaster.lineCode': filterThresholdInput.lineCode,
          'machineMaster.opCode': filterThresholdInput.opCode,
          'machineMaster.machineCode': filterThresholdInput.machineCode,
        },
      });
    }

    stages.push({
      $project: {
        workshopCode: '$machineMaster.workshopCode',
        lineCode: '$machineMaster.lineCode',
        opCode: '$operationMaster.opCode',
        opName: '$operationMaster.opName',
        machineCode: '$machineMaster.machineCode',
        machineName: '$machineMaster.machineName',
        maxThresholdCt: 1,
        maxThresholdLoad: 1,
        minThresholdCt: 1,
        minThresholdLoad: 1,
        thresholdLoss: 1,
        predictPeriod: 1,
        createAt: 1,
        updateAt: 1,
        tool1Threshold: 1,
        tool2Threshold: 1,
        tool3Threshold: 1,
        tool4Threshold: 1,
        remark: 1,
        selected: 1,
        // thresholdCode: 1,
      },
    });

    const aggregateResult = await this.thresholdModel.aggregate(stages);

    if (aggregateResult && aggregateResult.length > 0) {
      const result: ThresholdQueryOutput[] = aggregateResult.map((d) => {
        return {
          ...d,
          tool1Name: 'T1',
          tool2Name: 'T2',
          tool3Name: 'T3',
          tool4Name: 'T4',
          thresholdId: d._id.toHexString(),
          maxThresholdCt: d.maxThresholdCt
            ? d.maxThresholdCt / 1000000000
            : d.maxThresholdCt,

          minThresholdCt: d.minThresholdCt
            ? d.minThresholdCt / 1000000000
            : d.minThresholdCt,
        };
      });

      return result;
    }

    return [];
  }
  private generateThresholdId() {
    const now = new Date(Date.now());
    const year = now.getFullYear(); // 2023
    const month = (now.getMonth() + 1).toString().padStart(2, '0'); // 7 (0~11)
    const date = now.getDate().toString().padStart(2, '0'); // 8 (1~31)
    const hour = now.getHours().toString().padStart(2, '0'); // 7
    const minute = now.getMinutes().toString().padStart(2, '0'); // 2
    const second = now.getSeconds().toString().padStart(2, '0'); // 1

    return `threshold_${year}${month}${date}${hour}${minute}${second}`;
  }
}
