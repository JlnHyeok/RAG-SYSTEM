import { Injectable } from '@nestjs/common';
import { CreateWorkshopInput } from './dto/create-workshop.input';
import { UpdateWorkshopInput } from './dto/update-workshop.input';
import { InjectModel } from '@nestjs/mongoose';
import { Workshop } from './entities/workshop.entity';
import { Model } from 'mongoose';
import { FilterWorkshopInput } from './dto/filter-workshop.input';
import { WorkshopMutationOutput } from './dto/workshop.output';
import { ErrorCode, GetErrorMsg } from 'src/common/dto/common.enum';

@Injectable()
export class WorkshopService {
  constructor(
    @InjectModel(Workshop.name)
    private readonly workshopModel: Model<Workshop>,
  ) {}

  async create(createWorkshopInput: CreateWorkshopInput) {
    const output = new WorkshopMutationOutput();
    const currentWorkshop = await this.findOne(
      createWorkshopInput.workshopCode,
    );

    // Workshop 코드 중복 검증
    if (currentWorkshop) {
      output.isSuccess = false;
      output.errorCode = ErrorCode.DUPLICATED;
      // output.errorMsg = GetErrorMsg(ErrorCode.DUPLICATED);
      output.errorMsg = '공장 코드가 중복되었습니다';

      return output;
    }

    const newWorkshop = await this.workshopModel.create({
      workshopCode: createWorkshopInput.workshopCode,
      workshopName: createWorkshopInput.workshopName,
      createAt: Date.now(),
      updateAt: Date.now(),
    });

    if (newWorkshop) {
      output.isSuccess = true;
      output.workshopCode = newWorkshop.workshopCode;
      output.workshopName = newWorkshop.workshopName;
      output.createAt = newWorkshop.createAt;
      output.updateAt = newWorkshop.updateAt;

      return output;
    }

    output.isSuccess = false;
    output.errorCode = ErrorCode.ETC;
    output.errorMsg = GetErrorMsg(ErrorCode.ETC);
    return output;
  }

  find(filterWorkshopInput: FilterWorkshopInput) {
    if (filterWorkshopInput) {
      return this.workshopModel
        .find({
          workshopCode: { $in: filterWorkshopInput.workshopCodes },
        })
        .sort({
          workshopCode: 1,
        });
    }

    return this.workshopModel.find().sort({
      workshopCode: 1,
    });
  }

  findOne(workshopCode: string) {
    return this.workshopModel.findOne({ workshopCode });
  }

  async update(
    workshopCode: string,
    updateWorkshopInput: UpdateWorkshopInput,
  ): Promise<WorkshopMutationOutput> {
    const updateResult = await this.workshopModel.findOneAndUpdate(
      { workshopCode },
      {
        workshopName: updateWorkshopInput.workshopName,
        updateAt: Date.now(),
      },
      {
        returnDocument: 'after',
      },
    );

    if (updateResult) {
      return {
        isSuccess: true,
        workshopCode: updateResult.workshopCode,
        workshopName: updateResult.workshopName,
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

  async delete(workshopCode: string): Promise<WorkshopMutationOutput> {
    const deleteResult = await this.workshopModel.findOneAndDelete(
      { workshopCode },
      {
        returnDocument: 'before',
      },
    );

    if (deleteResult) {
      return {
        isSuccess: true,
        workshopCode: deleteResult.workshopCode,
        workshopName: deleteResult.workshopName,
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
}
