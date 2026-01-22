import { Injectable } from '@nestjs/common';
import { CreateOperationInput } from './dto/create-operation.input';
import { UpdateOperationInput } from './dto/update-operation.input';
import { FilterOperationInput } from './dto/filter-operation.input';
import { InjectModel } from '@nestjs/mongoose';
import { Operation } from './entities/operation.entity';
import { Model } from 'mongoose';
import { OperationMutationOutput } from './dto/operation.output';
import { ErrorCode, GetErrorMsg } from 'src/common/dto/common.enum';

@Injectable()
export class OperationService {
  constructor(
    @InjectModel(Operation.name)
    private readonly operationModel: Model<Operation>,
  ) {}

  async create(createOperationInput: CreateOperationInput) {
    const output = new OperationMutationOutput();
    const currentOperation = await this.findOne(createOperationInput.opCode);

    if (currentOperation) {
      output.isSuccess = false;
      output.errorCode = ErrorCode.DUPLICATED;
      // output.errorMsg = GetErrorMsg(ErrorCode.DUPLICATED);
      output.errorMsg = '공정 코드가 중복되었습니다';
      return output;
    }
    const newOperation = await this.operationModel.create({
      workshopCode: createOperationInput.workshopCode,
      lineCode: createOperationInput.lineCode,
      opCode: createOperationInput.opCode,
      opName: createOperationInput.opName,
      createAt: Date.now(),
      updateAt: Date.now(),
    });

    if (newOperation) {
      output.isSuccess = true;
      output.workshopCode = newOperation.workshopCode;
      output.lineCode = newOperation.lineCode;
      output.opCode = newOperation.opCode;
      output.opName = newOperation.opName;
      output.createAt = newOperation.createAt;
      output.updateAt = newOperation.updateAt;

      return output;
    }

    output.isSuccess = false;
    output.errorCode = ErrorCode.ETC;
    output.errorMsg = GetErrorMsg(ErrorCode.ETC);
    return output;
  }

  findOne(opCode: string) {
    return this.operationModel.findOne({ opCode });
  }

  async find(filterOperationInput: FilterOperationInput) {
    if (filterOperationInput) {
      const operations = await this.operationModel.find({
        workshopCode: filterOperationInput.workshopCode,
        lineCode: filterOperationInput.lineCode,
      });

      return operations;
    }

    return await this.operationModel.find().sort({
      opCode: 1,
    });
  }

  async update(
    opCode: string,
    updateOperationInput: UpdateOperationInput,
  ): Promise<OperationMutationOutput> {
    const updateResult = await this.operationModel.findOneAndUpdate(
      {
        opCode,
      },
      {
        ...updateOperationInput,
        updateAt: new Date(),
      },
      {
        returnDocument: 'after',
      },
    );

    if (updateResult) {
      return {
        isSuccess: true,
        workshopCode: updateResult.workshopCode,
        lineCode: updateResult.lineCode,
        opCode: updateResult.opCode,
        opName: updateResult.opName,
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

  async delete(opCode: string): Promise<OperationMutationOutput> {
    const deleteResult = await this.operationModel.findOneAndDelete(
      {
        opCode,
      },
      {
        returnDocument: 'before',
      },
    );

    if (deleteResult) {
      return {
        isSuccess: true,
        workshopCode: deleteResult.workshopCode,
        lineCode: deleteResult.lineCode,
        opCode: deleteResult.opCode,
        opName: deleteResult.opName,
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
