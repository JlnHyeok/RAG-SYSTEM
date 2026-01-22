import { Injectable } from '@nestjs/common';
import { CreateLineInput } from './dto/create-line.input';
import { UpdateLineInput } from './dto/update-line.input';
import { InjectModel } from '@nestjs/mongoose';
import { Line } from './entities/line.entity';
import { Model } from 'mongoose';
import { LineMutationOutput } from './dto/line.output';
import { FilterLineInput } from './dto/filter-line.input';
import { ErrorCode, GetErrorMsg } from 'src/common/dto/common.enum';

@Injectable()
export class LineService {
  constructor(
    @InjectModel(Line.name)
    private readonly lineModel: Model<Line>,
  ) {}

  async create(createLineInput: CreateLineInput) {
    const output = new LineMutationOutput();
    const currentLine = await this.findOne(createLineInput.lineCode);

    if (currentLine) {
      output.isSuccess = false;
      output.errorCode = ErrorCode.DUPLICATED;
      // output.errorMsg = GetErrorMsg(ErrorCode.DUPLICATED);
      output.errorMsg = '라인 코드가 중복되었습니다';

      return output;
    }
    const newLine = await this.lineModel.create({
      workshopCode: createLineInput.workshopCode,
      lineCode: createLineInput.lineCode,
      lineName: createLineInput.lineName,
      createAt: Date.now(),
      updateAt: Date.now(),
    });

    if (newLine) {
      output.isSuccess = true;
      output.workshopCode = newLine.workshopCode;
      output.lineCode = newLine.lineCode;
      output.lineName = newLine.lineName;
      output.createAt = newLine.createAt;
      output.updateAt = newLine.updateAt;

      return output;
    }

    output.isSuccess = false;
    output.errorCode = ErrorCode.ETC;
    output.errorMsg = GetErrorMsg(ErrorCode.ETC);
    return output;
  }

  async find(filterLineInput: FilterLineInput) {
    if (!filterLineInput) {
      return await this.lineModel.find();
    }

    const lines = await this.lineModel.find({
      workshopCode: filterLineInput.workshopCode,
    });

    return lines;
  }

  async findOne(lineCode: string) {
    return await this.lineModel.findOne({ lineCode });
  }

  async update(
    lineCode: string,
    updateLineInput: UpdateLineInput,
  ): Promise<LineMutationOutput> {
    const updateResult = await this.lineModel.findOneAndUpdate(
      {
        lineCode,
      },
      {
        ...updateLineInput,
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
        lineName: updateResult.lineName,
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

  async delete(lineCode: string): Promise<LineMutationOutput> {
    const deleteResult = await this.lineModel.findOneAndDelete(
      {
        lineCode,
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
        lineName: deleteResult.lineName,
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
