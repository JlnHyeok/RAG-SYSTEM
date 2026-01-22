import { forwardRef, Inject, Injectable } from '@nestjs/common';
import { CreateMachineInput } from './dto/create-machine.input';
import { UpdateMachineInput } from './dto/update-machine.input';
import { InjectModel } from '@nestjs/mongoose';
import { Model } from 'mongoose';
import { Machine } from './entities/machine.entity';
import {
  MachineMutationOutput,
  MachineQueryOutput,
} from './dto/machine.output';
import { FilterMachineInput } from './dto/filter-machine.input';
import { ErrorCode, GetErrorMsg } from 'src/common/dto/common.enum';
import { ObjectId } from 'mongoose';
import { OperationService } from '../operation/operation.service';
import { ToolService } from '../tool/tool.service';

@Injectable()
export class MachineService {
  constructor(
    @InjectModel(Machine.name)
    private readonly machineModel: Model<Machine>,
    @Inject(forwardRef(() => OperationService))
    private readonly operationService: OperationService,
    @Inject(forwardRef(() => ToolService))
    private readonly toolService: ToolService,
  ) {}

  async create(createMachineInput: CreateMachineInput) {
    const output = new MachineMutationOutput();
    const currentMachine = await this.findOne(createMachineInput.machineCode);

    if (currentMachine) {
      output.isSuccess = false;
      output.errorCode = ErrorCode.DUPLICATED;
      // output.errorMsg = GetErrorMsg(ErrorCode.DUPLICATED);
      output.errorMsg = '설비 코드가 중복되었습니다';

      return output;
    }

    const currentOpMachines = await this.find({
      workshopCode: createMachineInput.workshopCode,
      lineCode: createMachineInput.lineCode,
      opCode: createMachineInput.opCode,
    });

    if (currentOpMachines && currentOpMachines.length > 0) {
      output.isSuccess = false;
      output.errorCode = ErrorCode.DUPLICATED;
      // output.errorMsg = GetErrorMsg(ErrorCode.DUPLICATED);
      output.errorMsg = '공정에 이미 등록된 설비가 있습니다';

      return output;
    }

    const newMachine = await this.machineModel.create({
      workshopCode: createMachineInput.workshopCode,
      lineCode: createMachineInput.lineCode,
      opCode: createMachineInput.opCode,
      machineCode: createMachineInput.machineCode,
      machineName: createMachineInput.machineName,
      machineIp: createMachineInput.machineIp,
      machinePort: createMachineInput.machinePort,
      createAt: Date.now(),
      updateAt: Date.now(),
    });

    if (newMachine) {
      // await this.toolService.createDefaultTools(newMachine.machineCode);

      output.isSuccess = true;
      output.workshopCode = newMachine.workshopCode;
      output.lineCode = newMachine.lineCode;
      output.opCode = newMachine.opCode;
      output.machineCode = newMachine.machineCode;
      output.machineName = newMachine.machineName;
      output.machineIp = newMachine.machineIp;
      output.machinePort = newMachine.machinePort;
      output.createAt = newMachine.createAt;
      output.updateAt = newMachine.updateAt;

      return output;
    }

    output.isSuccess = false;
    output.errorCode = ErrorCode.ETC;
    output.errorMsg = GetErrorMsg(ErrorCode.ETC);
    return output;
  }

  async find(filterMachineInput: FilterMachineInput) {
    const output: MachineQueryOutput[] = [];
    let machines: Machine[] = [];

    if (filterMachineInput) {
      machines = await this.machineModel.find({
        workshopCode: filterMachineInput.workshopCode,
        lineCode: filterMachineInput.lineCode,
        opCode: filterMachineInput.opCode,
      });
    } else {
      machines = await this.machineModel.find().sort({
        machineCode: 1,
      });
    }

    for (const m of machines) {
      const tempOutput = new MachineQueryOutput();
      tempOutput.workshopCode = m.workshopCode;
      tempOutput.lineCode = m.lineCode;
      tempOutput.opCode = m.opCode;
      tempOutput.opName = m.opCode;
      tempOutput.machineCode = m.machineCode;
      tempOutput.machineName = m.machineName;
      tempOutput.machineIp = m.machineIp;
      tempOutput.machinePort = m.machinePort;
      tempOutput.createAt = m.createAt;
      tempOutput.updateAt = m.updateAt;

      const currentOperation = await this.operationService.findOne(m.opCode);

      if (currentOperation) {
        tempOutput.opName = currentOperation.opName;
      }

      output.push(tempOutput);
    }

    return output;
  }

  findOne(machineCode: string) {
    return this.machineModel.findOne({ machineCode });
  }

  async update(
    machineCode: string,
    updateMachineInput: UpdateMachineInput,
  ): Promise<MachineMutationOutput> {
    const updateResult = await this.machineModel.findOneAndUpdate(
      {
        machineCode,
      },
      {
        ...updateMachineInput,
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
        machineCode: updateResult.machineCode,
        machineName: updateResult.machineName,
        machineIp: updateResult.machineIp,
        machinePort: updateResult.machinePort,
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

  async delete(machineCode: string): Promise<MachineMutationOutput> {
    const deleteResult = await this.machineModel.findOneAndDelete(
      {
        machineCode,
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
        machineCode: deleteResult.machineCode,
        machineName: deleteResult.machineName,
        machineIp: deleteResult.machineIp,
        machinePort: deleteResult.machinePort,
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
