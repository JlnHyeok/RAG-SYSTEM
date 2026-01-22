import { Resolver, Query, Mutation, Args } from '@nestjs/graphql';
import { MachineService } from './machine.service';
import { Machine } from './entities/machine.entity';
import { CreateMachineInput } from './dto/create-machine.input';
import { UpdateMachineInput } from './dto/update-machine.input';
import { FilterMachineInput } from './dto/filter-machine.input';
import { ThresholdService } from 'src/master/threshold/threshold.service';
import {
  MachineMutationOutput,
  MachineQueryOutput,
} from './dto/machine.output';
import { UseGuards } from '@nestjs/common';
import { AuthGuard } from 'src/auth/auth.guard';
import { ROLE_ADMIN } from 'src/role/role.constants';
import { Role } from 'src/role/role.decorator';
import { RoleGuard } from 'src/role/role.guard';

@Resolver(() => Machine)
export class MachineResolver {
  constructor(
    private readonly machineService: MachineService,
    private readonly thresholdService: ThresholdService,
  ) {}

  @UseGuards(...[AuthGuard, RoleGuard])
  @Role(ROLE_ADMIN)
  @Mutation(() => MachineMutationOutput)
  async createMachine(
    @Args('createMachineInput') createMachineInput: CreateMachineInput,
  ) {
    const newMachine = await this.machineService.create(createMachineInput);

    if (!newMachine.isSuccess) {
      return newMachine;
    }

    // 등록한 설비 정보를 이용하여 임계치 기준정보 생성
    const newThreshold = await this.thresholdService.create({
      workshopCode: newMachine.workshopCode,
      lineCode: newMachine.lineCode,
      opCode: newMachine.opCode,
      machineCode: newMachine.machineCode,
      maxThresholdCt: 0,
      maxThresholdLoad: 0,
      thresholdLoss: 0,
      predictPeriod: 0,
      minThresholdCt: 0,
      minThresholdLoad: 0,
      tool1Threshold: 0,
      tool2Threshold: 0,
      tool3Threshold: 0,
      tool4Threshold: 0,
      remark: '',
      selected: 'Y',
      // thresholdCode: 0,
    });

    // 임계치 기준 정보 생성에 실패할 경우 등록된 설비 정보 삭제
    if (!newThreshold.isSuccess) {
      await this.machineService.delete(newMachine.machineCode);

      const failResult: MachineMutationOutput = {
        isSuccess: false,
      };

      return failResult;
    }

    return newMachine;
  }

  @UseGuards(...[AuthGuard])
  @Query(() => [MachineQueryOutput], { name: 'machines' })
  async find(
    @Args('filterMachineInput', { nullable: true })
    filterMachineInput: FilterMachineInput,
  ) {
    return await this.machineService.find(filterMachineInput);
  }

  @UseGuards(...[AuthGuard])
  @Query(() => Machine, { name: 'machine' })
  findOne(@Args('machineCode', { type: () => String }) machineCode: string) {
    return this.machineService.findOne(machineCode);
  }

  @UseGuards(...[AuthGuard, RoleGuard])
  @Role(ROLE_ADMIN)
  @Mutation(() => MachineMutationOutput)
  async updateMachine(
    @Args('machineCode', { type: () => String }) machineCode: string,
    @Args('updateMachineInput') updateMachineInput: UpdateMachineInput,
  ) {
    return await this.machineService.update(machineCode, updateMachineInput);
  }

  @UseGuards(...[AuthGuard, RoleGuard])
  @Role(ROLE_ADMIN)
  @Mutation(() => MachineMutationOutput)
  async deleteMachine(
    @Args('machineCode', { type: () => String }) machineCode: string,
  ) {
    // 삭제한 설비 정보를 이용하여 임계치 기준정보 삭제
    const deletedThreshold = await this.thresholdService.delete(machineCode);

    // 임계치 기준정보 삭제에 실패할 경우 삭제한 설비 정보 등록
    if (!deletedThreshold.isSuccess) {
      const failResult: MachineMutationOutput = {
        isSuccess: false,
      };

      return failResult;
    }

    const deletedMachine = await this.machineService.delete(machineCode);
    if (!deletedMachine.isSuccess) {
      return deletedMachine;
    }

    return deletedMachine;
  }
}
