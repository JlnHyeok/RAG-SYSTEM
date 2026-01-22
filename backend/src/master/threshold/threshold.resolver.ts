import { Resolver, Query, Mutation, Args } from '@nestjs/graphql';
import { ThresholdService } from './threshold.service';
import { Threshold } from './entities/threshold.entity';
import {
  UpdateMultiThresholdInput,
  UpdateThresholdInput,
} from './dto/update-threshold.input';
import { FilterThresholdInput } from './dto/filter-threshold.input';
import { CreateThresholdInput } from './dto/create-threshold.input';
import {
  ThresholdMutationOutput,
  ThresholdQueryOutput,
} from './dto/threshold.output';
import { UseGuards } from '@nestjs/common';
import { AuthGuard } from 'src/auth/auth.guard';
import { ROLE_ADMIN } from 'src/role/role.constants';
import { Role } from 'src/role/role.decorator';
import { RoleGuard } from 'src/role/role.guard';

@Resolver(() => Threshold)
export class ThresholdResolver {
  //! Create/Delete는 설비 정보 등록/삭제 시 함께 호출

  constructor(private readonly thresholdService: ThresholdService) {}

  @UseGuards(...[AuthGuard])
  @Query(() => [ThresholdQueryOutput], { name: 'thresholds' })
  async find(
    @Args('filterThresholdInput', { nullable: true })
    filterThresholdInput: FilterThresholdInput,
  ) {
    return await this.thresholdService.find(filterThresholdInput);
  }

  // @Query(() => ThresholdQueryOutput, { name: 'threshold' })
  // async findOne(
  //   @Args('machineCode', { type: () => String }) machineCode: string,
  // ) {
  //   return await this.thresholdService.findOne(machineCode);
  // }

  @UseGuards(...[AuthGuard, RoleGuard])
  @Role(ROLE_ADMIN)
  @Mutation(() => ThresholdMutationOutput)
  updateThresholdAll(
    @Args('updateThresholdInput', { type: () => [UpdateMultiThresholdInput] })
    UpdateMultiThresholdInput: UpdateMultiThresholdInput[],
  ) {
    return this.thresholdService.update(UpdateMultiThresholdInput);
  }

  @UseGuards(...[AuthGuard, RoleGuard])
  @Role(ROLE_ADMIN)
  @Mutation(() => ThresholdMutationOutput)
  createThreshold(
    // @Args('machineCode', { type: () => String }) machineCode: string,
    @Args('createThresholdInput') createThresholdInput: CreateThresholdInput,
  ) {
    // return this.thresholdService.create(createThresholdInput);
    return this.thresholdService.createThreshold(createThresholdInput);
  }

  @UseGuards(...[AuthGuard, RoleGuard])
  @Role(ROLE_ADMIN)
  @Mutation(() => ThresholdMutationOutput)
  deleteThreshold(
    @Args('thresholdId', { type: () => String }) thresholdId: string,
  ) {
    return this.thresholdService.deleteOne(thresholdId);
  }

  @UseGuards(...[AuthGuard, RoleGuard])
  @Role(ROLE_ADMIN)
  @Mutation(() => ThresholdMutationOutput)
  updateThreshold(
    @Args('thresholdId', { type: () => String }) thresholdId: string,
    @Args('updateThresholdInput') updateThresholdInput: UpdateThresholdInput,
  ) {
    return this.thresholdService.updateOne(thresholdId, updateThresholdInput);
  }
}
