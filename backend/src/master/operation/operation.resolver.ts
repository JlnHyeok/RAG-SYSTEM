import { Resolver, Query, Mutation, Args } from '@nestjs/graphql';
import { OperationService } from './operation.service';
import { Operation } from './entities/operation.entity';
import { CreateOperationInput } from './dto/create-operation.input';
import { UpdateOperationInput } from './dto/update-operation.input';
import { FilterOperationInput } from './dto/filter-operation.input';
import { UseGuards } from '@nestjs/common';
import { AuthGuard } from 'src/auth/auth.guard';
import { RoleGuard } from 'src/role/role.guard';
import { ROLE_ADMIN } from 'src/role/role.constants';
import { Role } from 'src/role/role.decorator';
import { OperationMutationOutput } from './dto/operation.output';

@Resolver(() => Operation)
export class OperationResolver {
  constructor(private readonly operationService: OperationService) {}

  @UseGuards(...[AuthGuard, RoleGuard])
  @Role(ROLE_ADMIN)
  @Mutation(() => OperationMutationOutput)
  async createOperation(
    @Args('createOperationInput') createOperationInput: CreateOperationInput,
  ) {
    return await this.operationService.create(createOperationInput);
  }

  @UseGuards(...[AuthGuard])
  @Query(() => Operation, { name: 'operation' })
  async findOne(@Args('opCode', { type: () => String }) opCode: string) {
    return await this.operationService.findOne(opCode);
  }

  @UseGuards(...[AuthGuard])
  @Query(() => [Operation], { name: 'operations' })
  async find(
    @Args('filterOperationInput', { nullable: true })
    filterOperationInput: FilterOperationInput,
  ) {
    return await this.operationService.find(filterOperationInput);
  }

  @UseGuards(...[AuthGuard, RoleGuard])
  @Role(ROLE_ADMIN)
  @Mutation(() => OperationMutationOutput)
  async updateOperation(
    @Args('opCode', { type: () => String }) opCode: string,
    @Args('updateOperationInput') updateOperationInput: UpdateOperationInput,
  ) {
    return await this.operationService.update(opCode, updateOperationInput);
  }

  @UseGuards(...[AuthGuard, RoleGuard])
  @Role(ROLE_ADMIN)
  @Mutation(() => OperationMutationOutput)
  async deleteOperation(
    @Args('opCode', { type: () => String }) opCode: string,
  ) {
    return await this.operationService.delete(opCode);
  }
}
