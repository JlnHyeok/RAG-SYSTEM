import { Resolver, Query, Mutation, Args } from '@nestjs/graphql';
import { ToolService } from './tool.service';
import { Tool } from './entities/tool.entity';
import { CreateToolInput } from './dto/create-tool.input';
import { UpdateToolInput } from './dto/update-tool.input';
import { ToolMutationOutput } from './dto/tool.output';
import { FilterToolCodeInput, FilterToolInput } from './dto/filter-tool.input';
import { UseGuards } from '@nestjs/common';
import { AuthGuard } from 'src/auth/auth.guard';
import { ROLE_ADMIN } from 'src/role/role.constants';
import { Role } from 'src/role/role.decorator';
import { RoleGuard } from 'src/role/role.guard';

@Resolver(() => Tool)
export class ToolResolver {
  constructor(private readonly toolService: ToolService) {}

  @UseGuards(...[AuthGuard, RoleGuard])
  @Role(ROLE_ADMIN)
  @Mutation(() => ToolMutationOutput)
  async createTool(@Args('createToolInput') createToolInput: CreateToolInput) {
    return await this.toolService.create(createToolInput);
  }

  @UseGuards(...[AuthGuard])
  @Query(() => [Tool], { name: 'tools' })
  async find(
    @Args('filterToolInput', { nullable: true })
    filterToolInput: FilterToolInput,
  ) {
    return await this.toolService.find(filterToolInput);
  }

  @UseGuards(...[AuthGuard])
  @Query(() => Tool, { name: 'tool' })
  findOne(
    @Args('filterToolCodeInput')
    filterToolCodeInput: FilterToolCodeInput,
  ) {
    return this.toolService.findOne(filterToolCodeInput);
  }

  @UseGuards(...[AuthGuard, RoleGuard])
  @Role(ROLE_ADMIN)
  @Mutation(() => ToolMutationOutput)
  async updateTool(
    @Args('filterToolCodeInput')
    filterToolCodeInput: FilterToolCodeInput,
    @Args('updateToolInput') updateToolInput: UpdateToolInput,
  ) {
    return await this.toolService.update(filterToolCodeInput, updateToolInput);
  }

  @UseGuards(...[AuthGuard, RoleGuard])
  @Role(ROLE_ADMIN)
  @Mutation(() => ToolMutationOutput)
  async deleteTool(
    @Args('filterToolCodeInput')
    filterToolCodeInput: FilterToolCodeInput,
  ) {
    return await this.toolService.delete(filterToolCodeInput);
  }
}
