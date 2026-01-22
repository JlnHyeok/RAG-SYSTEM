import { Resolver, Query, Mutation, Args } from '@nestjs/graphql';
import { LineService } from './line.service';
import { Line } from './entities/line.entity';
import { CreateLineInput } from './dto/create-line.input';
import { UpdateLineInput } from './dto/update-line.input';
import { LineMutationOutput } from './dto/line.output';
import { FilterLineInput } from './dto/filter-line.input';
import { UseGuards } from '@nestjs/common';
import { AuthGuard } from 'src/auth/auth.guard';
import { ROLE_ADMIN } from 'src/role/role.constants';
import { Role } from 'src/role/role.decorator';
import { RoleGuard } from 'src/role/role.guard';

@Resolver(() => Line)
export class LineResolver {
  constructor(private readonly lineService: LineService) {}

  @UseGuards(...[AuthGuard, RoleGuard])
  @Role(ROLE_ADMIN)
  @Mutation(() => LineMutationOutput)
  createLine(@Args('createLineInput') createLineInput: CreateLineInput) {
    return this.lineService.create(createLineInput);
  }

  @UseGuards(...[AuthGuard])
  @Query(() => [Line], { name: 'lines' })
  find(
    @Args('filterLineInput', { nullable: true })
    filterLineInput: FilterLineInput,
  ) {
    return this.lineService.find(filterLineInput);
  }

  @UseGuards(...[AuthGuard])
  @Query(() => Line, { name: 'line' })
  findOne(@Args('lineCode', { type: () => String }) lineCode: string) {
    return this.lineService.findOne(lineCode);
  }

  @UseGuards(...[AuthGuard, RoleGuard])
  @Role(ROLE_ADMIN)
  @Mutation(() => LineMutationOutput)
  updateLine(
    @Args('lineCode', { type: () => String }) lineCode: string,
    @Args('updateLineInput') updateLineInput: UpdateLineInput,
  ) {
    return this.lineService.update(lineCode, updateLineInput);
  }

  @UseGuards(...[AuthGuard, RoleGuard])
  @Role(ROLE_ADMIN)
  @Mutation(() => LineMutationOutput)
  deleteLine(@Args('lineCode', { type: () => String }) lineCode: string) {
    return this.lineService.delete(lineCode);
  }
}
