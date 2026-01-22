import { Resolver, Query, Mutation, Args } from '@nestjs/graphql';
import { WorkshopService } from './workshop.service';
import { Workshop } from './entities/workshop.entity';
import { CreateWorkshopInput } from './dto/create-workshop.input';
import { UpdateWorkshopInput } from './dto/update-workshop.input';
import { FilterWorkshopInput } from './dto/filter-workshop.input';
import { UseGuards } from '@nestjs/common';
import { AuthGuard } from 'src/auth/auth.guard';
import { ROLE_ADMIN } from 'src/role/role.constants';
import { Role } from 'src/role/role.decorator';
import { RoleGuard } from 'src/role/role.guard';
import { WorkshopMutationOutput } from './dto/workshop.output';

@Resolver(() => Workshop)
export class WorkshopResolver {
  constructor(private readonly workshopService: WorkshopService) {}

  @UseGuards(...[AuthGuard, RoleGuard])
  @Role(ROLE_ADMIN)
  @Mutation(() => WorkshopMutationOutput)
  createWorkshop(
    @Args('createWorkshopInput') createWorkshopInput: CreateWorkshopInput,
  ) {
    return this.workshopService.create(createWorkshopInput);
  }

  @UseGuards(...[AuthGuard])
  @Query(() => [Workshop], { name: 'workshops' })
  find(
    @Args('filterWorkshopInput', { nullable: true })
    filterWorkshopInput: FilterWorkshopInput,
  ) {
    return this.workshopService.find(filterWorkshopInput);
  }

  @UseGuards(...[AuthGuard])
  @Query(() => Workshop, { name: 'workshop' })
  findOne(@Args('workshopCode', { type: () => String }) workshopCode: string) {
    return this.workshopService.findOne(workshopCode);
  }

  @UseGuards(...[AuthGuard, RoleGuard])
  @Role(ROLE_ADMIN)
  @Mutation(() => WorkshopMutationOutput)
  updateWorkshop(
    @Args('workshopCode', { type: () => String }) workshopCode: string,
    @Args('updateWorkshopInput') updateWorkshopInput: UpdateWorkshopInput,
  ) {
    return this.workshopService.update(workshopCode, updateWorkshopInput);
  }

  @UseGuards(...[AuthGuard, RoleGuard])
  @Role(ROLE_ADMIN)
  @Mutation(() => WorkshopMutationOutput)
  deleteWorkshop(
    @Args('workshopCode', { type: () => String }) workshopCode: string,
  ) {
    return this.workshopService.delete(workshopCode);
  }
}
