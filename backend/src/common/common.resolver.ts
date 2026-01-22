import { Query, Resolver } from '@nestjs/graphql';
import { CommonService } from './common.service';
import { WorkshopListOutput } from './dto/operation-info.output';

@Resolver()
export class CommonResolver {
  constructor(private readonly commonService: CommonService) {}

  @Query(() => [WorkshopListOutput], { name: 'workshopListOutput' })
  findDashboardMenus() {
    return this.commonService.findDashboardMenus();
  }
}
