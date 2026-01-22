import { forwardRef, Module } from '@nestjs/common';
import { CommonService } from './common.service';
import { CommonResolver } from './common.resolver';
import { LineModule } from 'src/master/line/line.module';
import { MachineModule } from 'src/master/machine/machine.module';
import { OperationModule } from 'src/master/operation/operation.module';
import { WorkshopModule } from 'src/master/workshop/workshop.module';

@Module({
  imports: [
    forwardRef(() => WorkshopModule),
    forwardRef(() => LineModule),
    forwardRef(() => OperationModule),
    forwardRef(() => MachineModule),
  ],
  providers: [CommonResolver, CommonService],
  exports: [CommonService],
})
export class CommonModule {}
