import { Module } from '@nestjs/common';
import { MonitorService } from './monitor.service';
import { MonitorResolver } from './monitor.resolver';
import { RawModule } from 'src/raw/raw.module';
import { ProductModule } from 'src/product/product.module';
import { AbnormalModule } from 'src/abnormal/abnormal.module';
import { ToolHistoryModule } from 'src/tool-history/tool-history.module';
import { OperationModule } from 'src/master/operation/operation.module';
import { LineModule } from 'src/master/line/line.module';
import { WorkshopModule } from 'src/master/workshop/workshop.module';
import { MachineModule } from 'src/master/machine/machine.module';

@Module({
  // imports: [AbnormalModule, ProductModule, RawModule, ToolModule],
  imports: [
    RawModule,
    ProductModule,
    AbnormalModule,
    ToolHistoryModule,
    OperationModule,
    WorkshopModule,
    LineModule,
    MachineModule,
  ],
  providers: [MonitorResolver, MonitorService],
})
export class MonitorModule {}
