import { forwardRef, Module } from '@nestjs/common';
import { ToolHistoryService } from './tool-history.service';
import { ToolHistoryResolver } from './tool-history.resolver';
import { ToolHistoryController } from './tool-history.controller';
import { MongooseModule, SchemaFactory } from '@nestjs/mongoose';
import { PubsubModule } from 'src/pubsub/pubsub.module';
import { ToolHistory } from './entities/tool-history.entity';
import { ToolModule } from 'src/master/tool/tool.module';
import { ToolChangeModule } from 'src/tool-change/tool-change.module';
import { MachineModule } from 'src/master/machine/machine.module';
import { createToolInfluxProvider } from 'src/app.provider';
import { InfluxModule } from 'src/influx/influx.module';

@Module({
  imports: [
    PubsubModule,
    MongooseModule.forFeature([
      {
        name: ToolHistory.name,
        schema: SchemaFactory.createForClass(ToolHistory),
      },
    ]),
    forwardRef(() => MachineModule),
    forwardRef(() => ToolModule),
    forwardRef(() => ToolChangeModule),
    forwardRef(() => InfluxModule),
  ],
  providers: [
    ToolHistoryResolver,
    ToolHistoryService,
    createToolInfluxProvider(),
  ],
  controllers: [ToolHistoryController],
  exports: [ToolHistoryService],
})
export class ToolHistoryModule {}
