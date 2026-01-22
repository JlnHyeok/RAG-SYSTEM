import { forwardRef, Module } from '@nestjs/common';
import { ToolChangeService } from './tool-change.service';
import { ToolChangeResolver } from './tool-change.resolver';
import { MongooseModule, SchemaFactory } from '@nestjs/mongoose';
import { ToolChange } from './entities/tool-change.entity';
import { ToolHistoryModule } from 'src/tool-history/tool-history.module';
import { ToolModule } from 'src/master/tool/tool.module';
import { MachineModule } from 'src/master/machine/machine.module';
import { PubsubModule } from 'src/pubsub/pubsub.module';
import { ToolChangeController } from './tool-change.controller';

@Module({
  imports: [
    MongooseModule.forFeature([
      {
        name: ToolChange.name,
        schema: SchemaFactory.createForClass(ToolChange),
      },
    ]),
    PubsubModule,
    forwardRef(() => MachineModule),
    forwardRef(() => ToolModule),
    forwardRef(() => ToolHistoryModule),
  ],
  providers: [ToolChangeResolver, ToolChangeService],
  exports: [ToolChangeService],
  controllers: [ToolChangeController],
})
export class ToolChangeModule {}
