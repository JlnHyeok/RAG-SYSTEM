import { forwardRef, Module } from '@nestjs/common';
import { ThresholdService } from './threshold.service';
import { ThresholdResolver } from './threshold.resolver';
import { MongooseModule, SchemaFactory } from '@nestjs/mongoose';
import { Threshold } from './entities/threshold.entity';
import { ThresholdController } from './threshold.controller';
import { MachineModule } from '../machine/machine.module';
import { ToolModule } from '../tool/tool.module';

@Module({
  imports: [
    MongooseModule.forFeature([
      {
        name: Threshold.name,
        schema: SchemaFactory.createForClass(Threshold),
      },
    ]),
    forwardRef(() => MachineModule),
    forwardRef(() => ToolModule),
  ],
  providers: [ThresholdResolver, ThresholdService],
  exports: [ThresholdService],
  controllers: [ThresholdController],
})
export class ThresholdModule {}
