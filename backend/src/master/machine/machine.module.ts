import { forwardRef, Module } from '@nestjs/common';
import { MachineService } from './machine.service';
import { MachineResolver } from './machine.resolver';
import { MongooseModule, SchemaFactory } from '@nestjs/mongoose';
import { Machine } from './entities/machine.entity';
import { ThresholdModule } from 'src/master/threshold/threshold.module';
import { OperationModule } from '../operation/operation.module';
import { ToolModule } from '../tool/tool.module';

@Module({
  imports: [
    MongooseModule.forFeature([
      {
        name: Machine.name,
        schema: SchemaFactory.createForClass(Machine),
      },
    ]),
    forwardRef(() => ThresholdModule),
    forwardRef(() => OperationModule),
    forwardRef(() => ToolModule),
  ],
  providers: [MachineResolver, MachineService],
  exports: [MachineService],
})
export class MachineModule {}
