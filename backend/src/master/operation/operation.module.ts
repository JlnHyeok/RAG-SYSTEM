import { Module } from '@nestjs/common';
import { OperationService } from './operation.service';
import { OperationResolver } from './operation.resolver';
import { MongooseModule, SchemaFactory } from '@nestjs/mongoose';
import { Operation } from './entities/operation.entity';

@Module({
  imports: [
    MongooseModule.forFeature([
      {
        name: Operation.name,
        schema: SchemaFactory.createForClass(Operation),
      },
    ]),
  ],
  providers: [OperationResolver, OperationService],
  exports: [OperationService],
})
export class OperationModule {}
