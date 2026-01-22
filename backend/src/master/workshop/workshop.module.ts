import { Module } from '@nestjs/common';
import { WorkshopService } from './workshop.service';
import { WorkshopResolver } from './workshop.resolver';
import { MongooseModule, SchemaFactory } from '@nestjs/mongoose';
import { Workshop } from './entities/workshop.entity';

@Module({
  imports: [
    MongooseModule.forFeature([
      { name: Workshop.name, schema: SchemaFactory.createForClass(Workshop) },
    ]),
  ],
  providers: [WorkshopResolver, WorkshopService],
  exports: [WorkshopService]
})
export class WorkshopModule {}
