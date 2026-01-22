import { Module } from '@nestjs/common';
import { LineService } from './line.service';
import { LineResolver } from './line.resolver';
import { MongooseModule, SchemaFactory } from '@nestjs/mongoose';
import { Line } from './entities/line.entity';

@Module({
  imports: [
    MongooseModule.forFeature([
      {
        name: Line.name,
        schema: SchemaFactory.createForClass(Line),
      },
    ]),
  ],
  providers: [LineResolver, LineService],
  exports: [LineService],
})
export class LineModule {}
