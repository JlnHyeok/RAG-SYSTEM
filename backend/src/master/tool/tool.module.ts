import { Module } from '@nestjs/common';
import { ToolService } from './tool.service';
import { ToolResolver } from './tool.resolver';
import { MongooseModule, SchemaFactory } from '@nestjs/mongoose';
import { Tool } from './entities/tool.entity';
import { ToolController } from './tool.controller';

@Module({
  imports: [
    MongooseModule.forFeature([
      { name: Tool.name, schema: SchemaFactory.createForClass(Tool) },
    ]),
  ],
  controllers: [ToolController],
  providers: [ToolResolver, ToolService],
  exports: [ToolService],
})
export class ToolModule {}
