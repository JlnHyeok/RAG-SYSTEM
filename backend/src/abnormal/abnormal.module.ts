import { forwardRef, Module } from '@nestjs/common';
import { AbnormalService } from './abnormal.service';
import { AbnormalResolver } from './abnormal.resolver';
import { AbnormalController } from './abnormal.controller';
import { PubsubModule } from 'src/pubsub/pubsub.module';
import { MongooseModule, SchemaFactory } from '@nestjs/mongoose';
import { Abnormal, AbnormalSummary } from './entities/abnormal.entity';
import { ProductModule } from 'src/product/product.module';
import { RawModule } from 'src/raw/raw.module';
import { ThresholdModule } from 'src/master/threshold/threshold.module';

@Module({
  imports: [
    PubsubModule,
    MongooseModule.forFeature([
      { name: Abnormal.name, schema: SchemaFactory.createForClass(Abnormal) },
    ]),
    MongooseModule.forFeature([
      {
        name: AbnormalSummary.name,
        schema: SchemaFactory.createForClass(AbnormalSummary),
      },
    ]),
    forwardRef(() => RawModule),
    forwardRef(() => ProductModule),
    forwardRef(() => ThresholdModule),
  ],
  providers: [AbnormalResolver, AbnormalService],
  controllers: [AbnormalController],
  exports: [AbnormalService],
})
export class AbnormalModule {}
