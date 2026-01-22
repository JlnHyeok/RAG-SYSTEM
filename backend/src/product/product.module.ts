import { forwardRef, Module } from '@nestjs/common';
import { ProductService } from './product.service';
import { ProductResolver } from './product.resolver';
import { ProductController } from './product.controller';
import { MongooseModule, SchemaFactory } from '@nestjs/mongoose';
import { Product } from './entities/product.entity';
import { PubsubModule } from 'src/pubsub/pubsub.module';
import { AbnormalModule } from 'src/abnormal/abnormal.module';
import { ThresholdModule } from 'src/master/threshold/threshold.module';
import { MachineModule } from 'src/master/machine/machine.module';
import { createProductInfluxProvider } from 'src/app.provider';
import { InfluxModule } from 'src/influx/influx.module';

@Module({
  imports: [
    MongooseModule.forFeature([
      { name: Product.name, schema: SchemaFactory.createForClass(Product) },
    ]),
    PubsubModule,
    forwardRef(() => AbnormalModule),
    forwardRef(() => MachineModule),
    forwardRef(() => ThresholdModule),
    forwardRef(() => InfluxModule),
  ],
  providers: [ProductResolver, ProductService, createProductInfluxProvider()],
  controllers: [ProductController],
  exports: [ProductService],
})
export class ProductModule {}
