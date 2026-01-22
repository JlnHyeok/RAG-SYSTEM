import { forwardRef, Logger, Module } from '@nestjs/common';
import { RawService } from './raw.service';
import { RawResolver } from './raw.resolver';
import { InfluxModule } from 'src/influx/influx.module';
import { createRawProvider } from 'src/app.provider';
import { ProductModule } from 'src/product/product.module';

@Module({
  imports: [InfluxModule, forwardRef(() => ProductModule)],
  providers: [RawResolver, RawService, createRawProvider(), Logger],
  exports: [RawService],
})
export class RawModule {}
