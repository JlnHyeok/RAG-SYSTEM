import { Resolver, Query, Args, Subscription, Info } from '@nestjs/graphql';
import { ProductService } from './product.service';
import { Product } from './entities/product.entity';
import {
  FilterProductInfluxInput,
  FilterProductInput,
  FilterProductSumReportInput,
} from './dto/filter-product.input';
import { FilterCommonInput } from 'src/common/dto/filter-common.input';
import { TOPIC_MONITOR_PRODUCT } from 'src/pubsub/pubsub.constants';
import {
  ProductAbnormalOutput,
  ProductInfluxOutput,
  ProductLastOutput,
  ProductPaginationOutput,
  ProductSubscriptionOutput,
  ProductSumReportOutput,
} from './dto/product.output';
import { AbnormalService } from 'src/abnormal/abnormal.service';
import { UseGuards } from '@nestjs/common';
import { AuthGuard } from 'src/auth/auth.guard';
// import { UpdateProductRebuildInput } from './dto/update-product_rebuild.input';

@Resolver(() => Product)
export class ProductResolver {
  constructor(
    private readonly productService: ProductService,
    private readonly abnormalService: AbnormalService,
  ) {}

  @UseGuards(...[AuthGuard])
  @Query(() => ProductPaginationOutput, { name: 'products' })
  async find(
    @Args('filterProductInput', { nullable: true })
    filterProductInput: FilterProductInput,
  ) {
    return await this.productService.findPagination(filterProductInput);
  }

  @UseGuards(...[AuthGuard])
  @Query(() => ProductLastOutput, { name: 'lastProduct' })
  async findLast(
    @Args('filterCommonInput')
    filterCommonInput: FilterCommonInput,
  ) {
    return await this.productService.findLast(filterCommonInput);
  }

  @UseGuards(...[AuthGuard])
  @Query(() => [ProductAbnormalOutput], { name: 'productAbnormals' })
  async findProductAbnormal(
    @Args('filterProductInput')
    filterProductInput: FilterProductInput,
  ) {
    const products = await this.productService.find(filterProductInput);

    const productAbnormals = await Promise.all(
      products.map(async (v) => {
        let tempProduct = new ProductAbnormalOutput();
        const currentAbnormals = await this.abnormalService.find({
          commonFilter: filterProductInput.commonFilter,
          productNo: v.productNo,
        });

        tempProduct = {
          productNo: v.productNo,
          productBeginDate: v.startTime,
          productEndDate: v.endTime,
          productResult: v.productResult ? v.productResult : 'Y',
          productCt: v.ct,
          productCtResult: v.ctResult ? v.ctResult : 'Y',
          productAi: v.ai,
          productAiResult: v.aiResult ? v.aiResult : 'Y',
          productLoadSum: v.loadSum,
          productLoadSumResult: v.loadSumResult ? v.loadSumResult : 'Y',
          productCompleteStatus: v.completeStatus,

          abnormals: currentAbnormals.abnormals.sort((a, b) => {
            if (a.abnormalBeginDate > b.abnormalBeginDate) {
              return -1;
            }

            return 1;
          }),
        };

        return tempProduct;
      }),
    );

    return productAbnormals;
  }

  @UseGuards(...[AuthGuard])
  @Query(() => [ProductInfluxOutput], { name: 'productInfoReports' })
  async findInflux(
    @Args('filterProductInfluxInput')
    filterProductInfluxInput: FilterProductInfluxInput,
    @Info() info,
  ) {
    const selectedFields: Array<object> =
      info.fieldNodes[0].selectionSet.selections;
    const selectedNames = selectedFields.map((p) => p['name'].value);

    return await this.productService.findInflux(
      filterProductInfluxInput,
      selectedNames,
    );
  }

  @UseGuards(...[AuthGuard])
  @Query(() => [ProductSumReportOutput], { name: 'productSumReports' })
  async aggregateSum(
    @Args('filterProductSumReportInput')
    filterProductSumReportInput: FilterProductSumReportInput,
    @Info() info,
  ) {
    const selectedFields: Array<object> =
      info.fieldNodes[0].selectionSet.selections;
    const selectedNames = selectedFields.map((p) => p['name'].value);

    return await this.productService.aggregateSum(
      filterProductSumReportInput,
      selectedNames,
    );
  }

  // @Subscription(() => ProductSubscriptionOutput, {
  //   name: TOPIC_MONITOR_PRODUCT,
  // })
  // async monitor() {
  //   return await this.productService.monitor();
  // }
  @Subscription(() => ProductSubscriptionOutput, {
    name: TOPIC_MONITOR_PRODUCT,
  })
  async monitorTest(
    @Args('filterCommonInput')
    filterCommonInput: FilterCommonInput,
  ) {
    return await this.productService.monitor(filterCommonInput);
  }
}
