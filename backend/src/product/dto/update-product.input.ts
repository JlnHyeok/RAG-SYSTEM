import { InputType, Field, Int } from '@nestjs/graphql';

@InputType()
export class UpdateProductInput {
  @Field(() => String, { description: '생산 결과', nullable: true })
  productResult?: string;
  @Field(() => Int, { description: '부하 이상 값', nullable: true })
  ai?: number;
  @Field(() => String, { description: '부하 이상 발생 여부', nullable: true })
  aiResult?: string;
}
