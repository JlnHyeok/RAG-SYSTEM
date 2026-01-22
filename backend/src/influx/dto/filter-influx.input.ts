import { InputType, Field } from '@nestjs/graphql';
import { IInfluxRelationFilter } from '../interface/influx.interface';

@InputType()
export class FilterInfluxInput {
  @Field(() => Date, { description: '조회 시작 일시', nullable: true })
  rangeStart?: Date;

  @Field(() => Date, { description: '조회 종료 일시', nullable: true })
  rangeStop?: Date;

  @Field(() => String, {
    description: '조회 시작 일시 (상대 시간 문자열)',
    nullable: true,
  })
  rangeStartString?: string;

  @Field(() => [FilterInfluxTagInput], {
    description: '조회 Tag',
    nullable: true,
  })
  tags?: FilterInfluxTagInput[];

  @Field(() => String, { description: '통계 구분', nullable: true })
  aggregateInterval?: string;
}

@InputType()
export class FilterInfluxTagInput {
  @Field(() => String, { description: 'Tag 명' })
  tagName: string;

  @Field(() => String, { description: 'Tag 값' })
  tagValue: string;

  getInfluxFilter(): IInfluxRelationFilter {
    return {
      property: this.tagName,
      operator: '==',
      value: this.tagValue,
    };
  }
}
