import { ObjectType, Field } from '@nestjs/graphql';
import { Prop, Schema } from '@nestjs/mongoose';

@Schema({ collection: 'lineMaster' })
@ObjectType()
export class Line {
  @Prop({ name: 'workshop_code' })
  @Field(() => String, { description: '공장 코드' })
  workshopCode: string;

  @Prop({ name: 'line_code' })
  @Field(() => String, { description: '라인 코드' })
  lineCode: string;

  @Prop({ name: 'line_name' })
  @Field(() => String, { description: '라인명' })
  lineName: string;

  @Prop({ name: 'create_at', default: new Date() })
  @Field(() => Date, { description: '생성 일시' })
  createAt?: Date;

  @Prop({ name: 'update_at', default: new Date() })
  @Field(() => Date, { description: '수정 일시', nullable: true })
  updateAt?: Date;
}
