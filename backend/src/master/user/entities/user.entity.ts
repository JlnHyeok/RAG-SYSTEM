import { ObjectType, Field, Int } from '@nestjs/graphql';
import { Prop, Schema } from '@nestjs/mongoose';

@Schema({ collection: 'userMaster' })
@ObjectType()
export class User {
  @Prop({ name: 'user_id' })
  @Field(() => String, { description: '사용자 ID' })
  userId: string;

  @Prop({ name: 'password' })
  @Field(() => String, { description: '비밀번호' })
  password: string;

  @Prop({ name: 'user_name' })
  @Field(() => String, { description: '사용자 이름' })
  userName: string;

  @Prop({ name: 'user_role' })
  @Field(() => Int, { description: '사용자 권한' })
  userRole: number;

  @Prop({ name: 'reset_flag' })
  @Field(() => String, { description: '패스워드 재설정 여부' })
  resetFlag: string;

  @Prop({ name: 'create_at', default: new Date() })
  @Field(() => Date, { description: '생성 일시' })
  createAt?: Date;

  @Prop({ name: 'update_at', default: new Date() })
  @Field(() => Date, { description: '수정 일시', nullable: true })
  updateAt?: Date;
}
