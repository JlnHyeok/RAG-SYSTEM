import { InputType, Int, Field } from '@nestjs/graphql';
import { User } from '../entities/user.entity';

@InputType()
export class CreateUserInput {
  @Field(() => String, { description: '사용자 ID' })
  userId: string;

  @Field(() => String, { description: '사용자 이름' })
  userName: string;

  @Field(() => Int, { description: '사용자 권한' })
  userRole: number;

  getEntity(): User {
    return {
      userId: this.userId,
      password: '',
      userName: this.userName,
      userRole: this.userRole,
      resetFlag: 'Y',
    };
  }
}
