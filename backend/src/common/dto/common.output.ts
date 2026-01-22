import { Field, ObjectType } from '@nestjs/graphql';
import { IsEnum } from 'class-validator';
import { ErrorCode } from './common.enum';

@ObjectType()
export class CommonMutationOutput {
  @Field(() => Boolean, { description: '성공 여부' })
  isSuccess: boolean;

  @IsEnum(ErrorCode)
  @Field(() => ErrorCode, { description: '에러 코드', nullable: true })
  errorCode?: ErrorCode;
  @Field(() => String, { description: '에러 내용', nullable: true })
  errorMsg?: string;
}
