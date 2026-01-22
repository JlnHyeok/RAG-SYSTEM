import { Resolver, Query, Mutation, Args } from '@nestjs/graphql';
import { UsersService } from './users.service';
import { User } from './entities/user.entity';
import { CreateUserInput } from './dto/create-user.input';
import { UpdateUserInput } from './dto/update-user.input';
import { FilterUserInput } from './dto/filter-user.input';
import { LoginUserInput } from './dto/login-user.input';
import { LoginUserOutput } from './dto/login-user.output';
import { LogoutUserOutput } from './dto/logout-user.output';
import { ConfigService } from '@nestjs/config';
import { ROLE_ADMIN } from 'src/role/role.constants';
import { UseGuards } from '@nestjs/common';
import { AuthGuard } from 'src/auth/auth.guard';
import { Role } from 'src/role/role.decorator';
import { RoleGuard } from 'src/role/role.guard';
import { UserMutationOutput } from './dto/user.output';

@Resolver(() => User)
export class UsersResolver {
  constructor(
    private readonly usersService: UsersService, // 환경 변수 접근을 위해 ConfigService Inject
    private readonly config: ConfigService,
  ) {}

  // CRUD Method
  @UseGuards(...[AuthGuard])
  @Query(() => [User], { name: 'users' })
  async find(
    @Args('filterUserInput', { nullable: true })
    filterUserInput: FilterUserInput,
  ) {
    return await this.usersService.find(filterUserInput);
  }
  @UseGuards(...[AuthGuard])
  @Query(() => User, { name: 'user' })
  findOne(
    @Args('userId', { type: () => String })
    userId: string,
  ) {
    return this.usersService.findOne(userId);
  }
  @UseGuards(...[AuthGuard, RoleGuard])
  @Role(ROLE_ADMIN)
  @Mutation(() => UserMutationOutput)
  createUser(
    @Args('createUserInput')
    createUserInput: CreateUserInput,
  ) {
    return this.usersService.create(createUserInput);
  }
  @Mutation(() => UserMutationOutput)
  updateUser(
    @Args('userId', { type: () => String })
    userId: string,
    @Args('updateUserInput')
    updateUserInput: UpdateUserInput,
  ) {
    return this.usersService.update(userId, updateUserInput);
  }
  @UseGuards(...[AuthGuard, RoleGuard])
  @Role(ROLE_ADMIN)
  @Mutation(() => UserMutationOutput)
  deleteUser(
    @Args('userId', { type: () => String })
    userId: string,
  ) {
    return this.usersService.delete(userId);
  }
  @Mutation(() => UserMutationOutput)
  resetPassword(
    @Args('userId', { type: () => String })
    userId: string,
  ) {
    return this.usersService.resetPassword(userId);
  }

  // Login/Logout Method
  @Mutation(() => LoginUserOutput)
  async login(
    @Args('loginUserInput')
    loginUserInput: LoginUserInput,
  ): Promise<LoginUserOutput> {
    return await this.usersService.login(loginUserInput);
  }
  @Mutation(() => LogoutUserOutput)
  logout(
    @Args('userId', { type: () => String })
    userId: string,
  ) {
    return this.usersService.logout(userId);
  }
}
