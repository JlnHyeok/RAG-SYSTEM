import { Inject, Injectable } from '@nestjs/common';
import { CreateUserInput } from './dto/create-user.input';
import { UpdateUserInput } from './dto/update-user.input';
import { Model } from 'mongoose';
import { User } from './entities/user.entity';
import { InjectModel } from '@nestjs/mongoose';
import * as bcrypt from 'bcryptjs';
import { FilterUserInput } from './dto/filter-user.input';
import { LoginUserInput } from './dto/login-user.input';
import { LoginUserOutput } from './dto/login-user.output';
import { JwtService } from '@nestjs/jwt';
import { LogoutUserOutput } from './dto/logout-user.output';
import { ConfigService } from '@nestjs/config';
import { WorkshopService } from '../workshop/workshop.service';
import { Workshop } from '../workshop/entities/workshop.entity';
import { FilterWorkshopInput } from '../workshop/dto/filter-workshop.input';
import { LineService } from '../line/line.service';
import { OperationService } from '../operation/operation.service';
import { CACHE_MANAGER, Cache } from '@nestjs/cache-manager';
import { ROLE_ADMIN } from 'src/role/role.constants';
import * as crypto from 'crypto';
import { UserMutationOutput } from './dto/user.output';
import { ErrorCode, GetErrorMsg } from 'src/common/dto/common.enum';

const salts = 10;
const reservedIds = [
  'Admin',
  'Administrator',
  'Administration',
  'Root',
  'System',
  'admin',
  'administrator',
  'administration',
  'root',
  'system',
];

@Injectable()
export class UsersService {
  constructor(
    private readonly jwtService: JwtService,
    @InjectModel(User.name)
    private readonly userModel: Model<User>,
    // 환경 변수 접근을 위해 ConfigService Inject
    private readonly config: ConfigService,
    private readonly workshopService: WorkshopService,
    private readonly lineService: LineService,
    private readonly operationService: OperationService,
    @Inject(CACHE_MANAGER)
    private readonly cacheManager: Cache,
  ) {
    // 서비스 생성 시 등록된 정보를 이용하여 Super User 생성
    this.createSuperUser({
      userId: this.config.get<string>('SUPER_USER_ID'),
      userName: 'root',
      userRole: ROLE_ADMIN,
      getEntity: undefined,
    });
  }

  private async createSuperUser(createUserInput: CreateUserInput) {
    // ID 중복 검증
    const currentUser = await this.findOne(createUserInput.userId);

    if (currentUser) {
      return;
    }

    const result = await this.userModel.create({
      userId: createUserInput.userId,
      // password: this.config.get<string>('SUPER_USER_PASSWORD'),
      password: this.hashPassword(
        createUserInput.userId,
        this.config.get<string>('SUPER_USER_PASSWORD'),
      ),
      userName: createUserInput.userName,
      userRole: createUserInput.userRole,
      resetFlag: 'N',
    });
  }

  // CRUD Method
  async create(createUserInput: CreateUserInput): Promise<UserMutationOutput> {
    const output = new UserMutationOutput();

    // Reserved ID 검증
    if (reservedIds.includes(createUserInput.userId)) {
      return {
        isSuccess: false,
        errorCode: 0,
        errorMsg: `사용할 수 없는 ID입니다. (${reservedIds.join(',')})`,
      };
    }

    // ID 중복 검증
    const currentUser = await this.findOne(createUserInput.userId);

    // ID 중복 검증
    if (currentUser) {
      return {
        isSuccess: false,
        errorCode: 0,
        errorMsg: '이미 존재하는 ID입니다',
      };
    }

    // 일반 사용자 생성 시 디폴트 패스워드를 이용하여 생성
    const newUser = await this.userModel.create({
      userId: createUserInput.userId,
      // password: this.config.get<string>('DEFAULT_PASSWORD'),
      password: this.hashPassword(
        createUserInput.userId,
        this.config.get<string>('DEFAULT_PASSWORD'),
      ),
      userName: createUserInput.userName,
      userRole: createUserInput.userRole,
      resetFlag: 'Y',
      createAt: Date.now(),
      updateAt: Date.now(),
    });

    if (newUser) {
      return {
        isSuccess: true,
        userId: newUser.userId,
        password: newUser.password,
        userName: newUser.userName,
        userRole: newUser.userRole,
        resetFlag: newUser.resetFlag,
        createAt: newUser.createAt,
        updateAt: newUser.updateAt,
      };
    }

    output.isSuccess = false;
    output.errorCode = ErrorCode.ETC;
    output.errorMsg = GetErrorMsg(ErrorCode.ETC);

    return output;
  }

  async find(filterUserInput: FilterUserInput): Promise<User[]> {
    if (filterUserInput) {
      return await this.userModel.find({
        userId: { $in: filterUserInput.userIds },
      });
    }

    return await this.userModel.find();
  }

  async findOne(userId: string): Promise<User> {
    return await this.userModel.findOne({ userId });
  }

  async update(
    userId: string,
    updateUserInput: UpdateUserInput,
  ): Promise<UserMutationOutput> {
    const updateResult = await this.userModel.findOneAndUpdate(
      { userId },
      {
        // password: updateUserInput.password
        //   ? await bcrypt.hash(updateUserInput.password, salts)
        //   : updateUserInput.password,
        password: updateUserInput.password
          ? this.hashPassword(userId, updateUserInput.password)
          : updateUserInput.password,
        userName: updateUserInput.userName,
        userRole: updateUserInput.userRole,
        resetFlag: updateUserInput.password ? 'N' : undefined,
        updateAt: Date.now(),
      },
      {
        returnDocument: 'after',
      },
    );

    if (updateResult) {
      return {
        isSuccess: true,
        userId: updateResult.userId,
        password: updateResult.password,
        userName: updateResult.userName,
        userRole: updateResult.userRole,
        createAt: updateResult.createAt,
        updateAt: updateResult.updateAt,
      };
    }

    return {
      isSuccess: false,
      errorCode: ErrorCode.ETC,
      errorMsg: GetErrorMsg(ErrorCode.ETC),
    };
  }

  async resetPassword(userId: string): Promise<UserMutationOutput> {
    const updateResult = await this.userModel.findOneAndUpdate(
      { userId },
      {
        // password: this.config.get<string>('DEFAULT_PASSWORD'),
        password: this.hashPassword(
          userId,
          this.config.get<string>('DEFAULT_PASSWORD'),
        ),
        resetFlag: 'Y',
        updateAt: Date.now(),
      },
      {
        returnDocument: 'after',
      },
    );

    if (updateResult) {
      return {
        isSuccess: true,
        userId: updateResult.userId,
        password: updateResult.password,
        userName: updateResult.userName,
        userRole: updateResult.userRole,
        createAt: updateResult.createAt,
        updateAt: updateResult.updateAt,
      };
    }

    return {
      isSuccess: false,
      errorCode: ErrorCode.ETC,
      errorMsg: GetErrorMsg(ErrorCode.ETC),
    };
  }

  async delete(userId: string): Promise<UserMutationOutput> {
    const deleteResult = await this.userModel.findOneAndDelete(
      { userId },
      {
        returnDocument: 'before',
      },
    );

    if (deleteResult) {
      return {
        isSuccess: true,
        userId: deleteResult.userId,
        password: deleteResult.password,
        userName: deleteResult.userName,
        userRole: deleteResult.userRole,
        createAt: deleteResult.createAt,
        updateAt: deleteResult.updateAt,
      };
    }

    return {
      isSuccess: false,
      errorCode: ErrorCode.ETC,
      errorMsg: GetErrorMsg(ErrorCode.ETC),
    };
  }

  async login(loginUserInput: LoginUserInput) {
    const loginOutput: LoginUserOutput = { isSuccess: false };
    const currentUser = await this.findOne(loginUserInput.userId);

    // ID 유효성 검증
    if (!currentUser) {
      return loginOutput;
    }

    // P/W 유효성 검증
    // if (
    //   !(await bcrypt.compare(loginUserInput.password, currentUser.password))
    // ) {
    //   return loginOutput;
    // }
    const hashedPassword = this.hashPassword(
      loginUserInput.userId,
      loginUserInput.password,
    );
    if (hashedPassword != currentUser.password) {
      return loginOutput;
    }

    // JWT 발급
    const accessToken = await this.jwtService.signAsync({
      userId: currentUser.userId,
      userName: currentUser.userName,
      userRole: currentUser.userRole,
    });

    // JWT 캐시 저장
    await this.cacheManager.set(currentUser.userId, accessToken);
    const newToken = await this.cacheManager.get(currentUser.userId);

    loginOutput.isSuccess = true;
    loginOutput.loginDate = new Date(Date.now());
    loginOutput.accessToken = accessToken;
    loginOutput.resetFlag = currentUser.resetFlag;

    return loginOutput;
  }

  logout(userId: string): LogoutUserOutput {
    return {
      isSuccess: true,
      logoutDate: new Date(Date.now()),
    };
  }

  hashPassword(userId: string, password: string): string {
    const saltBuffer = new Buffer(`${userId}_${password}`).toString();
    // const hashed = crypto
    //   .pbkdf2Sync(password, saltBuffer, 1, 32, 'sha256')
    //   .toString('hex');
    const hashed = crypto
      .createHash('sha256')
      .update(`${password}${saltBuffer}`)
      .digest('hex');

    return hashed;
  }
}
