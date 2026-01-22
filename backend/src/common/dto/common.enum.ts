import { registerEnumType } from '@nestjs/graphql';

export enum ErrorCode {
  DUPLICATED,
  ETC,
}
export enum PeriodType {
  Yearly,
  Monthly,
  Weekly,
  Daily,
}
export enum AbnormalMinMax {
  Min,
  Max,
}
registerEnumType(ErrorCode, { name: 'ErrorCode' });
registerEnumType(PeriodType, { name: 'PeriodType' });
registerEnumType(AbnormalMinMax, { name: 'AbnormalMinMax' });

export function GetErrorMsg(code: ErrorCode) {
  if (code == ErrorCode.DUPLICATED) {
    return '이미 등록된 항목이 있습니다';
  } else {
    return '요청 처리 중 오류가 발생하였습니다';
  }
}
