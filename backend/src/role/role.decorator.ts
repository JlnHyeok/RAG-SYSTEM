import { SetMetadata } from '@nestjs/common';

export const ROLES_KEY = 'role';
export const Role = (minimumRole: number) =>
  SetMetadata(ROLES_KEY, minimumRole);
