import { forwardRef, Inject, Injectable } from '@nestjs/common';
import {
  CreateAbnormalInput,
  CreateAbnormalSummaryInput,
  CreateLossAbnormalInput,
} from './dto/create-abnormal.input';
import { PUB_SUB } from 'src/app.provider';
import { PubSub } from 'graphql-subscriptions';
import { InjectModel } from '@nestjs/mongoose';
import { Abnormal, AbnormalSummary } from './entities/abnormal.entity';
import { FilterQuery, Model } from 'mongoose';
import { TOPIC_MONITOR_ABNORMAL } from 'src/pubsub/pubsub.constants';
import {
  FilterAbnormalDetailInput,
  FilterAbnormalInput,
  FilterAbnormalReportInput,
} from './dto/filter-abnormal.input';
import {
  AbnormalDetailOutput,
  AbnormalPaginationOutput,
  AbnormalReportOutput,
  AbnormalSummaryMutationOutput,
  AbnormalSummaryPaginationOutput,
} from './dto/abnormal.output';
import { FilterCommonInput } from 'src/common/dto/filter-common.input';
import { AbnormalMinMax } from 'src/common/dto/common.enum';
import { RawService } from 'src/raw/raw.service';
import { UpdateAbnormalSummaryInput } from './dto/update-abnormal.input';

@Injectable()
export class AbnormalService {
  constructor(
    @Inject(PUB_SUB)
    private readonly pubSub: PubSub,
    @InjectModel(Abnormal.name)
    private readonly abnormalModel: Model<Abnormal>,
    @InjectModel(AbnormalSummary.name)
    private readonly abnormalSummaryModel: Model<AbnormalSummary>,
    @Inject(forwardRef(() => RawService))
    private readonly rawService: RawService,
  ) {}

  async create(createAbnormalInput: CreateAbnormalInput) {
    return await this.abnormalModel.create({
      workshopCode: createAbnormalInput.workshopId,
      lineCode: createAbnormalInput.lineId,
      opCode: createAbnormalInput.opCode,
      machineCode: createAbnormalInput.machineId,
      productNo: createAbnormalInput.productId,
      abnormalCode: createAbnormalInput.abnormalCode,
      abnormalDivision: createAbnormalInput.abnormalDivision,
      abnormalBeginDate: createAbnormalInput.abnormalBeginDate,
      abnormalEndDate: createAbnormalInput.abnormalEndDate,
      abnormalValue: createAbnormalInput.abnormalValue,
      abnormalTool: createAbnormalInput.abnormalTool,
      mainProgramNo: createAbnormalInput.mainProgramNo,
      subProgramNo: createAbnormalInput.subProgramNo,
      tCode: createAbnormalInput.tCode,
      mCode: createAbnormalInput.mCode,
      fov: createAbnormalInput.fov,
      sov: createAbnormalInput.sov,
      offsetX: createAbnormalInput.offsetX,
      offsetZ: createAbnormalInput.offsetZ,
      feed: createAbnormalInput.feed,
    });
  }
  async createLossAbnormal(createLossAbnormalInput: CreateLossAbnormalInput) {
    return await this.abnormalModel.create({
      workshopCode: createLossAbnormalInput.workshopId,
      lineCode: createLossAbnormalInput.lineId,
      opCode: createLossAbnormalInput.opCode,
      machineCode: createLossAbnormalInput.machineId,
      productNo: createLossAbnormalInput.productId,
      abnormalCode: 'AI',
      abnormalDivision: AbnormalMinMax.Max, // Loss는 상한치 이상밖에 없으므로 고정
      abnormalBeginDate: new Date(createLossAbnormalInput.startTime / 1000000),
      abnormalEndDate: new Date(createLossAbnormalInput.endTime / 1000000),
      abnormalValue: createLossAbnormalInput.loss,
      abnormalThreshold: createLossAbnormalInput.threshold,
      abnormalLossCount: createLossAbnormalInput.lossCount,
      mainProgramNo: createLossAbnormalInput.mainProg,
      // subProgramNo: createLossAbnormalInput.subProg,
      tCode: createLossAbnormalInput.tCode,
      mCode: '',
      fov: createLossAbnormalInput.fov,
      sov: createLossAbnormalInput.sov,
      offsetX: createLossAbnormalInput.offsetX,
      offsetZ: createLossAbnormalInput.offsetZ,
      feed: 0,
    });
  }
  async createSummary(createAbnormalSummaryInput: CreateAbnormalSummaryInput) {
    return await this.abnormalSummaryModel.create({
      workshopCode: createAbnormalSummaryInput.workshopId,
      lineCode: createAbnormalSummaryInput.lineId,
      opCode: createAbnormalSummaryInput.opCode,
      machineCode: createAbnormalSummaryInput.machineId,
      productNo: createAbnormalSummaryInput.productId,

      abnormalCt: createAbnormalSummaryInput.abnormalCt,
      abnormalCtValue: createAbnormalSummaryInput.abnormalCtValue,
      abnormalCtThreshold: createAbnormalSummaryInput.abnormalCtThreshold,
      abnormalMinCtThreshold: createAbnormalSummaryInput.abnormalMinCtThreshold,

      abnormalLoad: createAbnormalSummaryInput.abnormalLoad,
      abnormalLoadValue: createAbnormalSummaryInput.abnormalLoadValue,
      abnormalLoadThreshold: createAbnormalSummaryInput.abnormalLoadThreshold,
      abnormalMinLoadThreshold:
        createAbnormalSummaryInput.abnormalMinLoadThreshold,

      abnormalAi: createAbnormalSummaryInput.abnormalAi,
      abnormalAiValue: createAbnormalSummaryInput.abnormalAiValue,
      abnormalAiThreshold: createAbnormalSummaryInput.abnormalAiThreshold,
      abnormalAiCount: createAbnormalSummaryInput.abnormalAiCount,

      abnormalBeginDate: createAbnormalSummaryInput.abnormalBeginDate,
      abnormalEndDate: createAbnormalSummaryInput.abnormalEndDate,

      mainProgramNo: createAbnormalSummaryInput.mainProgramNo,
      subProgramNo: createAbnormalSummaryInput.subProgramNo,
      tCode: createAbnormalSummaryInput.tCode,
      mCode: createAbnormalSummaryInput.mCode,
      fov: createAbnormalSummaryInput.fov,
      sov: createAbnormalSummaryInput.sov,
      offsetX: createAbnormalSummaryInput.offsetX,
      offsetZ: createAbnormalSummaryInput.offsetZ,
      feed: createAbnormalSummaryInput.feed,
    });
  }

  async updateSummary(
    productNo: string,
    updateAbnormalSummaryInput: UpdateAbnormalSummaryInput,
  ): Promise<AbnormalSummaryMutationOutput> {
    const updateResult = await this.abnormalSummaryModel.findOneAndUpdate(
      {
        productNo: productNo,
      },
      {
        ...updateAbnormalSummaryInput,
        updateAt: new Date(),
      },
      {
        returnDocument: 'after',
      },
    );

    if (updateResult) {
      return {
        isSuccess: true,
        abnormalAi: updateResult.abnormalAi,
        abnormalAiValue: updateResult.abnormalAiValue,
      };
    } else {
      return {
        isSuccess: false,
      };
    }
  }

  async find(
    filterAbnormalInput?: FilterAbnormalInput,
  ): Promise<AbnormalPaginationOutput> {
    // Filter가 없을 경우 전체 조회
    if (!filterAbnormalInput) {
      const abnormals = await this.abnormalModel
        .find()
        .sort({ abnormalBeginDate: -1 });
      return {
        pageCount: null,
        abnormals,
      };
    }

    const DEFAULT_SORT_KEY = 'abnormalBeginDate';
    const countPerPage = filterAbnormalInput.count
      ? filterAbnormalInput.count
      : null;
    const skipCount =
      countPerPage && filterAbnormalInput.page
        ? countPerPage * (filterAbnormalInput.page - 1)
        : null;
    let abnormalFilter: FilterQuery<Abnormal> = {};
    const abnormalSort = {};
    if (filterAbnormalInput.sortInput) {
      abnormalSort[filterAbnormalInput.sortInput.sortColumn] =
        filterAbnormalInput.sortInput.sortDirection == 'asc' ? 1 : -1;

      if (filterAbnormalInput.sortInput.sortColumn != DEFAULT_SORT_KEY) {
        abnormalSort[DEFAULT_SORT_KEY] = -1;
      }
    }

    // 조회 데이터 일시가 없을 경우 생산 번호를 이용하여 조회
    if (!filterAbnormalInput.rangeStart && !filterAbnormalInput.rangeEnd) {
      if (filterAbnormalInput.productNo) {
        abnormalFilter = {
          workshopCode: filterAbnormalInput.commonFilter.workshopId,
          lineCode: filterAbnormalInput.commonFilter.lineId,
          opCode: filterAbnormalInput.commonFilter.opCode,
          machineCode: filterAbnormalInput.commonFilter.machineId,
          abnormalCode: filterAbnormalInput.abnormalCode,
          productNo: filterAbnormalInput.productNo,
          $or: this.initRegexFilter(filterAbnormalInput.filterKeyword),
        };
      } else {
        abnormalFilter = {
          workshopCode: filterAbnormalInput.commonFilter.workshopId,
          lineCode: filterAbnormalInput.commonFilter.lineId,
          opCode: filterAbnormalInput.commonFilter.opCode,
          machineCode: filterAbnormalInput.commonFilter.machineId,
          abnormalCode: filterAbnormalInput.abnormalCode,
          // productNo: filterAbnormalInput.filterKeyword
          //   ? `${filterAbnormalInput.filterKeyword}`
          //   : undefined,
          // abnormalDescription: filterAbnormalInput.filterKeyword
          //   ? { $regex: '.*' + filterAbnormalInput.filterKeyword + '.*' }
          //   : undefined,
          $or: this.initRegexFilter(filterAbnormalInput.filterKeyword),
        };
      }
    } else {
      abnormalFilter = {
        workshopCode: filterAbnormalInput.commonFilter.workshopId,
        lineCode: filterAbnormalInput.commonFilter.lineId,
        opCode: filterAbnormalInput.commonFilter.opCode,
        machineCode: filterAbnormalInput.commonFilter.machineId,
        abnormalCode: filterAbnormalInput.abnormalCode,
        // productNo: filterAbnormalInput.productNo,

        abnormalBeginDate: {
          $gte: filterAbnormalInput.rangeStart,
          $lte: filterAbnormalInput.rangeEnd,
        },
        $or: this.initRegexFilter(filterAbnormalInput.filterKeyword),
      };
    }

    if (countPerPage && (skipCount || skipCount == 0)) {
      const countResult =
        await this.abnormalModel.countDocuments(abnormalFilter);
      const abnormals = await this.abnormalModel
        .find(abnormalFilter)
        .sort(abnormalSort)
        .skip(skipCount)
        .limit(countPerPage);

      return {
        pageCount:
          countResult % countPerPage == 0
            ? countResult / countPerPage
            : Math.floor(countResult / countPerPage + 1),
        abnormals,
      };
    }

    const abnormals = await this.abnormalModel
      .find(abnormalFilter)
      .sort(abnormalSort);

    return {
      pageCount: null,
      abnormals,
    };
  }
  async findSummary(
    filterAbnormalInput?: FilterAbnormalInput,
  ): Promise<AbnormalSummaryPaginationOutput> {
    // Filter가 없을 경우 전체 조회
    if (!filterAbnormalInput) {
      const abnormals = await this.abnormalSummaryModel
        .find()
        .sort({ abnormalBeginDate: -1 });
      return {
        pageCount: null,
        abnormals,
      };
    }

    const DEFAULT_SORT_KEY = 'abnormalBeginDate';
    const countPerPage = filterAbnormalInput.count
      ? filterAbnormalInput.count
      : null;
    const skipCount =
      countPerPage && filterAbnormalInput.page
        ? countPerPage * (filterAbnormalInput.page - 1)
        : null;
    let abnormalFilter: FilterQuery<AbnormalSummary> = {};
    const abnormalSort = {};
    if (filterAbnormalInput.sortInput) {
      abnormalSort[filterAbnormalInput.sortInput.sortColumn] =
        filterAbnormalInput.sortInput.sortDirection == 'asc' ? 1 : -1;

      if (filterAbnormalInput.sortInput.sortColumn != DEFAULT_SORT_KEY) {
        abnormalSort[DEFAULT_SORT_KEY] = -1;
      }
    }

    // 조회 데이터 일시가 없을 경우 생산 번호를 이용하여 조회
    if (!filterAbnormalInput.rangeStart && !filterAbnormalInput.rangeEnd) {
      if (filterAbnormalInput.productNo) {
        abnormalFilter = {
          workshopCode: filterAbnormalInput.commonFilter.workshopId,
          lineCode: filterAbnormalInput.commonFilter.lineId,
          opCode: filterAbnormalInput.commonFilter.opCode,
          machineCode: filterAbnormalInput.commonFilter.machineId,
          abnormalCt:
            filterAbnormalInput.abnormalCode == 'CT' ? 'N' : undefined,
          abnormalLoad:
            filterAbnormalInput.abnormalCode == 'LOAD' ? 'N' : undefined,
          abnormalAi:
            filterAbnormalInput.abnormalCode == 'AI' ? 'N' : undefined,
          // abnormalCode: filterAbnormalInput.abnormalCode,
          productNo: filterAbnormalInput.productNo,
          $or: this.initRegexFilter(filterAbnormalInput.filterKeyword),
        };
      } else {
        abnormalFilter = {
          workshopCode: filterAbnormalInput.commonFilter.workshopId,
          lineCode: filterAbnormalInput.commonFilter.lineId,
          opCode: filterAbnormalInput.commonFilter.opCode,
          machineCode: filterAbnormalInput.commonFilter.machineId,
          abnormalCt:
            filterAbnormalInput.abnormalCode == 'CT' ? 'N' : undefined,
          abnormalLoad:
            filterAbnormalInput.abnormalCode == 'LOAD' ? 'N' : undefined,
          abnormalAi:
            filterAbnormalInput.abnormalCode == 'AI' ? 'N' : undefined, // productNo: filterAbnormalInput.filterKeyword
          //   ? `${filterAbnormalInput.filterKeyword}`
          //   : undefined,
          // abnormalDescription: filterAbnormalInput.filterKeyword
          //   ? { $regex: '.*' + filterAbnormalInput.filterKeyword + '.*' }
          //   : undefined,
          $or: this.initRegexFilter(filterAbnormalInput.filterKeyword),
        };
      }
    } else {
      abnormalFilter = {
        workshopCode: filterAbnormalInput.commonFilter.workshopId,
        lineCode: filterAbnormalInput.commonFilter.lineId,
        opCode: filterAbnormalInput.commonFilter.opCode,
        machineCode: filterAbnormalInput.commonFilter.machineId,
        abnormalCt: filterAbnormalInput.abnormalCode == 'CT' ? 'N' : undefined,
        abnormalLoad:
          filterAbnormalInput.abnormalCode == 'LOAD' ? 'N' : undefined,
        abnormalAi: filterAbnormalInput.abnormalCode == 'AI' ? 'N' : undefined, // productNo: filterAbnormalInput.productNo,

        abnormalBeginDate: {
          $gte: filterAbnormalInput.rangeStart,
          $lte: filterAbnormalInput.rangeEnd,
        },
        $or: this.initRegexFilter(filterAbnormalInput.filterKeyword),
      };
    }

    if (countPerPage && (skipCount || skipCount == 0)) {
      const countResult =
        await this.abnormalSummaryModel.countDocuments(abnormalFilter);
      const abnormals = await this.abnormalSummaryModel
        .find(abnormalFilter)
        .sort(abnormalSort)
        .skip(skipCount)
        .limit(countPerPage);

      for (const a of abnormals) {
        const aiAbnormals = await this.find({
          commonFilter: {
            ...filterAbnormalInput.commonFilter,
          },
          productNo: a.productNo,
          abnormalCode: 'AI',
        });

        a.abnormalAiCount = 0;
        a.abnormalAiThreshold = 0;
        a.abnormalAiValue = aiAbnormals.abnormals.length;
      }
      // abnormals.forEach((a) => {
      //   a.abnormalAiCount = 0;
      //   a.abnormalAiThreshold = 0;
      // });

      return {
        pageCount:
          countResult % countPerPage == 0
            ? countResult / countPerPage
            : Math.floor(countResult / countPerPage + 1),
        abnormals,
      };
    }

    const abnormals = await this.abnormalSummaryModel
      .find(abnormalFilter)
      .sort(abnormalSort);

    for (const a of abnormals) {
      const aiAbnormals = await this.find({
        commonFilter: {
          ...filterAbnormalInput.commonFilter,
        },
        productNo: a.productNo,
        abnormalCode: 'AI',
      });

      a.abnormalAiCount = 0;
      a.abnormalAiThreshold = 0;
      a.abnormalAiValue = aiAbnormals.abnormals.length;
    }
    // abnormals.forEach((a) => {
    //   a.abnormalAiCount = 0;
    //   a.abnormalAiThreshold = 0;
    // });

    return {
      pageCount: null,
      abnormals,
    };
  }
  async findDetail(
    filterAbnormalDetailInput: FilterAbnormalDetailInput,
  ): Promise<AbnormalDetailOutput> {
    const output = new AbnormalDetailOutput();

    const currentSummary = await this.abnormalSummaryModel.findOne({
      workshopCode: filterAbnormalDetailInput.workshopId,
      lineCode: filterAbnormalDetailInput.lineId,
      opCode: filterAbnormalDetailInput.opCode,
      machineCode: filterAbnormalDetailInput.machineId,
      productNo: filterAbnormalDetailInput.productNo,
    });
    output.abnormalCt = currentSummary.abnormalCt;
    output.abnormalCtValue = currentSummary.abnormalCtValue;
    output.abnormalCtThreshold = currentSummary.abnormalCtThreshold;
    output.abnormalMinCtThreshold = currentSummary.abnormalMinCtThreshold;
    output.abnormalLoad = currentSummary.abnormalLoad;
    output.abnormalLoadValue = currentSummary.abnormalLoadValue;
    output.abnormalLoadThreshold = currentSummary.abnormalLoadThreshold;
    output.abnormalMinLoadThreshold = currentSummary.abnormalMinLoadThreshold;
    output.abnormalAi = [];
    output.raws = [];

    const currentAbnormalRaw = await this.abnormalModel.find({
      productNo: currentSummary.productNo,
      abnormalCode: 'AI',
    });
    if (currentAbnormalRaw && currentAbnormalRaw.length > 0) {
      output.abnormalAi = currentAbnormalRaw.map((r) => {
        return {
          abnormalAiBeginDate: r.abnormalBeginDate,
          abnormalAiEndDate: r.abnormalEndDate,
          abnormalAiValue: r.abnormalValue,
        };
      });
    }

    const currentRaw = await this.rawService.find(
      {
        commonFilter: {
          workshopId: filterAbnormalDetailInput.workshopId,
          lineId: filterAbnormalDetailInput.lineId,
          opCode: filterAbnormalDetailInput.opCode,
          machineId: filterAbnormalDetailInput.machineId,
        },
        rangeStart: currentSummary.abnormalBeginDate,
        rangeStop: currentSummary.abnormalEndDate,
      },
      ['time', 'Load', 'Predict', 'Loss', 'SV_X_Pos', 'SV_Z_Pos'],
    );
    if (currentRaw && currentRaw.length > 0) {
      output.raws = currentRaw;
    }

    return output;
  }
  async aggregate(filterAbnormalReportInput: FilterAbnormalReportInput) {
    const aggregateResult = await this.abnormalModel.aggregate([
      {
        $match: {
          workshopCode: filterAbnormalReportInput.commonFilter.workshopId,
          lineCode: filterAbnormalReportInput.commonFilter.lineId,
          opCode: filterAbnormalReportInput.commonFilter.opCode,
          machineCode: filterAbnormalReportInput.commonFilter.machineId,
          abnormalBeginDate: {
            $gte: filterAbnormalReportInput.rangeStart,
            $lte: filterAbnormalReportInput.rangeEnd,
          },
        },
      },
      {
        $group: {
          _id: ['$abnormalTool', '$abnormalCode'],
          abnormalCount: {
            $count: {},
          },
        },
      },
      {
        $project: {
          abnormalTool: { $arrayElemAt: ['$_id', 0] },
          abnormalCode: { $arrayElemAt: ['$_id', 1] },
          abnormalCount: '$abnormalCount',
        },
      },
    ]);

    const filteredResult = aggregateResult
      .filter((p) => p.abnormalCode != null)
      .sort((p, q) => {
        if (p.abnormalCount < q.abnormalCount) {
          return 1;
        }

        if (p.abnormalCount == q.abnormalCount) {
          if (p.abnormalTool > q.abnormalTool) {
            return 1;
          }

          if (p.abnormalTool == q.abnormalTool) {
            if (p.abnormalCode > q.abnormalCode) {
              return 1;
            }

            return -1;
          }

          return -1;
        }

        return -1;
      });

    return filteredResult.map((p) => {
      const temp = new AbnormalReportOutput();
      temp.abnormalCode = p.abnormalCode;
      temp.abnormalTool = p.abnormalTool;
      temp.abnormalCount = p.abnormalCount;
      return temp;
    });
  }

  // * Subscription Method
  // 1. 이상 감지 모니터링 등록
  // async monitor() {
  //   // Pub-Sub 토픽 등록
  //   return this.pubSub.asyncIterator(TOPIC_MONITOR_ABNORMAL);
  // }
  async monitor(filterCommonInput: FilterCommonInput) {
    // Pub-Sub 토픽 등록
    const topic = `${filterCommonInput.workshopId}/${filterCommonInput.lineId}/${filterCommonInput.opCode}/${TOPIC_MONITOR_ABNORMAL}`;

    return this.pubSub.asyncIterator(topic);
  }

  initRegexFilter(keyword: string): FilterQuery<Abnormal>[] {
    const result: FilterQuery<Abnormal>[] = [];

    result.push({
      productNo: keyword ? { $regex: `.*${keyword}.*` } : undefined,
    });
    result.push({
      abnormalDescription: keyword ? { $regex: `.*${keyword}.*` } : undefined,
    });

    return result;
  }
}
