"""
MongoDB 커넥터
생산 이력, 이상감지, 설비, 공구 정보를 조회하기 위한 커넥터입니다.
에이전트가 하이브리드 RAG에서 실시간 컨텍스트를 가져올 때 사용합니다.

Backend MongoDB 컬렉션:
- products: 생산 이력 (productNo, ct, loadSum, productResult)
- abnormals: 이상감지 이력 (abnormalCode, abnormalValue, abnormalTool)
- machines: 설비 마스터 (machineCode, machineName, opCode)
- tools: 공구 마스터 (toolCode, toolName, maxCount)
- thresholds: 임계치 설정 (maxThresholdCt, maxThresholdLoad)
"""

import logging
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from datetime import datetime, timedelta
from motor.motor_asyncio import AsyncIOMotorClient
from app.core.config import settings

logger = logging.getLogger(__name__)


@dataclass
class FilterCommon:
    """
    공통 필터 - Backend의 FilterCommonInput과 동일한 구조
    (InfluxDB 커넥터와 동일한 구조 재사용)
    """
    workshop_id: str
    line_id: str
    op_code: str
    machine_id: Optional[str] = None
    
    def to_mongo_filter(self) -> Dict[str, str]:
        """MongoDB 쿼리용 필터 딕셔너리 생성"""
        # [FIX] 사용자 데이터 확인 결과 CamelCase 사용됨 (w001, F01 등)
        filter_dict = {
            "workshopCode": self.workshop_id,
            "lineCode": self.line_id,
            "opCode": self.op_code,
        }
        if self.machine_id:
            filter_dict["machineCode"] = self.machine_id
        return filter_dict
    
    def to_dict(self) -> Dict[str, str]:
        """일반 딕셔너리 변환"""
        return {
            "workshop_id": self.workshop_id,
            "line_id": self.line_id,
            "op_code": self.op_code,
            "machine_id": self.machine_id or ""
        }
    
    @property
    def did(self) -> Optional[str]:
        """InfluxDB 조회용 did 태그 생성 (format: workshopId_lineId_opCode_machineId)"""
        if self.machine_id:
            return f"{self.workshop_id}_{self.line_id}_{self.op_code}_{self.machine_id}"
        return None


class MongoDBConnector:
    """MongoDB 비동기 커넥터 - 생산/이상감지/설비/공구 데이터 조회"""
    
    def __init__(self, uri: Optional[str] = None):
        """
        Args:
            uri: MongoDB 연결 URI (미지정 시 환경변수에서 조합)
        """
        # URI 생성 (분리된 환경변수에서 조합)
        if uri:
            self.uri = uri
        elif settings.MONGODB_USER and settings.MONGODB_PASSWORD:
            # 인증 정보가 있으면 조합
            base_url = settings.MONGODB_URL  # mongodb://host:port
            db_name = settings.MONGODB_DB_NAME
            user = settings.MONGODB_USER
            password = settings.MONGODB_PASSWORD
            # mongodb://user:password@host:port/db?authSource=admin
            host_part = base_url.replace("mongodb://", "")
            self.uri = f"mongodb://{user}:{password}@{host_part}/{db_name}?authSource=admin"
        else:
            # 인증 없이 연결
            self.uri = settings.MONGODB_URL
        
        self.client: Optional[AsyncIOMotorClient] = None
        self.db = None
        self._initialized = False
        
        # 컬렉션명 설정
        self.col_products = settings.MONGODB_COLLECTION_PRODUCTS
        self.col_abnormals = settings.MONGODB_COLLECTION_ABNORMALS
        self.col_machines = settings.MONGODB_COLLECTION_MACHINES
        self.col_tools = settings.MONGODB_COLLECTION_TOOLS
        self.col_thresholds = settings.MONGODB_COLLECTION_THRESHOLDS
        self.col_lines = settings.MONGODB_COLLECTION_LINES
        self.col_operations = settings.MONGODB_COLLECTION_OPERATIONS
        self.col_workshops = settings.MONGODB_COLLECTION_WORKSHOPS
    
    async def initialize(self, db_name: Optional[str] = None):
        """MongoDB 연결 초기화"""
        if self._initialized:
            return
        
        db_name = db_name or settings.MONGODB_DB_NAME
        
        try:
            self.client = AsyncIOMotorClient(self.uri)
            self.db = self.client[db_name]
            # 연결 테스트
            await self.client.admin.command('ping')
            self._initialized = True
            logger.info(f"MongoDB 연결 성공: {db_name}")
        except Exception as e:
            logger.error(f"MongoDB 연결 실패: {e}")
            raise
    
    async def close(self):
        """연결 종료"""
        if self.client:
            self.client.close()
            self._initialized = False
            logger.info("MongoDB 연결 종료")
    
    # ============ 생산 이력 조회 ============
    
    async def get_recent_products(
        self, 
        filter_common: FilterCommon,
        hours: int = 24,
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """
        최근 N시간 생산 이력 조회
        
        Args:
            filter_common: 공통 필터
            hours: 조회 시간 범위
            limit: 최대 결과 수
            
        Returns:
            생산 이력 목록
        """
        if not self._initialized:
            await self.initialize()
        
        cutoff_time = datetime.utcnow() - timedelta(hours=hours)
        query = {
            **filter_common.to_mongo_filter(),
            "startTime": {"$gte": cutoff_time}
        }
        
        try:
            cursor = self.db[self.col_products].find(query).sort("startTime", -1).limit(limit)
            products = await cursor.to_list(length=limit)
            logger.info(f"생산 이력 조회: {len(products)}건")
            return self._serialize_docs(products)
        except Exception as e:
            logger.error(f"생산 이력 조회 실패: {e}")
            return []
    
    async def get_product_by_no(self, product_no: str) -> Optional[Dict[str, Any]]:
        """생산 번호로 상세 조회"""
        if not self._initialized:
            await self.initialize()
        
        try:
            product = await self.db[self.col_products].find_one({"productNo": product_no})
            return self._serialize_doc(product) if product else None
        except Exception as e:
            logger.error(f"생산 상세 조회 실패: {e}")
            return None
    
    async def get_last_product(self, filter_common: FilterCommon) -> Optional[Dict[str, Any]]:
        """최근 생산 이력 1건 조회"""
        if not self._initialized:
            await self.initialize()
        
        try:
            cursor = self.db[self.col_products].find(
                filter_common.to_mongo_filter()
            ).sort("startTime", -1).limit(1)
            products = await cursor.to_list(length=1)
            return self._serialize_doc(products[0]) if products else None
        except Exception as e:
            logger.error(f"최근 생산 조회 실패: {e}")
            return None
    
    async def get_today_production_count(self, filter_common: FilterCommon) -> int:
        """오늘 생산 수량 집계"""
        if not self._initialized:
            await self.initialize()
        
        today = datetime.utcnow().replace(hour=0, minute=0, second=0, microsecond=0)
        tomorrow = today + timedelta(days=1)
        
        try:
            count = await self.db[self.col_products].count_documents({
                **filter_common.to_mongo_filter(),
                "endTime": {"$gte": today, "$lt": tomorrow}
            })
            return count
        except Exception as e:
            logger.error(f"생산 수량 집계 실패: {e}")
            return 0
    
    async def get_product_stats(
        self, 
        filter_common: FilterCommon, 
        hours: int = 24
    ) -> Dict[str, Any]:
        """
        생산 통계 조회 (CT, LoadSum 평균/최대/최소)
        """
        if not self._initialized:
            await self.initialize()
        
        cutoff_time = datetime.utcnow() - timedelta(hours=hours)
        
        try:
            pipeline = [
                {
                    "$match": {
                        **filter_common.to_mongo_filter(),
                        "startTime": {"$gte": cutoff_time}
                    }
                },
                {
                    "$group": {
                        "_id": None,
                        "count": {"$sum": 1},
                        "avgCt": {"$avg": "$ct"},
                        "maxCt": {"$max": "$ct"},
                        "minCt": {"$min": "$ct"},
                        "avgLoadSum": {"$avg": "$loadSum"},
                        "maxLoadSum": {"$max": "$loadSum"},
                        "minLoadSum": {"$min": "$loadSum"},
                    }
                }
            ]
            
            cursor = self.db[self.col_products].aggregate(pipeline)
            results = await cursor.to_list(length=1)
            
            if results:
                stats = results[0]
                # CT는 ns 단위이므로 초 단위로 변환
                return {
                    "filter": filter_common.to_dict(),
                    "hours": hours,
                    "count": stats.get("count", 0),
                    "ct": {
                        "avg": stats.get("avgCt", 0) / 1e9 if stats.get("avgCt") else None,
                        "max": stats.get("maxCt", 0) / 1e9 if stats.get("maxCt") else None,
                        "min": stats.get("minCt", 0) / 1e9 if stats.get("minCt") else None,
                    },
                    "loadSum": {
                        "avg": stats.get("avgLoadSum"),
                        "max": stats.get("maxLoadSum"),
                        "min": stats.get("minLoadSum"),
                    }
                }
            return {"filter": filter_common.to_dict(), "hours": hours, "count": 0}
        except Exception as e:
            logger.error(f"생산 통계 조회 실패: {e}")
            return {"error": str(e)}
    
    # ============ 이상감지 이력 조회 ============
    
    async def get_recent_abnormals(
        self, 
        filter_common: FilterCommon,
        hours: int = 24,
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """
        최근 N시간 이상감지 이력 조회
        """
        if not self._initialized:
            await self.initialize()
        
        cutoff_time = datetime.utcnow() - timedelta(hours=hours)
        query = {
            **filter_common.to_mongo_filter(),
            "abnormalBeginDate": {"$gte": cutoff_time}
        }
        
        logger.info(f"🔍 [MongoDB Query] 이상감지 조회 쿼리: {query}")
        
        try:
            cursor = self.db[self.col_abnormals].find(query).sort("abnormalBeginDate", -1).limit(limit)
            abnormals = await cursor.to_list(length=limit)
            logger.info(f"✅ [MongoDB Result] 이상감지 이력 조회 결과: {len(abnormals)}건")
            return self._serialize_docs(abnormals)
        except Exception as e:
            logger.error(f"이상감지 이력 조회 실패: {e}")
            return []
    
    async def get_abnormals_by_code(
        self, 
        filter_common: FilterCommon,
        abnormal_code: str,
        hours: int = 24
    ) -> List[Dict[str, Any]]:
        """
        이상감지 유형별 조회 (AI, CT, LoadSum 등)
        """
        if not self._initialized:
            await self.initialize()
        
        cutoff_time = datetime.utcnow() - timedelta(hours=hours)
        query = {
            **filter_common.to_mongo_filter(),
            "abnormalCode": abnormal_code,
            "abnormalBeginDate": {"$gte": cutoff_time}
        }
        
        try:
            cursor = self.db[self.col_abnormals].find(query).sort("abnormalBeginDate", -1)
            return self._serialize_docs(await cursor.to_list(length=100))
        except Exception as e:
            logger.error(f"이상감지 유형별 조회 실패: {e}")
            return []
    
    async def get_abnormal_summary(
        self, 
        filter_common: FilterCommon, 
        hours: int = 24
    ) -> Dict[str, Any]:
        """
        이상감지 요약 (유형별 발생 건수)
        """
        if not self._initialized:
            await self.initialize()
        
        cutoff_time = datetime.utcnow() - timedelta(hours=hours)
        
        try:
            pipeline = [
                {
                    "$match": {
                        **filter_common.to_mongo_filter(),
                        "abnormalBeginDate": {"$gte": cutoff_time}
                    }
                },
                {
                    "$group": {
                        "_id": "$abnormalCode",
                        "count": {"$sum": 1}
                    }
                }
            ]
            
            cursor = self.db[self.col_abnormals].aggregate(pipeline)
            results = await cursor.to_list(length=100)
            
            summary = {
                "filter": filter_common.to_dict(),
                "hours": hours,
                "by_code": {r["_id"]: r["count"] for r in results},
                "total": sum(r["count"] for r in results)
            }
            return summary
        except Exception as e:
            logger.error(f"이상감지 요약 조회 실패: {e}")
            return {"error": str(e)}
    
    # ============ 설비 마스터 조회 ============
    
    async def get_machine_by_code(self, machine_code: str) -> Optional[Dict[str, Any]]:
        """설비 코드로 상세 조회"""
        if not self._initialized:
            await self.initialize()
        
        try:
            machine = await self.db[self.col_machines].find_one({"machineCode": machine_code})
            return self._serialize_doc(machine) if machine else None
        except Exception as e:
            logger.error(f"설비 조회 실패: {e}")
            return None
    
    async def get_machines_by_filter(
        self, 
        workshop_code: Optional[str] = None,
        line_code: Optional[str] = None,
        op_code: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """필터별 설비 목록 조회"""
        if not self._initialized:
            await self.initialize()
        
        query = {}
        if workshop_code:
            query["workshopCode"] = workshop_code
        if line_code:
            query["lineCode"] = line_code
        if op_code:
            query["opCode"] = op_code
        
        try:
            cursor = self.db[self.col_machines].find(query)
            return self._serialize_docs(await cursor.to_list(length=100))
        except Exception as e:
            logger.error(f"설비 목록 조회 실패: {e}")
            return []
    
    async def get_all_machines(self) -> List[Dict[str, Any]]:
        """전체 설비 목록 조회"""
        if not self._initialized:
            await self.initialize()
        
        try:
            cursor = self.db[self.col_machines].find()
            return self._serialize_docs(await cursor.to_list(length=500))
        except Exception as e:
            logger.error(f"전체 설비 조회 실패: {e}")
            return []
    
    # ============ 공구 마스터 조회 ============
    
    async def get_tools_by_machine(self, machine_code: str) -> List[Dict[str, Any]]:
        """설비별 공구 목록 조회"""
        if not self._initialized:
            await self.initialize()
        
        try:
            cursor = self.db[self.col_tools].find({"machineCode": machine_code}).sort("toolOrder", 1)
            return self._serialize_docs(await cursor.to_list(length=50))
        except Exception as e:
            logger.error(f"공구 목록 조회 실패: {e}")
            return []
    
    async def get_tool_by_code(
        self, 
        machine_code: str, 
        tool_code: str
    ) -> Optional[Dict[str, Any]]:
        """공구 코드로 상세 조회"""
        if not self._initialized:
            await self.initialize()
        
        try:
            tool = await self.db[self.col_tools].find_one({
                "machineCode": machine_code,
                "toolCode": tool_code
            })
            return self._serialize_doc(tool) if tool else None
        except Exception as e:
            logger.error(f"공구 조회 실패: {e}")
            return None
    
    # ============ 임계치 조회 ============
    
    async def get_threshold_by_machine(self, machine_code: str) -> Optional[Dict[str, Any]]:
        """설비별 임계치 설정 조회"""
        if not self._initialized:
            await self.initialize()
        
        try:
            threshold = await self.db[self.col_thresholds].find_one({"machineCode": machine_code})
            return self._serialize_doc(threshold) if threshold else None
        except Exception as e:
            logger.error(f"임계치 조회 실패: {e}")
            return None
    
    # ============ 마스터 데이터 조회 ============
    
    async def get_workshops(self) -> List[Dict[str, Any]]:
        """공장 목록 조회"""
        if not self._initialized:
            await self.initialize()
        
        try:
            cursor = self.db[self.col_workshops].find()
            return self._serialize_docs(await cursor.to_list(length=100))
        except Exception as e:
            logger.error(f"공장 목록 조회 실패: {e}")
            return []
    
    async def get_lines(self, workshop_code: Optional[str] = None) -> List[Dict[str, Any]]:
        """라인 목록 조회"""
        if not self._initialized:
            await self.initialize()
        
        query = {"workshopCode": workshop_code} if workshop_code else {}
        
        try:
            cursor = self.db[self.col_lines].find(query)
            return self._serialize_docs(await cursor.to_list(length=100))
        except Exception as e:
            logger.error(f"라인 목록 조회 실패: {e}")
            return []
    
    async def get_operations(
        self, 
        workshop_code: Optional[str] = None,
        line_code: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """공정 목록 조회"""
        if not self._initialized:
            await self.initialize()
        
        query = {}
        if workshop_code:
            query["workshopCode"] = workshop_code
        if line_code:
            query["lineCode"] = line_code
        
        try:
            cursor = self.db[self.col_operations].find(query)
            return self._serialize_docs(await cursor.to_list(length=100))
        except Exception as e:
            logger.error(f"공정 목록 조회 실패: {e}")
            return []
    
    # ============ 헬스체크 ============
    
    async def health_check(self) -> Dict[str, Any]:
        """MongoDB 연결 상태 확인"""
        try:
            if not self._initialized:
                return {"status": "disconnected", "error": "Not initialized"}
            
            await self.client.admin.command('ping')
            return {"status": "connected", "uri": self.uri[:30] + "..."}
        except Exception as e:
            return {"status": "error", "error": str(e)}
    
    # ============ 공구 사용량 계산 ============
    
    async def get_last_tool_change(self, machine_code: str, tool_code: str) -> Optional[Dict]:
        """
        마지막 공구 교체 정보 조회
        """
        if not self._initialized:
            await self.initialize()
        
        try:
            collection = self.db[settings.MONGODB_COLLECTION_TOOL_CHANGE]
            cursor = collection.find({
                "machineCode": machine_code,
                "toolCode": tool_code
            }).sort("changeDate", -1).limit(1)
            
            results = await cursor.to_list(length=1)
            return self._serialize_doc(results[0]) if results else None
        except Exception as e:
            logger.error(f"마지막 공구 교체 조회 실패: {e}")
            return None

    async def get_tool_use_count(
        self, 
        filter_common: FilterCommon,
        tool_code: str
    ) -> int:
        """
        공구 사용량 계산 (마지막 교체 이후 toolHistory 개수)
        """
        if not self._initialized:
            await self.initialize()
        
        try:
            # 1. 마지막 교체 일시 조회
            last_change = await self.get_last_tool_change(
                filter_common.machine_id, 
                tool_code
            )
            
            since_date = datetime(1970, 1, 1)  # 기본값: 1970년 (전체 이력)
            if last_change and last_change.get("changeDate"):
                since_date = last_change["changeDate"]
            
            # 2. 교체 이후 toolHistory 개수 조회
            collection = self.db[settings.MONGODB_COLLECTION_TOOL_HISTORY]
            count = await collection.count_documents({
                "workshopCode": filter_common.workshop_id,
                "lineCode": filter_common.line_id,
                "opCode": filter_common.op_code,
                "machineCode": filter_common.machine_id,
                "toolCode": tool_code,
                "toolUseStartDate": {"$gte": since_date}
            })
            
            return count
        except Exception as e:
            logger.error(f"공구 사용량 계산 실패: {e}")
            return 0

    async def get_current_tool_counts(
        self, 
        filter_common: FilterCommon
    ) -> List[Dict]:
        """
        해당 설비의 모든 공구 현재 사용량 조회
        """
        if not self._initialized:
            await self.initialize()
        
        logger.info(f" [ToolDebug] get_current_tool_counts 호출 - Machine: {filter_common.machine_id}")
        
        try:
            # 1. 해당 설비의 공구 목록 조회
            tools = await self.get_tools_by_machine(filter_common.machine_id)
            if not tools:
                logger.warning(f" [ToolDebug] 해당 설비({filter_common.machine_id})에 등록된 공구가 없습니다.")
                return []
            
            # 2. 각 공구별 사용량 계산
            results = []
            for tool in tools:
                tool_code = tool.get("toolCode")
                if not tool_code:
                    continue
                
                use_count = await self.get_tool_use_count(filter_common, tool_code)
                logger.info(f" [ToolDebug] {tool_code}: {use_count}회")
                max_count = tool.get("maxCount", 0)
                warn_rate = tool.get("warnRate", 90)
                
                # 상태 계산
                if max_count > 0:
                    usage_rate = (use_count / max_count) * 100
                    if usage_rate >= 100:
                        status = "ERROR"
                    elif usage_rate >= warn_rate:
                        status = "WARN"
                    else:
                        status = "OK"
                else:
                    usage_rate = 0
                    status = "OK"
                
                results.append({
                    "toolCode": tool_code,
                    "toolName": tool.get("toolName", ""),
                    "useCount": use_count,
                    "maxCount": max_count,
                    "usageRate": round(usage_rate, 1),
                    "status": status
                })
            
            return results
        except Exception as e:
            logger.error(f"공구 사용량 목록 조회 실패: {e}")
            return []
    
    # ============ 유틸리티 ============
    
    def _serialize_doc(self, doc: Dict) -> Dict:
        """MongoDB 문서 직렬화 (ObjectId 처리)"""
        if doc is None:
            return None
        result = dict(doc)
        if "_id" in result:
            result["_id"] = str(result["_id"])
        return result
    
    def _serialize_docs(self, docs: List[Dict]) -> List[Dict]:
        """MongoDB 문서 목록 직렬화"""
        return [self._serialize_doc(doc) for doc in docs]


# 싱글톤 인스턴스
_mongodb_connector: Optional[MongoDBConnector] = None


def get_mongodb_connector() -> MongoDBConnector:
    """MongoDB 커넥터 싱글톤 인스턴스 반환"""
    global _mongodb_connector
    if _mongodb_connector is None:
        _mongodb_connector = MongoDBConnector()
    return _mongodb_connector
