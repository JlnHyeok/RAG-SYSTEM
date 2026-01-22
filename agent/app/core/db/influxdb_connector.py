"""
InfluxDB 커넥터
제조 현장의 시계열 센서 데이터(부하, 속도, 공구 상태 등)를 조회하기 위한 커넥터입니다.
에이전트가 하이브리드 RAG에서 실시간 센서 컨텍스트를 가져올 때 사용합니다.

Backend의 InfluxDB 구조:
- Measurement: raw, product, tool_history
- 태그: did (=workshop_line_op_machine), TCode, ProductId
- 필드: Load, Feed, Fov, Sov, Run, TCount1-4 등
"""

import logging
import asyncio
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from datetime import datetime, timedelta
from influxdb_client import InfluxDBClient
from influxdb_client.client.flux_table import FluxTable
from app.core.config import settings

logger = logging.getLogger(__name__)


@dataclass
class FilterCommon:
    """
    공통 필터 - Backend의 FilterCommonInput과 동일한 구조
    
    Args:
        workshop_id: 공장 코드 (예: "WS001")
        line_id: 라인 코드 (예: "L01")
        op_code: 공정 코드 (예: "OP10")
        machine_id: 설비 코드 (예: "CNC-001") - 선택
    """
    workshop_id: str
    line_id: str
    op_code: str
    machine_id: Optional[str] = None
    
    @property
    def did(self) -> str:
        """did 태그 생성 (Backend 형식과 동일)"""
        machine = self.machine_id or ""
        return f"{self.workshop_id}_{self.line_id}_{self.op_code}_{machine}"
    
    def to_dict(self) -> Dict[str, str]:
        """딕셔너리 변환"""
        return {
            "workshop_id": self.workshop_id,
            "line_id": self.line_id,
            "op_code": self.op_code,
            "machine_id": self.machine_id or ""
        }


class InfluxDBConnector:
    """InfluxDB 커넥터 - 제조 현장 시계열 센서 데이터 조회"""
    
    # Raw 데이터 필드 목록 (Backend와 동일)
    RAW_FIELDS = [
        "Load",      # 스핀들 부하
        "Feed",      # 이송 속도
        "Fov",       # 이송 속도 오버라이드 (%)
        "Sov",       # 스핀들 속도 오버라이드 (%)
        "Run",       # 가동 상태
        "TCount1", "TCount2", "TCount3", "TCount4",  # 공구 사용 수량
        "SV_X_Pos", "SV_Z_Pos",  # 상대 좌표
        "SV_X_Offset", "SV_Z_Offset",  # 공구 오프셋
        "Loss",      # 오차율
        "Predict"    # AI 예측값
    ]
    
    def __init__(
        self, 
        url: Optional[str] = None,
        token: Optional[str] = None,
        org: Optional[str] = None,
        bucket: Optional[str] = None
    ):
        """
        Args:
            url: InfluxDB URL
            token: 인증 토큰
            org: 조직 이름
            bucket: 기본 버킷 이름
        """
        self.url = url or settings.INFLUXDB_URL
        self.token = token or settings.INFLUXDB_TOKEN
        self.org = org or settings.INFLUXDB_ORG
        self.bucket = bucket or settings.INFLUXDB_BUCKET
        self.measurement_raw = settings.INFLUXDB_MEASUREMENT_RAW
        self.measurement_product = settings.INFLUXDB_MEASUREMENT_PRODUCT
        self.client: Optional[InfluxDBClient] = None
        self.query_api = None
        self._initialized = False
    
    async def initialize(self):
        """InfluxDB 연결 초기화"""
        if self._initialized:
            return
        
        try:
            self.client = InfluxDBClient(
                url=self.url,
                token=self.token,
                org=self.org,
                timeout=60000  # Backend와 동일한 timeout
            )
            self.query_api = self.client.query_api()
            # 연결 테스트
            health = self.client.health()
            if health.status == "pass":
                self._initialized = True
                logger.info(f"InfluxDB 연결 성공: {self.url}")
            else:
                raise Exception(f"InfluxDB health check failed: {health.message}")
        except Exception as e:
            logger.error(f"InfluxDB 연결 실패: {e}")
            raise
    
    def close(self):
        """연결 종료"""
        if self.client:
            self.client.close()
            self._initialized = False
            logger.info("InfluxDB 연결 종료")
    
    def _execute_query(self, query: str) -> List[Dict[str, Any]]:
        """Flux 쿼리 실행 및 결과 변환"""
        if not self._initialized or not self.client:
            raise Exception("InfluxDB not initialized")
        
        query_api = self.client.query_api()
        
        # 디버깅을 위한 쿼리 로그 (사용자 요청)
        logger.info(f"Flux Query Execution:\n{query.strip()}")
        
        tables: List[FluxTable] = query_api.query(query)
        
        results = []
        for table in tables:
            for record in table.records:
                result = {
                    "time": record.get_time(),
                    "field": record.get_field(),
                    "value": record.get_value(),
                }
                # 태그 추가
                for key, val in record.values.items():
                    if key not in ['_time', '_value', '_field', '_measurement', 'result', 'table', '_start', '_stop']:
                        result[key] = val
                results.append(result)
        
        return results
    
    def _build_did_filter(self, filter_common: FilterCommon) -> str:
        """did 태그 필터 생성 (Regex 또는 Exact Match)"""
        # machine_id가 없으면 해당 공정의 모든 설비 조회 (Regex)
        if not filter_common.machine_id:
            # workshop_line_op_.*
            base = f"{filter_common.workshop_id}_{filter_common.line_id}_{filter_common.op_code}_"
            return f'r["did"] =~ /^{base}.*/'
        
        # machine_id가 있으면 정확히 매칭
        return f'r["did"] == "{filter_common.did}"'
    
    # ============ Raw 센서 데이터 조회 ============
    
    async def get_latest_raw(
        self, 
        filter_common: FilterCommon,
        fields: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        최신 Raw 센서 데이터 조회
        
        Args:
            filter_common: 공통 필터 (공장/라인/공정/설비)
            fields: 조회할 필드 목록 (None이면 전체)
            
        Returns:
            최신 센서 데이터 딕셔너리
        """
        if not self._initialized:
            await self.initialize()
        
        field_filter = ""
        if fields:
            field_conditions = " or ".join([f'r["_field"] == "{f}"' for f in fields])
            field_filter = f'|> filter(fn: (r) => {field_conditions})'
        
        query = f'''
        from(bucket: "{self.bucket}")
          |> range(start: -1h)
          |> filter(fn: (r) => r["_measurement"] == "{self.measurement_raw}")
          |> filter(fn: (r) => {self._build_did_filter(filter_common)})
          {field_filter}
          |> group(columns: ["_field"])
          |> last()
        '''
        
        try:
            results = self._execute_query(query)
            # 필드별로 정리
            sensor_data = {
                "filter": filter_common.to_dict(),
                "data": {},
                "query_time": datetime.utcnow().isoformat()
            }
            for r in results:
                field = r.get("field", "unknown")
                sensor_data["data"][field] = {
                    "value": r.get("value"),
                    "time": r.get("time")
                }
            return sensor_data
        except Exception as e:
            logger.error(f"최신 Raw 조회 실패: {e}")
            return {"error": str(e)}
    
    async def get_raw_data(
        self, 
        filter_common: FilterCommon,
        hours: int = 1,
        fields: Optional[List[str]] = None,
        aggregate_interval: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Raw 센서 데이터 시계열 조회
        
        Args:
            filter_common: 공통 필터
            hours: 조회 시간 범위 (시간 단위)
            fields: 조회할 필드 목록
            aggregate_interval: 집계 간격 (예: "1m", "5m", "1h")
            
        Returns:
            시계열 센서 데이터 리스트
        """
        if not self._initialized:
            await self.initialize()
        
        field_filter = ""
        if fields:
            field_conditions = " or ".join([f'r["_field"] == "{f}"' for f in fields])
            field_filter = f'|> filter(fn: (r) => {field_conditions})'
        
        aggregate_query = ""
        if aggregate_interval:
            aggregate_query = f'|> aggregateWindow(every: {aggregate_interval}, fn: mean, createEmpty: false)'
        
        query = f'''
        from(bucket: "{self.bucket}")
          |> range(start: -{hours}h)
          |> filter(fn: (r) => r["_measurement"] == "{self.measurement_raw}")
          |> filter(fn: (r) => {self._build_did_filter(filter_common)})
          {field_filter}
          {aggregate_query}
        '''
        
        try:
            return self._execute_query(query)
        except Exception as e:
            logger.error(f"Raw 데이터 조회 실패: {e}")
            return []
    
    async def get_raw_stats(
        self, 
        filter_common: FilterCommon, 
        hours: int = 24,
        field: str = "Load",
        measurement: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        특정 필드의 통계 조회 (평균, 최대, 최소)
        
        Args:
            filter_common: 공통 필터
            hours: 조회 시간 범위
            field: 통계 대상 필드 (기본: Load)
            measurement: 대상 measurement (기본: settings.INFLUXDB_MEASUREMENT_RAW)
        """
        if not self._initialized:
            await self.initialize()
        
        target_measurement = measurement or self.measurement_raw
        stats = {"filter": filter_common.to_dict(), "field": field, "hours": hours, "measurement": target_measurement}
        
        for stat_fn in ["mean", "max", "min"]:
            query = f'''
            from(bucket: "{self.bucket}")
              |> range(start: -{hours}h)
              |> filter(fn: (r) => r["_measurement"] == "{target_measurement}")
              |> filter(fn: (r) => {self._build_did_filter(filter_common)})
              |> filter(fn: (r) => r["_field"] == "{field}")
              |> {stat_fn}()
            '''
            
            try:
                results = self._execute_query(query)
                stats[stat_fn] = results[0]["value"] if results else None
            except Exception as e:
                logger.error(f"{field} {stat_fn} 조회 실패: {e}")
                stats[stat_fn] = None
        
        return stats
    
    async def get_running_stats(
        self, 
        filter_common: FilterCommon,
        hours: int = 24,
        field: str = "Load"
    ) -> Dict[str, Any]:
        """가동 중(Run > 0)인 구간의 필드 통계 조회 (Pivot 사용)"""
        if not self._initialized:
            await self.initialize()
        
        stats = {"filter": filter_common.to_dict(), "field": field, "hours": hours, "condition": "Run > 0"}
        
        # Flux Query: Load와 Run을 같이 가져와서 Pivot 후 Run > 0 필터링
        base_query = f'''
        from(bucket: "{self.bucket}")
          |> range(start: -{hours}h)
          |> filter(fn: (r) => r["_measurement"] == "{self.measurement_raw}")
          |> filter(fn: (r) => {self._build_did_filter(filter_common)})
          |> filter(fn: (r) => r["_field"] == "Run" or r["_field"] == "{field}")
          |> pivot(rowKey:["_time"], colKey:["_field"], valueColumn:"_value")
          |> filter(fn: (r) => r["Run"] > 0)
          |> drop(columns: ["_start", "_stop"])
        '''
        
        for stat_fn in ["mean", "max", "min"]:
            # pivot 결과 테이블에서 바로 집계 후 _value로 rename하여 Connector 호환
            if stat_fn == "mean":
                query = f'{base_query} |> mean(column: "{field}")'
            elif stat_fn == "max":
                query = f'{base_query} |> max(column: "{field}")'
            elif stat_fn == "min":
                query = f'{base_query} |> min(column: "{field}")'
            
            query += f' |> rename(columns: {{{"{field}": "_value"}}})'
                
            try:
                results = self._execute_query(query)
                stats[stat_fn] = results[0]["value"] if results else None
            except Exception as e:
                logger.error(f"가동 중 {field} {stat_fn} 조회 실패: {e}")
                stats[stat_fn] = None
        
        return stats
    
    async def get_raw_trend(
        self, 
        filter_common: FilterCommon,
        hours: int = 24,
        interval: str = "1h",
        field: str = "Load"
    ) -> List[Dict[str, Any]]:
        """
        센서 데이터 트렌드 조회 (시간별 집계)
        
        Args:
            filter_common: 공통 필터
            hours: 조회 시간 범위
            interval: 집계 간격 (예: "1h", "30m")
            field: 트렌드 대상 필드
            
        Returns:
            시간별 집계 데이터 리스트
        """
        if not self._initialized:
            await self.initialize()
        
        query = f'''
        from(bucket: "{self.bucket}")
          |> range(start: -{hours}h)
          |> filter(fn: (r) => r["_measurement"] == "{self.measurement_raw}")
          |> filter(fn: (r) => {self._build_did_filter(filter_common)})
          |> filter(fn: (r) => r["_field"] == "{field}")
          |> aggregateWindow(every: {interval}, fn: mean, createEmpty: false)
        '''
        
        try:
            return self._execute_query(query)
        except Exception as e:
            logger.error(f"{field} 트렌드 조회 실패: {e}")
            return []
    
    async def get_current_status(
        self, 
        filter_common: FilterCommon
    ) -> Dict[str, Any]:
        """
        현재 설비 상태 조회 (가동 여부, 현재 부하, 공구 등)
        
        Args:
            filter_common: 공통 필터
            
        Returns:
            현재 상태 정보
        """
        if not self._initialized:
            await self.initialize()
        
        # 주요 상태 필드만 조회
        status_fields = ["Run", "Load", "Feed", "Fov", "Sov", "TCount1", "TCount2", "TCount3", "TCount4"]
        raw_data = await self.get_latest_raw(filter_common, status_fields)
        
        if "error" in raw_data:
            return raw_data
        
        data = raw_data.get("data", {})
        
        # 상태 해석
        run_value = data.get("Run", {}).get("value", 0)
        is_running = run_value != 0 if run_value is not None else False
        
        return {
            "filter": filter_common.to_dict(),
            "is_running": is_running,
            "run_status": "가동중" if is_running else "정지",
            "current_load": data.get("Load", {}).get("value"),
            "current_feed": data.get("Feed", {}).get("value"),
            "fov": data.get("Fov", {}).get("value"),
            "sov": data.get("Sov", {}).get("value"),
            "tool_counts": {
                "T1": data.get("TCount1", {}).get("value"),
                "T2": data.get("TCount2", {}).get("value"),
                "T3": data.get("TCount3", {}).get("value"),
                "T4": data.get("TCount4", {}).get("value"),
            },
            "last_update": data.get("Run", {}).get("time"),
            "query_time": raw_data.get("query_time")
        }
    
    async def get_tool_stats(self, filter: FilterCommon, hours: int = 24) -> List[Dict[str, Any]]:
        """공구 사용 통계 조회 (tool_history measurement)"""
        # filter.did 체크 제거 (machine_id 없으면 전체/Regex 조회)
        
        # Count 필드 조회 (TCode별로 그룹화하여 합계 계산)
        # _build_did_filter 사용하여 유연한 필터링
        query = f'''
        from(bucket: "{self.bucket}")
          |> range(start: -{hours}h)
          |> filter(fn: (r) => r["_measurement"] == "{settings.INFLUXDB_MEASUREMENT_TOOL_HISTORY}")
          |> filter(fn: (r) => {self._build_did_filter(filter)})
          |> filter(fn: (r) => r["_field"] == "Count")
          |> group(columns: ["TCode"])
          |> sum()
        '''
        
        try:
            # 디버깅을 위한 쿼리 로그
            logger.info(f"Flux Query Execution (Tool Stats):\n{query.strip()}")
            loop = asyncio.get_running_loop()
            result = await loop.run_in_executor(None, lambda: self.query_api.query(query=query, org=self.org))
            
            stats = []
            for table in result:
                for record in table.records:
                    if record.get_value():
                        tool_code = record.values.get("TCode")
                        if tool_code:
                            # T 접두사 제거 (예: T505 -> 505)
                            tool_code = tool_code.replace("T", "") if tool_code.startswith("T") else tool_code
                            
                            stats.append({
                                "tool_code": tool_code,
                                "total_use_count": record.get_value(),
                                "period_hours": hours
                            })
            
            return stats
        except Exception as e:
            logger.error(f"공구 통계 조회 실패: {e}")
            return []

    async def get_machine_runtime(self, filter: FilterCommon, hours: int = 24) -> Dict[str, Any]:
        """설비 가동 시간 조회 (Run=3 상태 집계)"""
    async def get_machine_runtime(self, filter: FilterCommon, hours: int = 24) -> Dict[str, Any]:
        """설비 가동 시간 조회 (Run=3 상태 집계)"""
        # did 태그가 없어도 실행 (전체/Regex 조회)
        
        # Run 상태가 3(가동중)인 데이터 포인트 개수 * 수집주기(초) 
        # 정확한 계산을 위해 상태 지속 시간을 계산해야 하지만, 여기서는 근사치로 계산
        query = f'''
        from(bucket: "{self.bucket}")
          |> range(start: -{hours}h)
          |> filter(fn: (r) => r["_measurement"] == "{settings.INFLUXDB_MEASUREMENT_RAW}")
          |> filter(fn: (r) => {self._build_did_filter(filter)})
          |> filter(fn: (r) => r["_field"] == "Run")
          |> filter(fn: (r) => r["_value"] == 3)
          |> count() 
        '''
        
        try:
            # 디버깅을 위한 쿼리 로그
            logger.info(f"Flux Query Execution (Machine Runtime):\n{query.strip()}")
            loop = asyncio.get_running_loop()
            result = await loop.run_in_executor(None, lambda: self.query_api.query(query=query, org=self.org))
            
            runtime_seconds = 0
            for table in result:
                for record in table.records:
                    if record.get_value():
                        runtime_seconds = record.get_value()
            
            # 가동률 계산
            total_seconds = hours * 3600
            operating_rate = (runtime_seconds / total_seconds) * 100 if total_seconds > 0 else 0
            
            return {
                "runtime_seconds": runtime_seconds,
                "runtime_minutes": round(runtime_seconds / 60, 1),
                "runtime_hours": round(runtime_seconds / 3600, 2),
                "operating_rate": round(operating_rate, 2),
                "period_hours": hours
            }
        except Exception as e:
            logger.error(f"가동 시간 조회 실패: {e}")
            return {"runtime_seconds": 0, "operating_rate": 0.0}
    
    async def get_daily_runtime(self, filter: FilterCommon, days: int = 7) -> Dict[str, Any]:
        """일별 가동 시간/률 조회 (각 일별 + 총 가동률)"""
        logger.info(f"get_daily_runtime 호출: filter={filter.to_dict()}, days={days}")
        
        # 일별 Run=3 카운트 집계
        query = f'''
        from(bucket: "{self.bucket}")
          |> range(start: -{days}d)
          |> filter(fn: (r) => r["_measurement"] == "{settings.INFLUXDB_MEASUREMENT_RAW}")
          |> filter(fn: (r) => {self._build_did_filter(filter)})
          |> filter(fn: (r) => r["_field"] == "Run")
          |> filter(fn: (r) => r["_value"] == 3)
          |> aggregateWindow(every: 1d, fn: count, createEmpty: true, location: {{ zone: "Asia/Seoul", offset: 0s}})
        '''
        
        try:
            # 디버깅을 위한 쿼리 로그 (사용자 요청)
            logger.info(f"Flux Query Execution (Daily Runtime):\n{query.strip()}")
            loop = asyncio.get_running_loop()
            result = await loop.run_in_executor(None, lambda: self.query_api.query(query=query, org=self.org))
            
            daily_stats = []
            total_runtime_seconds = 0
            
            for table in result:
                for record in table.records:
                    date = record.get_time()
                    count = record.get_value() or 0
                    
                    # 1일 = 86400초
                    daily_seconds = count
                    daily_rate = (count / 86400) * 100 if count else 0
                    
                    daily_stats.append({
                        "date": date.strftime("%Y-%m-%d") if hasattr(date, 'strftime') else str(date),
                        "runtime_seconds": daily_seconds,
                        "runtime_hours": round(daily_seconds / 3600, 2),
                        "operating_rate": round(daily_rate, 2)
                    })
                    total_runtime_seconds += daily_seconds
            
            # 날짜순 정렬
            daily_stats.sort(key=lambda x: x["date"])
            
            # 총 가동률 계산
            total_seconds = days * 86400
            total_rate = (total_runtime_seconds / total_seconds) * 100 if total_seconds > 0 else 0
            
            return {
                "daily": daily_stats,
                "total": {
                    "period_days": days,
                    "runtime_seconds": total_runtime_seconds,
                    "runtime_hours": round(total_runtime_seconds / 3600, 2),
                    "operating_rate": round(total_rate, 2)
                }
            }
        except Exception as e:
            logger.error(f"일별 가동 시간 조회 실패: {e}")
            return {"daily": [], "total": {"runtime_seconds": 0, "operating_rate": 0.0}}
    
    async def get_weekly_runtime(self, filter: FilterCommon, weeks: int = 4) -> Dict[str, Any]:
        """주별 가동 시간/률 조회 (30일 초과 기간용)"""
        logger.info(f"get_weekly_runtime 호출: filter={filter.to_dict()}, weeks={weeks}")
        
        # 주별 Run=3 카운트 집계
        query = f'''
        from(bucket: "{self.bucket}")
          |> range(start: -{weeks}w)
          |> filter(fn: (r) => r["_measurement"] == "{settings.INFLUXDB_MEASUREMENT_RAW}")
          |> filter(fn: (r) => {self._build_did_filter(filter)})
          |> filter(fn: (r) => r["_field"] == "Run")
          |> filter(fn: (r) => r["_value"] == 3)
          |> aggregateWindow(every: 1w, fn: count, createEmpty: true, offset: -3d, location: {{ zone: "Asia/Seoul", offset: 0s}})
        '''
        
        try:
            # 디버깅을 위한 쿼리 로그 (사용자 요청)
            logger.info(f"Flux Query Execution (Weekly Runtime):\n{query.strip()}")
            loop = asyncio.get_running_loop()
            result = await loop.run_in_executor(None, lambda: self.query_api.query(query=query, org=self.org))
            
            weekly_stats = []
            total_runtime_seconds = 0
            
            for table in result:
                for record in table.records:
                    date = record.get_time()
                    count = record.get_value() or 0
                    
                    # 1주 = 604800초 (7 * 86400)
                    weekly_seconds = count
                    weekly_rate = (count / 604800) * 100 if count else 0
                    
                    weekly_stats.append({
                        "week_start": date.strftime("%Y-%m-%d") if hasattr(date, 'strftime') else str(date),
                        "runtime_seconds": weekly_seconds,
                        "runtime_hours": round(weekly_seconds / 3600, 2),
                        "operating_rate": round(weekly_rate, 2)
                    })
                    total_runtime_seconds += weekly_seconds
            
            # 날짜순 정렬
            weekly_stats.sort(key=lambda x: x["week_start"])
            
            # 총 가동률 계산
            total_seconds = weeks * 7 * 86400
            total_rate = (total_runtime_seconds / total_seconds) * 100 if total_seconds > 0 else 0
            
            return {
                "weekly": weekly_stats,
                "total": {
                    "period_weeks": weeks,
                    "runtime_seconds": total_runtime_seconds,
                    "runtime_hours": round(total_runtime_seconds / 3600, 2),
                    "operating_rate": round(total_rate, 2)
                }
            }
        except Exception as e:
            logger.error(f"주별 가동 시간 조회 실패: {e}")
            return {"weekly": [], "total": {"runtime_seconds": 0, "operating_rate": 0.0}}
    
    async def get_monthly_runtime(self, filter: FilterCommon, months: int = 12) -> Dict[str, Any]:
        """월별 가동 시간/률 조회 (12주 초과 기간용)"""
        logger.info(f"get_monthly_runtime 호출: filter={filter.to_dict()}, months={months}")
        
        # 월별 Run=3 카운트 집계
        query = f'''
        from(bucket: "{self.bucket}")
          |> range(start: -{months}mo)
          |> filter(fn: (r) => r["_measurement"] == "{settings.INFLUXDB_MEASUREMENT_RAW}")
          |> filter(fn: (r) => {self._build_did_filter(filter)})
          |> filter(fn: (r) => r["_field"] == "Run")
          |> filter(fn: (r) => r["_value"] == 3)
          |> aggregateWindow(every: 1mo, fn: count, createEmpty: true, location: {{ zone: "Asia/Seoul", offset: 0s}})
        '''
        
        try:
            # 디버깅을 위한 쿼리 로그 (사용자 요청)
            logger.info(f"Flux Query Execution (Monthly Runtime):\n{query.strip()}")
            loop = asyncio.get_running_loop()
            result = await loop.run_in_executor(None, lambda: self.query_api.query(query=query, org=self.org))
            
            monthly_stats = []
            total_runtime_seconds = 0
            
            for table in result:
                for record in table.records:
                    date = record.get_time()
                    count = record.get_value() or 0
                    
                    # 월 평균 = 30일 = 2592000초
                    monthly_seconds = count
                    monthly_rate = (count / 2592000) * 100 if count else 0
                    
                    monthly_stats.append({
                        "month": date.strftime("%Y-%m") if hasattr(date, 'strftime') else str(date)[:7],
                        "runtime_seconds": monthly_seconds,
                        "runtime_hours": round(monthly_seconds / 3600, 2),
                        "operating_rate": round(monthly_rate, 2)
                    })
                    total_runtime_seconds += monthly_seconds
            
            # 날짜순 정렬
            monthly_stats.sort(key=lambda x: x["month"])
            
            # 총 가동률 계산
            total_seconds = months * 30 * 86400
            total_rate = (total_runtime_seconds / total_seconds) * 100 if total_seconds > 0 else 0
            
            return {
                "monthly": monthly_stats,
                "total": {
                    "period_months": months,
                    "runtime_seconds": total_runtime_seconds,
                    "runtime_hours": round(total_runtime_seconds / 3600, 2),
                    "operating_rate": round(total_rate, 2)
                }
            }
        except Exception as e:
            logger.error(f"월별 가동 시간 조회 실패: {e}")
            return {"monthly": [], "total": {"runtime_seconds": 0, "operating_rate": 0.0}}
    
    # ============ 기간별 통계 (Stats) ============
    
    async def get_daily_stats(
        self, 
        filter_common: FilterCommon,
        days: int = 7,
        field: str = "Load",
        measurement: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        일별 필드 통계 조회 (각 일별 mean/max/min + 총 통계)
        
        Args:
            filter_common: 공통 필터
            days: 조회 일수 (기본: 7일)
            field: 통계 대상 필드 (기본: Load)
            
        Returns:
            {"daily": [...], "total": {...}}
        """
        logger.info(f"get_daily_stats 호출: filter={filter_common.to_dict()}, days={days}, field={field}")
        
        if not self._initialized:
            await self.initialize()
            
        target_measurement = measurement or self.measurement_raw
        
        daily_map = {}
        all_values = []
        
        # 날짜별 초기화 (빈 데이터 채우기 위함)
        from datetime import datetime, timedelta
        now = datetime.now()
        for i in range(days):
            d = now - timedelta(days=days - i - 1)
            d_str = d.strftime("%Y-%m-%d")
            daily_map[d_str] = {"day": d_str, "mean": 0, "max": 0, "min": 0, "count": 0}

        for stat_fn in ["mean", "max", "min"]:
            query = f'''
            import "timezone"
            option location = timezone.location(name: "Asia/Seoul")
            
            from(bucket: "{self.bucket}")
              |> range(start: -{days}d)
              |> filter(fn: (r) => r["_measurement"] == "{target_measurement}")
              |> filter(fn: (r) => {self._build_did_filter(filter_common)})
              |> filter(fn: (r) => r["_field"] == "{field}")
              |> aggregateWindow(every: 1d, fn: {stat_fn}, createEmpty: false, timeSrc: "_start")
            '''
            
            try:
                results = self._execute_query(query)
                for res in results:
                    dt = res.get("time")
                    val = res.get("value")
                    if dt and val is not None:
                        dt = dt.astimezone()
                        # 날짜 문자열 키 생성
                        d_key = dt.strftime("%Y-%m-%d")
                        
                        if d_key in daily_map:
                            daily_map[d_key][stat_fn] = round(val, 2)
                            if stat_fn == "mean":
                                all_values.append(val)
            except Exception as e:
                logger.error(f"일별 {field} {stat_fn} 전체 조회 실패: {e}")
                
        daily_list = sorted(daily_map.values(), key=lambda x: x["day"])
        
        total = {
            "period_days": days,
            "field": field,
            "mean": round(sum(all_values) / len(all_values), 2) if all_values else 0,
            "max": max([d["max"] for d in daily_list if d["max"] is not None], default=0),
            "min": min([d["min"] for d in daily_list if d["min"] is not None], default=0)
        }
        
        return {"daily": daily_list, "total": total}
    
    async def get_weekly_stats(
        self, 
        filter_common: FilterCommon,
        weeks: int = 4,
        field: str = "Load",
        measurement: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        주별 필드 통계 조회 (각 주별 mean/max/min + 총 통계)
        
        Args:
            filter_common: 공통 필터
            weeks: 조회 주수 (기본: 4주)
            field: 통계 대상 필드 (기본: Load)
            
        Returns:
            {"weekly": [...], "total": {...}}
        """
        logger.info(f"get_weekly_stats 호출: filter={filter_common.to_dict()}, weeks={weeks}, field={field}")
        
        if not self._initialized:
            await self.initialize()
        
        target_measurement = measurement or self.measurement_raw
        
        weekly_map = {}
        all_values = []
        
        for stat_fn in ["mean", "max", "min"]:
            query = f'''
            import "timezone"
            option location = timezone.location(name: "Asia/Seoul")
            
            from(bucket: "{self.bucket}")
              |> range(start: -{weeks}w)
              |> filter(fn: (r) => r["_measurement"] == "{target_measurement}")
              |> filter(fn: (r) => {self._build_did_filter(filter_common)})
              |> filter(fn: (r) => r["_field"] == "{field}")
              |> aggregateWindow(every: 1w, fn: {stat_fn}, createEmpty: false, timeSrc: "_start")
            '''
            
            try:
                results = self._execute_query(query)
                for res in results:
                    dt = res.get("time")
                    val = res.get("value")
                    
                    if dt and val is not None:
                        dt = dt.astimezone()
                        d_str = dt.strftime("%Y-%m-%d")
                        w_key = f"{d_str} 주"
                        
                        if w_key not in weekly_map:
                            weekly_map[w_key] = {"week": w_key, "mean": 0, "max": 0, "min": 0}
                        
                        weekly_map[w_key][stat_fn] = round(val, 2)
                        
                        if stat_fn == "mean":
                            all_values.append(val)
            except Exception as e:
                logger.error(f"주별 {field} {stat_fn} 전체 조회 실패: {e}")

        weekly_list = sorted(weekly_map.values(), key=lambda x: x["week"])
        
        total = {
            "period_weeks": weeks,
            "field": field,
            "mean": round(sum(all_values) / len(all_values), 2) if all_values else 0,
            "max": max([w["max"] for w in weekly_list if w["max"] is not None], default=0),
            "min": min([w["min"] for w in weekly_list if w["min"] is not None], default=0)
        }
        
        return {"weekly": weekly_list, "total": total}
    
    async def get_monthly_stats(
        self, 
        filter_common: FilterCommon,
        months: int = 12,
        field: str = "Load",
        measurement: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        월별 필드 통계 조회 (각 월별 mean/max/min + 총 통계)
        
        Args:
            filter_common: 공통 필터
            months: 조회 개월수 (기본: 12개월)
            field: 통계 대상 필드 (기본: Load)
            
        Returns:
            {"monthly": [...], "total": {...}}
        """
        logger.info(f"get_monthly_stats 호출: filter={filter_common.to_dict()}, months={months}, field={field}")
        
        if not self._initialized:
            await self.initialize()
        
        target_measurement = measurement or self.measurement_raw
        
        monthly_map = {}
        all_values = []
        
        for stat_fn in ["mean", "max", "min"]:
            query = f'''
            import "timezone"
            option location = timezone.location(name: "Asia/Seoul")
            
            from(bucket: "{self.bucket}")
              |> range(start: -{months}mo)
              |> filter(fn: (r) => r["_measurement"] == "{target_measurement}")
              |> filter(fn: (r) => {self._build_did_filter(filter_common)})
              |> filter(fn: (r) => r["_field"] == "{field}")
              |> aggregateWindow(every: 1mo, fn: {stat_fn}, createEmpty: false, timeSrc: "_start")
            '''
            
            try:
                results = self._execute_query(query)
                for res in results:
                    dt = res.get("time")
                    val = res.get("value")
                    if dt and val is not None:
                        dt = dt.astimezone()
                        m_key = dt.strftime("%Y-%m")
                        
                        if m_key not in monthly_map:
                            monthly_map[m_key] = {"month": m_key, "mean": 0, "max": 0, "min": 0}
                        
                        monthly_map[m_key][stat_fn] = round(val, 2)
                        if stat_fn == "mean":
                            all_values.append(val)
            except Exception as e:
                logger.error(f"월별 {field} {stat_fn} 전체 조회 실패: {e}")
        
        monthly_list = sorted(monthly_map.values(), key=lambda x: x["month"])
        
        total = {
            "period_months": months,
            "field": field,
            "mean": round(sum(all_values) / len(all_values), 2) if all_values else 0,
            "max": max([m["max"] for m in monthly_list if m["max"] is not None], default=0),
            "min": min([m["min"] for m in monthly_list if m["min"] is not None], default=0)
        }
        
        return {"monthly": monthly_list, "total": total}
    
    # ============ 헬스체크 ============
    
    async def health_check(self) -> Dict[str, Any]:
        """InfluxDB 연결 상태 확인"""
        try:
            if not self._initialized:
                return {"status": "disconnected", "error": "Not initialized"}
            
            health = self.client.health()
            return {
                "status": health.status,
                "version": health.version,
                "url": self.url
            }
        except Exception as e:
            return {"status": "error", "error": str(e)}


# 싱글톤 인스턴스
_influxdb_connector: Optional[InfluxDBConnector] = None


def get_influxdb_connector() -> InfluxDBConnector:
    """InfluxDB 커넥터 싱글톤 인스턴스 반환"""
    global _influxdb_connector
    if _influxdb_connector is None:
        _influxdb_connector = InfluxDBConnector()
    return _influxdb_connector
