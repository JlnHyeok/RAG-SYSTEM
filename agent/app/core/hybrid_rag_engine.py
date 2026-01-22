"""
하이브리드 RAG 엔진 (통합 오케스트레이터)
문서(Qdrant), 구조화 데이터(MongoDB), 시계열 데이터(InfluxDB)를 
통합하여 질문에 답변하는 시스템의 핵심 엔진입니다.

제조 도메인 데이터 소스:
- DOCUMENT: Qdrant 문서 검색 (매뉴얼, 가이드)
- PRODUCTION: MongoDB 생산 이력
- ABNORMAL: MongoDB 이상감지 이력
- MACHINE: MongoDB 설비 정보
- TOOL: MongoDB 공구 정보
- RAW_SENSOR: InfluxDB 실시간 센서 데이터
"""

import logging
import asyncio
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
import time
import re

# DB 및 검색 모듈
from app.core.db.mongodb_connector import get_mongodb_connector, MongoDBConnector, FilterCommon
from app.core.db.influxdb_connector import get_influxdb_connector, InfluxDBConnector
from app.core.retrieval.document_retriever import document_retriever, DocumentRetriever

# LLM 및 처리 모듈
from app.core.llm.gemini_service import gemini_service
from app.core.llm.question_classifier import QuestionClassifier, QuestionType, SensorQueryType
from app.core.llm.answer_generator import AnswerGenerator
from app.core.processing.text_processor import text_processor
from app.core.session.conversation_manager import conversation_manager

# 모델 및 설정
from app.models.schemas import QueryRequest, QueryResponse, SearchResult
from app.core.config import settings

logger = logging.getLogger(__name__)


class DataSourceType(str, Enum):
    """데이터 소스 유형"""
    DOCUMENT = "document"           # Qdrant 문서 검색
    PRODUCTION = "production"       # MongoDB 생산 이력
    ABNORMAL = "abnormal"           # MongoDB 이상감지
    MACHINE = "machine"             # MongoDB 설비 정보
    TOOL = "tool"                   # MongoDB 공구 정보
    RAW_SENSOR = "raw_sensor"       # InfluxDB 실시간 센서
    HYBRID = "hybrid"               # 다중 소스 통합


# QuestionType -> DataSourceType 매핑
QUESTION_TO_SOURCE_MAP = {
    QuestionType.DOCUMENT_QUERY: DataSourceType.DOCUMENT,
    QuestionType.PRODUCTION_QUERY: DataSourceType.PRODUCTION,
    QuestionType.ABNORMAL_QUERY: DataSourceType.ABNORMAL,
    QuestionType.MACHINE_QUERY: DataSourceType.MACHINE,
    QuestionType.TOOL_QUERY: DataSourceType.TOOL,
    QuestionType.RAW_SENSOR_QUERY: DataSourceType.RAW_SENSOR,
    QuestionType.HYBRID_QUERY: DataSourceType.HYBRID,
}


@dataclass
class QueryIntent:
    """질문 의도 분석 결과"""
    primary_source: DataSourceType
    secondary_sources: List[DataSourceType] = field(default_factory=list)
    entities: Dict[str, Any] = field(default_factory=dict)
    time_range: Optional[str] = None  # "1h", "24h", "7d" 등
    confidence: float = 0.0
    sensor_query_type: Optional[SensorQueryType] = None  # 센서 쿼리 세부 유형
    target_field: Optional[str] = None  # 조회 대상 필드 (CT, Load 등)


@dataclass
class HybridContext:
    """하이브리드 컨텍스트 데이터"""
    document_results: List[SearchResult] = field(default_factory=list)
    document_context: Optional[str] = None
    production_data: Optional[List[Dict]] = None
    abnormal_data: Optional[List[Dict]] = None
    machine_data: Optional[Dict] = None
    tool_data: Optional[List[Dict]] = None
    sensor_data: Optional[Dict] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class HybridRAGEngine:
    """
    하이브리드 RAG 엔진
    
    사용자 질문의 의도를 분석하고, 다양한 데이터 소스(문서, DB, 센서)에서 
    최적의 정보를 수집하여 지능적인 답변을 생성합니다.
    """
    
    def __init__(self):
        # 데이터 소스
        self.retriever: DocumentRetriever = document_retriever
        self.mongodb: MongoDBConnector = get_mongodb_connector()
        self.influxdb: InfluxDBConnector = get_influxdb_connector()
        
        # 처리 엔진
        self.gemini = gemini_service
        self.text_processor = text_processor
        self.conversation_manager = conversation_manager
        
        # 초기화 상태 및 의존성 주입 객체
        self._initialized = False
        self.question_classifier: Optional[QuestionClassifier] = None
        self.answer_generator: Optional[AnswerGenerator] = None
    
    async def initialize(self):
        """하이브리드 RAG 엔진 및 관련 모듈 초기화"""
        if self._initialized:
            return
        
        print("\n" + "="*50)
        print(f"🚀 {settings.APP_NAME} 초기화 시작")
        print("="*50)
        
        try:
            # 모든 하위 서비스 병렬 초기화
            results = await asyncio.gather(
                self.retriever.initialize(),
                self.gemini.initialize(test_connection=False),
                self.mongodb.initialize(),
                self.influxdb.initialize(),
                return_exceptions=True
            )
            
            # 각 서비스별 상태 확인 및 리포트
            component_names = ["Qdrant", "Gemini", "MongoDB", "InfluxDB"]
            all_success = True
            
            print("\n📋 컴포넌트 초기화:")
            for name, result in zip(component_names, results):
                if isinstance(result, Exception):
                    print(f"  ❌ {name.ljust(20)}: 실패 ({str(result)})")
                    all_success = False
                else:
                    print(f"  ✅ {name.ljust(20)}: 성공")
            
            if not all_success:
                logger.warning("일부 컴포넌트가 정상적으로 초기화되지 않았습니다.")

            # 의존성 주입 및 객체 생성
            self.question_classifier = QuestionClassifier(self.gemini, self.retriever.vector_store)
            self.answer_generator = AnswerGenerator(self.gemini)
            self.conversation_manager.set_gemini_service(self.gemini)
            
            self._initialized = True
            print("\n" + "="*50)
            print(f"✨ {settings.APP_NAME} 초기화 완료!")
            print("="*50 + "\n")
            
        except Exception as e:
            logger.error(f"하이브리드 RAG 엔진 초기화 치명적 실패: {e}")
            print(f"\n❌ 초기화 중 치명적 오류 발생: {e}")
            raise
    
    async def query(self, request: QueryRequest, on_status: Optional[callable] = None) -> QueryResponse:
        """사용자 질문에 대한 통합 하이브리드 RAG 파이프라인 실행"""
        start_time = time.time()
        
        if not self._initialized:
            await self.initialize()
            
        # 1. 대화 세션 관리
        conversation_key = self.conversation_manager.get_conversation_key(
            request.user_id, 
            request.conversation_id
        )
        self.conversation_manager.ensure_history_exists(conversation_key)
        
        try:
            # 2. 대화 맥락 분석 및 질문 보정
            if on_status: await on_status("대화 맥락 파악 중...")
            context_aware_question, _ = await self.conversation_manager.analyze_question_context(
                request.question, conversation_key
            )
            
            # 3. 메타 질문 및 의도 분석
            if on_status: await on_status("질문 의도 분석 중...")
            
            # 3.1 일반 메타 질문 (인사, 문서목록 등) 처리
            meta_response = await self.question_classifier.handle_meta_questions(
                context_aware_question, request.user_id
            )
            if meta_response:
                return self._create_simple_response(meta_response, request, conversation_key, start_time)
            
            # 3.2 데이터 소스 의도 분석
            intent = await self._analyze_intent(context_aware_question)
            logger.info(f"데이터 소스 의도: {intent.primary_source.value}, 엔티티: {intent.entities}")
            
            # 4. 데이터 수집 (병렬 처리)
            context = await self._gather_context(intent, request, context_aware_question, on_status)
            
            # 5. 검색 결과 및 데이터 유무에 따른 답변 생성 전략
            if on_status: await on_status("답변 생성 중...")
            
            # 데이터가 아무것도 없는 경우 일반 대화 시도
            has_data = {
                "document": bool(context.document_results),
                "production": bool(context.production_data),
                "abnormal": bool(context.abnormal_data),
                "machine": bool(context.machine_data),
                "tool": bool(context.tool_data),
                "sensor": bool(context.sensor_data)
            }
            logger.info(f"수집된 컨텍스트: {has_data}")
            
            if not any(has_data.values()):
                logger.info("컨텍스트 데이터 없음 - 일반 대화로 전환")
                return await self._handle_general_conversation(request.question, context_aware_question, conversation_key, start_time)

            # 답변 생성 (LLM)
            response = await self._generate_answer(request, context, actual_question=context_aware_question, history_key=conversation_key, start_time=start_time, intent=intent)
            
            proc_time = response.processing_time if response.processing_time is not None else 0.0
            logger.info(f"하이브리드 쿼리 완료: {proc_time:.2f}초")
            return response
                
        except Exception as e:
            logger.error(f"하이브리드 쿼리 처리 실패: {e}")
            return QueryResponse(
                answer="죄송합니다. 요청을 처리하는 중에 오류가 발생했습니다.",
                sources=[],
                confidence=0.0,
                processing_time=time.time() - start_time
            )

    async def _analyze_intent(self, question: str) -> QueryIntent:
        """질문 분석 및 의도 파악 (통합 분석 결과 사용)"""
        logger.info(f"🔍 의도 분석 시작 - 입력 질문: '{question}'")
        
        # 1. 통합 질문 분석 (의도, 엔티티, 시간 범위)
        # 이제 classify_question이 QuestionAnalysisResult를 반환합니다.
        analysis_result = await self.question_classifier.classify_question(question)
        
        question_type = analysis_result.primary_type
        primary_source = QUESTION_TO_SOURCE_MAP.get(question_type, DataSourceType.DOCUMENT)
        
        # 2. 보조 소스 결정
        secondary_sources = []
        for sec_type in analysis_result.secondary_types:
            source = QUESTION_TO_SOURCE_MAP.get(sec_type)
            if source and source not in secondary_sources and source != primary_source:
                secondary_sources.append(source)
                
        # 3. 엔티티 및 시간 범위 사용
        entities = analysis_result.entities
        time_range = analysis_result.time_range
        
        # [안전장치] machine_id가 추출되었다면 MACHINE 소스 자동 추가
        if entities.get("machine_id") and primary_source != DataSourceType.MACHINE:
             if DataSourceType.MACHINE not in secondary_sources:
                secondary_sources.append(DataSourceType.MACHINE)
                logger.info("엔티티 기반 MACHINE 소스 자동 추가")

        # 4. 복합 질문(HYBRID)인 경우 문서는 기본 포함
        if primary_source == DataSourceType.HYBRID:
            if DataSourceType.DOCUMENT not in secondary_sources:
                secondary_sources.append(DataSourceType.DOCUMENT)

        logger.info(f"🔍 분석 결과 - Primary: {primary_source}, Secondary: {secondary_sources}, Time: {time_range}, Entities: {entities}")
        
        return QueryIntent(
            primary_source=primary_source,
            secondary_sources=secondary_sources,
            entities=entities,
            time_range=time_range,
            confidence=0.9,
            sensor_query_type=analysis_result.sensor_query_type,
            target_field=analysis_result.target_field
        )
    


    async def _gather_context(
        self, 
        intent: QueryIntent, 
        request: QueryRequest, 
        question: str,
        on_status: Optional[callable]
    ) -> HybridContext:
        """다양한 소스에서 컨텍스트 병렬 수집"""
        context = HybridContext()
        tasks = []
        source_keys = []
        
        # [NEW] machine_id가 있다면 설비 정보를 먼저 조회하여 정확한 필터 정보 구성
        # (기본값 F01 대신 실제 workshopCode 등을 사용하기 위함 - 백엔드 로직 일치화)
        if intent.entities.get("machine_id"):
            try:
                mid = intent.entities["machine_id"]
                machine_info = await self.mongodb.get_machine_by_code(mid)
                if machine_info:
                    intent.entities["workshop_id"] = machine_info.get("workshopCode")
                    intent.entities["line_id"] = machine_info.get("lineCode")
                    intent.entities["op_code"] = machine_info.get("opCode")
                    logger.info(f"설비 정보 기반 필터 업데이트: {intent.entities}")
            except Exception as e:
                logger.warning(f"설비 정보 선행 조회 실패: {e}")

        # 기본 필터 생성
        filter_common = self._create_filter_common(intent.entities)
        
        sources = list(set([intent.primary_source] + intent.secondary_sources))
        
        for source in sources:
            if source == DataSourceType.DOCUMENT:
                if on_status: await on_status("문서 지식 검색 중...")
                tasks.append(self.retriever.search(question, request.user_id, request.max_results, request.score_threshold))
                source_keys.append("document")
                
            elif source == DataSourceType.PRODUCTION:
                if on_status: await on_status("생산 이력 조회 중...")
                hours = self._time_range_to_hours(intent.time_range)
                tasks.append(self._get_production_data(filter_common, hours))
                source_keys.append("production")
                
            elif source == DataSourceType.ABNORMAL:
                if on_status: await on_status("이상감지 이력 조회 중...")
                hours = self._time_range_to_hours(intent.time_range)
                tasks.append(self._get_abnormal_data(filter_common, hours, intent.entities.get("abnormal_code")))
                source_keys.append("abnormal")
                
            elif source == DataSourceType.MACHINE:
                if on_status: await on_status("설비 정보 조회 중...")
                tasks.append(self._get_machine_data(intent.entities))
                source_keys.append("machine")
                
            elif source == DataSourceType.TOOL:
                if on_status: await on_status("공구 정보 조회 중...")
                tasks.append(self._get_tool_data(intent.entities))
                source_keys.append("tool")
                
            elif source == DataSourceType.RAW_SENSOR:
                if on_status: await on_status("센서 데이터 조회 중...")
                hours = self._time_range_to_hours(intent.time_range)
                tasks.append(self._get_sensor_data(filter_common, hours, intent.sensor_query_type, intent.target_field))
                source_keys.append("sensor")

        if not tasks: 
            return context

        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        for key, res in zip(source_keys, results):
            if isinstance(res, Exception): 
                logger.warning(f"{key} 데이터 조회 실패: {res}")
                continue
            if key == "document": 
                context.document_results = res
                context.document_context = self.text_processor.build_context(res)
            elif key == "production": 
                context.production_data = res
            elif key == "abnormal": 
                context.abnormal_data = res
            elif key == "machine": 
                context.machine_data = res
            elif key == "tool": 
                context.tool_data = res
            elif key == "sensor": 
                context.sensor_data = res
            
        return context
    
    def _create_filter_common(self, entities: Dict) -> Optional[FilterCommon]:
        """엔티티에서 FilterCommon 생성"""
        workshop_id = entities.get("workshop_id") or settings.DEFAULT_WORKSHOP_ID
        line_id = entities.get("line_id") or settings.DEFAULT_LINE_ID
        op_code = entities.get("op_code") or settings.DEFAULT_OP_CODE
        machine_id = entities.get("machine_id")
        
        # 필수 필드가 없으면 None 반환
        if not workshop_id or not line_id or not op_code:
            return None
        
        return FilterCommon(
            workshop_id=workshop_id,
            line_id=line_id,
            op_code=op_code,
            machine_id=machine_id
        )
    
    def _time_range_to_hours(self, time_range: Optional[str]) -> int:
        """시간 범위 문자열을 시간 단위로 변환 (예: 1년, 3개월, 2주, 1d, 24h)"""
        if not time_range:
            return 24
            
        try:
            # 정규식으로 숫자와 단위 추출
            # 예: "1년", "1y", "3개월", "30d"
            match = re.search(r'(\d+)\s*(년|y|개월|m|달|주|w|일|d|시간|h|분|min)', time_range.lower())
            
            if match:
                val = int(match.group(1))
                unit = match.group(2)
                
                if unit in ['년', 'y', 'year', 'years']:
                    return val * 365 * 24
                if unit in ['개월', 'm', 'mon', 'month', 'months', '달']:
                    return val * 30 * 24
                if unit in ['주', 'w', 'week', 'weeks']:
                    return val * 7 * 24
                if unit in ['일', 'd', 'day', 'days']:
                    return val * 24
                if unit in ['시간', 'h', 'hour', 'hours']:
                    return val
                if unit in ['분', 'min', 'minute', 'minutes']:
                    return max(1, int(val / 60))
            
            # 기존 로직 (Fallback)
            if time_range.endswith("h"):
                return int(time_range[:-1])
            elif time_range.endswith("d"):
                return int(time_range[:-1]) * 24
                
        except Exception as e:
            logger.warning(f"시간 파싱 실패 ({time_range}): {e}")
            
        return 24
    
    # ============ 데이터 조회 메서드 ============
    
    async def _get_production_data(self, filter_common: Optional[FilterCommon], hours: int) -> List[Dict]:
        """생산 이력 조회"""
        if not filter_common:
            return []
        
        try:
            products = await self.mongodb.get_recent_products(filter_common, hours, limit=50)
            stats = await self.mongodb.get_product_stats(filter_common, hours)
            
            return {
                "recent_products": products[:10],  # 최근 10건
                "stats": stats,
                "total_count": len(products)
            }
        except Exception as e:
            logger.error(f"생산 데이터 조회 실패: {e}")
            return []
    
    async def _get_abnormal_data(
        self, 
        filter_common: Optional[FilterCommon], 
        hours: int,
        abnormal_code: Optional[str] = None
    ) -> List[Dict]:
        """이상감지 이력 조회"""
        if not filter_common:
            return []
        
        try:
            if abnormal_code:
                abnormals = await self.mongodb.get_abnormals_by_code(filter_common, abnormal_code, hours)
            else:
                abnormals = await self.mongodb.get_recent_abnormals(filter_common, hours)
            
            summary = await self.mongodb.get_abnormal_summary(filter_common, hours)
            
            return {
                "recent_abnormals": abnormals[:10],
                "summary": summary,
                "total_count": len(abnormals)
            }
        except Exception as e:
            logger.error(f"이상감지 데이터 조회 실패: {e}")
            return []
    
    async def _get_machine_data(self, entities: Dict) -> Dict:
        """설비 정보 조회"""
        machine_id = entities.get("machine_id")
        
        try:
            if machine_id:
                logger.info(f"설비 단건 조회: {machine_id}")
                machine = await self.mongodb.get_machine_by_code(machine_id)
                if machine:
                    threshold = await self.mongodb.get_threshold_by_machine(machine_id)
                    tools = await self.mongodb.get_tools_by_machine(machine_id)
                    
                    # InfluxDB 가동 시간 조회 및 공구 사용량 조회
                    runtime = {}
                    tool_counts = []
                    
                    # machine 정보에서 필터 값 직접 추출 (machineMaster 필드 사용) 또는 기본값
                    workshop_code = machine.get("workshopCode") or settings.DEFAULT_WORKSHOP_ID
                    line_code = machine.get("lineCode") or settings.DEFAULT_LINE_ID
                    op_code = machine.get("opCode") or settings.DEFAULT_OP_CODE
                    
                    logger.info(f"Machine 필터 정보: workshop={workshop_code}, line={line_code}, op={op_code}")
                    
                    # FilterCommon 직접 생성 (machine 정보 기반)
                    if workshop_code and line_code and op_code:
                        filter_common = FilterCommon(
                            workshop_id=workshop_code,
                            line_id=line_code,
                            op_code=op_code,
                            machine_id=machine_id
                        )
                        logger.info(f"FilterCommon 생성: {filter_common}")
                        
                        # 가동 시간 조회 (InfluxDB - 실패해도 진행)
                        try:
                            if filter_common.did:
                                hour_range = self._time_range_to_hours(entities.get("time_range"))
                                runtime = await self.influxdb.get_machine_runtime(filter_common, hours=hour_range)
                        except Exception as e:
                            logger.error(f"가동 시간 조회 실패 (InfluxDB 무시): {e}")

                        # 공구 사용량 조회 (MongoDB - 필수)
                        try:
                            tool_counts = await self.mongodb.get_current_tool_counts(filter_common)
                            logger.info(f"공구 사용량 조회 결과: {len(tool_counts)}개")
                        except Exception as e:
                            logger.error(f"공구 사용량 조회 실패: {e}")
                    else:
                        logger.warning(f"FilterCommon 생성 실패: 필수 필드 누락")
                    
                    return {"machine": machine, "threshold": threshold, "tools": tools, "runtime": runtime, "tool_counts": tool_counts}
            else:
                logger.info("설비 전체 목록 조회 시작")
                machines = await self.mongodb.get_all_machines()
                logger.info(f"조회된 설비 수: {len(machines)}")
                if machines:
                    logger.info(f"설비 코드 목록: {[m.get('machineCode') for m in machines]}")
                
                # 각 설비별 공구 정보 및 사용량 조회
                machines_with_tools = []
                for machine in machines[:20]:  # 최대 20개 설비
                    machine_code = machine.get("machineCode")
                    if machine_code:
                        # 1. 공구 마스터 조회
                        tools = await self.mongodb.get_tools_by_machine(machine_code)
                        machine["tools"] = tools
                        
                        # 2. 공구 사용량 조회 (FilterCommon 생성 필요)
                        try:
                            workshop_code = machine.get("workshopCode") or settings.DEFAULT_WORKSHOP_ID
                            line_code = machine.get("lineCode") or settings.DEFAULT_LINE_ID
                            op_code = machine.get("opCode") or settings.DEFAULT_OP_CODE
                            
                            if workshop_code and line_code and op_code:
                                filter_common = FilterCommon(
                                    workshop_id=workshop_code,
                                    line_id=line_code,
                                    op_code=op_code,
                                    machine_id=machine_code
                                )
                                tool_counts = await self.mongodb.get_current_tool_counts(filter_common)
                                machine["tool_counts"] = tool_counts
                        except Exception as e:
                            logger.error(f"설비 {machine_code} 공구 사용량 조회 실패: {e}")
                            
                    machines_with_tools.append(machine)
                
                return {"machines": machines_with_tools, "total_count": len(machines)}
        except Exception as e:
            logger.error(f"설비 데이터 조회 실패: {e}")
            return {}
    
    async def _get_tool_data(self, entities: Dict) -> Dict:
        """공구 정보 조회 (InfluxDB 사용 통계 포함)"""
        machine_id = entities.get("machine_id")
        tool_code = entities.get("tool_code")
        
        try:
            # InfluxDB 사용 통계 조회
            usage_stats = []
            filter_common = self._create_filter_common(entities)
            
            tool_counts = []
            
            if filter_common:
                logger.info(f" [ToolDebug] 공구 데이터 조회 시작 - Filter: {filter_common.to_mongo_filter()}")
                
                # InfluxDB 조회
                try:
                    hour_range = self._time_range_to_hours(entities.get("time_range"))
                    usage_stats = await self.influxdb.get_tool_stats(filter_common, hours=hour_range)
                    logger.info(f" [ToolDebug] InfluxDB Tool Stats: {len(usage_stats)}건")
                except Exception as e:
                    logger.error(f" [ToolDebug] InfluxDB Tool Stats 조회 실패: {e}")

                # MongoDB 공구 사용량 조회 (추가된 로직)
                try:
                    tool_counts = await self.mongodb.get_current_tool_counts(filter_common)
                    logger.info(f" [ToolDebug] MongoDB Tool Counts 조회 결과: {len(tool_counts)}건")
                    if tool_counts:
                         logger.info(f" [ToolDebug] 상세: {[t.get('toolCode') for t in tool_counts]}")
                except Exception as e:
                    logger.error(f" [ToolDebug] MongoDB Tool Counts 조회 실패: {e}")
            else:
                logger.warning(" [ToolDebug] FilterCommon 생성 실패 - 필수 엔티티 누락")
            
            if machine_id:
                if tool_code:
                    tool = await self.mongodb.get_tool_by_code(machine_id, tool_code)
                    return {"tool": tool, "usage_stats": usage_stats, "tool_counts": tool_counts} if tool else {"usage_stats": usage_stats, "tool_counts": tool_counts}
                else:
                    tools = await self.mongodb.get_tools_by_machine(machine_id)
                    return {"tools": tools, "usage_stats": usage_stats, "tool_counts": tool_counts}
            return {"usage_stats": usage_stats, "tool_counts": tool_counts}
        except Exception as e:
            logger.error(f"공구 데이터 조회 실패: {e}")
            return {}
    
    async def _get_sensor_data(
        self, 
        filter_common: Optional[FilterCommon], 
        hours: int = 1,
        sensor_query_type: Optional[SensorQueryType] = None,
        target_field: Optional[str] = None
    ) -> Dict:
        """센서 데이터 조회 - target_field 기반 동적 라우팅"""
        if not filter_common:
            return {}
        
        try:
            result = {}
            
            # 동적 필드 파싱 (콤마 구분 지원)
            raw_target = target_field or "Load"
            target_fields = [t.strip() for t in raw_target.split(",")]
            
            measurement_map = {
                "CT": "cnc_product",
                # 필요시 추가 매핑
            }
            
            # sensor_query_type에 따른 분기 로직
            if sensor_query_type == SensorQueryType.CURRENT_STATUS:
                # 현재 상태만 조회
                result["current_status"] = await self.influxdb.get_current_status(filter_common)
                
            elif sensor_query_type == SensorQueryType.RUNNING_STATS:
                # 가동 중 통계 (첫 번째 필드 기준)
                field = target_fields[0]
                result["running_stats"] = await self.influxdb.get_running_stats(filter_common, hours=hours, field=field)
                
            elif sensor_query_type == SensorQueryType.RAW_STATS or sensor_query_type == SensorQueryType.CT_STATS: 
                # 전체 통계 (평균/최대/최소) - 다중 필드 지원
                days = max(1, hours // 24)
                
                for t_field in target_fields:
                    measurement = measurement_map.get(t_field)
                    stats_res = {}
                    key_type = ""
                    
                    if days > 84:  # 12주 초과
                        months = max(1, days // 30)
                        stats_res = await self.influxdb.get_monthly_stats(filter_common, months=months, field=t_field, measurement=measurement)
                        key_type = "monthly_stats"
                    elif days > 30:
                        weeks = max(1, days // 7)
                        stats_res = await self.influxdb.get_weekly_stats(filter_common, weeks=weeks, field=t_field, measurement=measurement)
                        key_type = "weekly_stats"
                    elif days > 1:
                        stats_res = await self.influxdb.get_daily_stats(filter_common, days=days, field=t_field, measurement=measurement)
                        key_type = "daily_stats"
                    else:
                        stats_res = await self.influxdb.get_raw_stats(filter_common, hours=hours, field=t_field, measurement=measurement)
                        key_type = "raw_stats"
                    
                    # 결과 저장 (stats_{Field}) - 포맷터에서 식별 용이하게
                    result[f"stats_{t_field}"] = {"type": key_type, "data": stats_res}
                
            elif sensor_query_type == SensorQueryType.TREND:
                # 트렌드 조회
                result["trend"] = await self.influxdb.get_raw_trend(filter_common, hours=hours, interval="1h", field=field, measurement=measurement)
                
            elif sensor_query_type == SensorQueryType.RUNTIME:
                # 가동 시간/률 - 기간별 자동 집계 단위 선택
                # ≤30일: 일별, 31-84일(12주): 주별, >84일: 월별
                days = max(1, hours // 24)
                if days > 84:  # 12주 초과
                    months = max(1, days // 30)
                    result["monthly_runtime"] = await self.influxdb.get_monthly_runtime(filter_common, months=months)
                elif days > 30:
                    weeks = max(1, days // 7)
                    result["weekly_runtime"] = await self.influxdb.get_weekly_runtime(filter_common, weeks=weeks)
                else:
                    result["daily_runtime"] = await self.influxdb.get_daily_runtime(filter_common, days=days)
                
            else:
                # 기본: 현재 상태 + 가동 중 통계
                result["current_status"] = await self.influxdb.get_current_status(filter_common)
                result["running_stats"] = await self.influxdb.get_running_stats(filter_common, hours=hours, field=field)
            
            return result
        except Exception as e:
            logger.error(f"센서 데이터 조회 실패: {e}")
            return {}

    async def _generate_answer(
        self, 
        request: QueryRequest, 
        context: HybridContext, 
        actual_question: str,
        history_key: str,
        start_time: float,
        intent: Optional[QueryIntent] = None
    ) -> QueryResponse:
        """LLM을 통한 하이브리드 답변 생성"""
        
        # 질문 유형 결정 (intent의 primary_source를 QuestionType 형식으로 변환)
        question_type = "DOCUMENT_QUERY"
        if intent:
            source_to_question_type = {
                DataSourceType.DOCUMENT: "DOCUMENT_QUERY",
                DataSourceType.PRODUCTION: "PRODUCTION_QUERY",
                DataSourceType.ABNORMAL: "ABNORMAL_QUERY",
                DataSourceType.MACHINE: "MACHINE_QUERY",
                DataSourceType.TOOL: "TOOL_QUERY",
                DataSourceType.RAW_SENSOR: "RAW_SENSOR_QUERY",
                DataSourceType.HYBRID: "HYBRID_QUERY",
            }
            question_type = source_to_question_type.get(intent.primary_source, "DOCUMENT_QUERY")
        
        logger.info(f"답변 생성 - 질문 유형: {question_type}")
        
        # 1. 하이브리드 정보 포맷팅
        info_blocks = []
        if context.document_context:
            info_blocks.append(f"[관련 문서 내용]\n{context.document_context}")
        if context.production_data:
            info_blocks.append(f"[생산 이력 데이터]\n{self._format_production_data(context.production_data)}")
        if context.abnormal_data:
            info_blocks.append(f"[이상감지 현황]\n{self._format_abnormal_data(context.abnormal_data)}")
        if context.machine_data:
            info_blocks.append(f"[설비 정보]\n{self._format_machine_data(context.machine_data)}")
        if context.tool_data:
            info_blocks.append(f"[공구 정보]\n{self._format_tool_data(context.tool_data)}")
        if context.sensor_data:
            info_blocks.append(f"[실시간 센서 데이터]\n{self._format_sensor_data(context.sensor_data)}")
            
        full_context = "\n\n".join(info_blocks)
        
        # 2. 답변 생성 호출 (질문 유형 포함)
        answer = await self.answer_generator.generate_intelligent_answer(
            request.question,
            full_context,
            actual_question,
            self.conversation_manager.get_recent_history(history_key),
            self.conversation_manager.format_history_for_prompt,
            question_type=question_type
        )
        
        # 3. 결과 구성
        confidence = self.retriever.calculate_confidence(context.document_results) if context.document_results else 0.8
        
        self.conversation_manager.add_to_history(
            history_key, request.question, answer, context.document_results, confidence
        )
        
        # 사용된 데이터 소스
        sources_used = []
        if context.document_context: sources_used.append("document")
        if context.production_data: sources_used.append("production")
        if context.abnormal_data: sources_used.append("abnormal")
        if context.machine_data: sources_used.append("machine")
        if context.tool_data: sources_used.append("tool")
        if context.sensor_data: sources_used.append("sensor")
        
        return QueryResponse(
            answer=answer,
            sources=context.document_results,
            confidence=confidence,
            processing_time=time.time() - start_time,
            metadata={"sources_used": sources_used}
        )
    
    # ============ 데이터 포맷팅 메서드 ============
    
    def _format_production_data(self, data: Dict) -> str:
        """생산 데이터 포맷팅"""
        if not data:
            return "데이터 없음"
        
        lines = []
        if "stats" in data:
            stats = data["stats"]
            lines.append(f"* 조회 기간 내 생산 수: {stats.get('count', 0)}건")
            ct = stats.get("ct", {})
            if ct.get("avg"):
                lines.append(f"* 평균 CT: {ct['avg']:.2f}초 (최대: {ct.get('max', 0):.2f}, 최소: {ct.get('min', 0):.2f})")
            loadsum = stats.get("loadSum", {})
            if loadsum.get("avg"):
                lines.append(f"* 평균 LoadSum: {loadsum['avg']:.2f}")
        
        if "recent_products" in data and data["recent_products"]:
            lines.append(f"\n최근 생산 이력 ({len(data['recent_products'])}건):")
            for p in data["recent_products"][:5]:
                lines.append(f"  - {p.get('productNo', 'N/A')}: 결과={p.get('productResult', '-')}")
        
        return "\n".join(lines)
    
    def _format_abnormal_data(self, data: Dict) -> str:
        """이상감지 데이터 포맷팅"""
        if not data:
            return "데이터 없음"
        
        lines = []
        if "summary" in data:
            summary = data["summary"]
            lines.append(f"* 총 이상감지 건수: {summary.get('total', 0)}건")
            by_code = summary.get("by_code", {})
            if by_code:
                lines.append(f"* 유형별 발생: {', '.join([f'{k}={v}건' for k, v in by_code.items()])}")
        
        if "recent_abnormals" in data and data["recent_abnormals"]:
            lines.append(f"\n최근 이상감지 이력 ({len(data['recent_abnormals'])}건):")
            for a in data["recent_abnormals"][:5]:
                lines.append(f"  - {a.get('abnormalCode', 'N/A')}: 값={a.get('abnormalValue', '-')}, 공구={a.get('abnormalTool', '-')}")
        
        return "\n".join(lines)
    
    def _format_machine_data(self, data: Dict) -> str:
        """설비 데이터 포맷팅"""
        if not data:
            return "데이터 없음"
        
        if "machine" in data:
            m = data["machine"]
            lines = [
                f"* 설비 코드: {m.get('machineCode', 'N/A')}",
                f"* 설비명: {m.get('machineName', 'N/A')}",
                f"* 공정: {m.get('opCode', 'N/A')}",
                f"* IP/Port: {m.get('machineIp', 'N/A')}:{m.get('machinePort', 'N/A')}"
            ]
            
            # 임계치 정보 (상세)
            if "threshold" in data and data["threshold"]:
                t = data["threshold"]
                lines.append("\n[임계치 설정]")
                lines.append(f"* CT 임계치: {t.get('minThresholdCt', 0):,.0f} ~ {t.get('maxThresholdCt', 0):,.0f}")
                lines.append(f"* LoadSum 임계치: {t.get('minThresholdLoad', 0):,.0f} ~ {t.get('maxThresholdLoad', 0):,.0f}")
                
                # 공구별 임계치
                tool_thresholds = []
                for i in range(1, 5):
                    key = f"tool{i}Threshold"
                    if key in t and t[key]:
                        tool_thresholds.append(f"T{i}: {t[key]}")
                if tool_thresholds:
                    lines.append(f"* 공구별 임계치: {', '.join(tool_thresholds)}")
                
                # 비고
                if t.get("remark"):
                    lines.append(f"* 비고: {t.get('remark')}")
                if t.get("selected"):
                    lines.append(f"* 선택 상태: {t.get('selected')}")
            
            # 공구 정보 (상세)
            if "tools" in data and data["tools"]:
                lines.append(f"\n[등록된 공구 ({len(data['tools'])}개)]")
                for tool in data["tools"]:
                    tool_info = f"* {tool.get('toolCode', 'N/A')}: {tool.get('toolName', 'N/A')}"
                    tool_info += f" (최대 {tool.get('maxCount', 0)}회"
                    if tool.get('warnRate'):
                        tool_info += f", 경고 {tool.get('warnRate')}%"
                    tool_info += ")"
                    if tool.get('subToolCode'):
                        tool_info += f" [서브코드: {tool.get('subToolCode')}]"
                    lines.append(tool_info)
            
            # 가동 시간 정보 (상세)
            if "runtime" in data and data["runtime"]:
                r = data["runtime"]
                hours = r.get('period_hours', 24)
                
                # 가동률에 따라 상태 아이콘 표시
                status_icon = "🟢" if r.get('operating_rate', 0) > 80 else "🟡" if r.get('operating_rate', 0) > 50 else "🔴"
                
                lines.append(f"\n[최근 {hours}시간 가동 현황]")
                lines.append(f"* 가동 시간: {r.get('runtime_hours', 0)}시간 ({r.get('runtime_minutes', 0)}분)")
                lines.append(f"* 가동률: {status_icon} {r.get('operating_rate', 0)}%")
            
            # 공구 사용량 (계산된 값)
            if "tool_counts" in data and data["tool_counts"]:
                lines.append(f"\n[공구 사용량 현황]")
                for tc in data["tool_counts"]:
                    # 상태에 따른 아이콘
                    status = tc.get('status', 'OK')
                    icon = "🟢" if status == "OK" else "🟡" if status == "WARN" else "🔴"
                    
                    lines.append(f"* {tc.get('toolCode')}: {tc.get('useCount', 0)}/{tc.get('maxCount', 0)}회 ({tc.get('usageRate', 0)}%) {icon}")
            
            return "\n".join(lines)
        
        if "machines" in data:
            lines = [f"총 {data.get('total_count', 0)}대 설비:"]
            for m in data["machines"]:
                machine_info = f"  - {m.get('machineCode', 'N/A')}: {m.get('machineName', 'N/A')}"
                
                # 해당 설비의 공구 정보도 표시
                tools = m.get("tools", [])
                if tools:
                    tool_names = [t.get('toolCode', 'N/A') for t in tools[:5]]  # 최대 5개만
                    machine_info += f" | 공구: {', '.join(tool_names)}"
                    if len(tools) > 5:
                        machine_info += f" 외 {len(tools)-5}개"
                else:
                    machine_info += " | 공구: 없음"
                
                # 공구 사용량 요약 표시 (추가)
                if "tool_counts" in m and m["tool_counts"]:
                    status_counts = {"OK": 0, "WARN": 0, "ERROR": 0}
                    for tc in m["tool_counts"]:
                        status = tc.get("status", "OK")
                        if status in status_counts:
                            status_counts[status] += 1
                    
                    icons = []
                    if status_counts["ERROR"] > 0: icons.append(f"🔴{status_counts['ERROR']}")
                    if status_counts["WARN"] > 0: icons.append(f"🟡{status_counts['WARN']}")
                    if status_counts["OK"] > 0: icons.append(f"🟢{status_counts['OK']}")
                    
                    if icons:
                        machine_info += f" | 상태: {' '.join(icons)}"
                
                lines.append(machine_info)
            return "\n".join(lines)
        
        return "데이터 없음"
    
    def _format_tool_data(self, data: Dict) -> str:
        """공구 데이터 포맷팅"""
        if not data:
            return "데이터 없음"
        
        lines = []
        
        # 사용 통계
        if "usage_stats" in data and data["usage_stats"]:
            stats = data["usage_stats"]
            period = stats[0].get('period_hours', 24) if stats else 24
            lines.append(f"[최근 {period}시간 공구 사용 통계]")
            for s in stats:
                 lines.append(f"* 공구 {s.get('tool_code')}: {s.get('total_use_count')}회 사용")
            lines.append("")
        
        if "tool" in data:
            t = data["tool"]
            lines.append(f"* 공구 코드: {t.get('toolCode', 'N/A')}")
            lines.append(f"* 공구명: {t.get('toolName', 'N/A')}")
            lines.append(f"* 최대 수명: {t.get('maxCount', 0)}회")
            return "\n".join(lines)
        
        if "tools" in data:
            lines.append(f"공구 목록 ({len(data['tools'])}개):")
            for t in data["tools"]:
                lines.append(f"  - {t.get('toolCode', 'N/A')}: {t.get('toolName', 'N/A')} (최대 {t.get('maxCount', 0)}회)")
            return "\n".join(lines)
            
        if lines:
            return "\n".join(lines)
        
        return "데이터 없음"
    
    def _format_sensor_data(self, data: Dict) -> str:
        """센서 데이터 포맷팅"""
        if not data:
            return "데이터 없음"
        
        lines = []
        
        # [NEW] 다중 필드 통계 처리 (stats_{Field})
        stats_keys = sorted([k for k in data.keys() if k.startswith("stats_")])
        for k in stats_keys:
            field_name = k.replace("stats_", "")
            item = data[k]
            p_type = item.get("type")
            res_data = item.get("data")
            
            if not res_data: continue
            
            # Daily Stats
            if p_type == "daily_stats":
                daily_list = res_data.get("daily", [])
                total = res_data.get("total", {})
                if daily_list:
                    lines.append(f"\n[일별 {field_name} 통계 (최근 {total.get('period_days', len(daily_list))}일)]")
                    for d in daily_list:
                        day = d.get("day", "N/A")
                        vals = []
                        if d.get("mean") is not None: vals.append(f"평균 {d['mean']}")
                        if d.get("max") is not None: vals.append(f"최대 {d['max']}")
                        if d.get("min") is not None: vals.append(f"최소 {d['min']}")
                        lines.append(f"* {day}: {', '.join(vals)}")
                if total:
                    lines.append(f"* 전체 요약: 평균 {total.get('mean')}, 최대 {total.get('max')}, 최소 {total.get('min')}")

            # Weekly Stats
            elif p_type == "weekly_stats":
                weekly_list = res_data.get("weekly", [])
                total = res_data.get("total", {})
                if weekly_list:
                    lines.append(f"\n[주별 {field_name} 통계 (최근 {total.get('period_weeks', len(weekly_list))}주)]")
                    for w in weekly_list:
                        week = w.get("week", "N/A")
                        vals = []
                        if w.get("mean") is not None: vals.append(f"평균 {w['mean']}")
                        if w.get("max") is not None: vals.append(f"최대 {w['max']}")
                        if w.get("min") is not None: vals.append(f"최소 {w['min']}")
                        lines.append(f"* {week}: {', '.join(vals)}")
                if total:
                    lines.append(f"* 전체 요약: 평균 {total.get('mean')}, 최대 {total.get('max')}, 최소 {total.get('min')}")

            # Monthly Stats
            elif p_type == "monthly_stats":
                monthly_list = res_data.get("monthly", [])
                total = res_data.get("total", {})
                if monthly_list:
                    lines.append(f"\n[월별 {field_name} 통계 (최근 {total.get('period_months', len(monthly_list))}개월)]")
                    for m in monthly_list:
                        month = m.get("month", "N/A")
                        vals = []
                        if m.get("mean") is not None: vals.append(f"평균 {m['mean']}")
                        if m.get("max") is not None: vals.append(f"최대 {m['max']}")
                        if m.get("min") is not None: vals.append(f"최소 {m['min']}")
                        lines.append(f"* {month}: {', '.join(vals)}")
                if total:
                    lines.append(f"* 전체 요약: 평균 {total.get('mean')}, 최대 {total.get('max')}, 최소 {total.get('min')}")

            # Raw Stats
            elif p_type == "raw_stats":
                stats = res_data
                if stats.get("mean") is not None:
                    lines.append(f"\n[전체 {field_name} 통계 ({stats.get('hours')}시간)]")
                    lines.append(f"* 평균: {stats['mean']:.1f}, 최대: {stats['max']:.1f}, 최소: {stats['min']:.1f}")

        if "current_status" in data:
            status = data["current_status"]
            lines.append(f"* 가동 상태: {status.get('run_status', 'N/A')}")
            lines.append(f"* 현재 부하: {status.get('current_load', 'N/A')}")
            lines.append(f"* 이송 속도: {status.get('current_feed', 'N/A')}")
            lines.append(f"* FOV: {status.get('fov', 'N/A')}%, SOV: {status.get('sov', 'N/A')}%")
        
        if "running_stats" in data:
            stats = data["running_stats"]
            if stats.get("mean") is not None:
                 lines.append(f"\n[가동 중 Load 통계 ({stats.get('hours')}시간)]")
                 lines.append(f"* 평균: {stats['mean']:.1f}")
                 lines.append(f"* 최대: {stats['max']:.1f}")
                 lines.append(f"* 최소: {stats['min']:.1f}")
        
        if "raw_stats" in data:
            stats = data["raw_stats"]
            if stats.get("mean") is not None:
                 lines.append(f"\n[전체 Load 통계 ({stats.get('hours')}시간)]")
                 lines.append(f"* 평균: {stats['mean']:.1f}")
                 lines.append(f"* 최대: {stats['max']:.1f}")
                 lines.append(f"* 최소: {stats['min']:.1f}")
        
        if "trend" in data:
            trend = data["trend"]
            if trend:
                lines.append(f"\n[Load 트렌드 ({len(trend)}개 포인트)]")
                for t in trend[:10]:  # 최대 10개만 표시
                    time_str = t.get('time', 'N/A')
                    if hasattr(time_str, 'strftime'):
                        time_str = time_str.strftime("%H:%M")
                    lines.append(f"* {time_str}: {t.get('value', 'N/A'):.1f}")
                if len(trend) > 10:
                    lines.append(f"  ... 외 {len(trend) - 10}개")
        
        if "runtime" in data:
            rt = data["runtime"]
            lines.append(f"\n[가동 시간/률 ({rt.get('period_hours', 24)}시간)]")
            lines.append(f"* 가동 시간: {rt.get('runtime_hours', 0)}시간 ({rt.get('runtime_minutes', 0)}분)")
            lines.append(f"* 가동률: {rt.get('operating_rate', 0)}%")
        
        if "daily_runtime" in data:
            daily_data = data["daily_runtime"]
            daily_list = daily_data.get("daily", [])
            total = daily_data.get("total", {})
            
            if daily_list:
                lines.append(f"\n[일별 가동률 (최근 {total.get('period_days', len(daily_list))}일)]")
                for d in daily_list:
                    date_str = d.get("date", "N/A")
                    hours = d.get("runtime_hours", 0)
                    rate = d.get("operating_rate", 0)
                    # 가동률에 따른 상태 아이콘
                    icon = "🟢" if rate > 80 else "🟡" if rate > 50 else "🔴"
                    lines.append(f"* {date_str}: {hours}시간 ({rate}%) {icon}")
            
            if total:
                lines.append(f"\n[총 가동률]")
                lines.append(f"* 기간: {total.get('period_days', 0)}일")
                lines.append(f"* 총 가동 시간: {total.get('runtime_hours', 0)}시간")
                lines.append(f"* 평균 가동률: {total.get('operating_rate', 0)}%")
        
        if "weekly_runtime" in data:
            weekly_data = data["weekly_runtime"]
            weekly_list = weekly_data.get("weekly", [])
            total = weekly_data.get("total", {})
            
            if weekly_list:
                lines.append(f"\n[주별 가동률 (최근 {total.get('period_weeks', len(weekly_list))}주)]")
                for w in weekly_list:
                    week_start = w.get("week_start", "N/A")
                    hours = w.get("runtime_hours", 0)
                    rate = w.get("operating_rate", 0)
                    icon = "🟢" if rate > 80 else "🟡" if rate > 50 else "🔴"
                    lines.append(f"* {week_start} 주: {hours}시간 ({rate}%) {icon}")
            
            if total:
                lines.append(f"\n[총 가동률]")
                lines.append(f"* 기간: {total.get('period_weeks', 0)}주")
                lines.append(f"* 총 가동 시간: {total.get('runtime_hours', 0)}시간")
                lines.append(f"* 평균 가동률: {total.get('operating_rate', 0)}%")
        
        if "monthly_runtime" in data:
            monthly_data = data["monthly_runtime"]
            monthly_list = monthly_data.get("monthly", [])
            total = monthly_data.get("total", {})
            
            if monthly_list:
                lines.append(f"\n[월별 가동률 (최근 {total.get('period_months', len(monthly_list))}개월)]")
                for m in monthly_list:
                    month = m.get("month", "N/A")
                    hours = m.get("runtime_hours", 0)
                    rate = m.get("operating_rate", 0)
                    icon = "🟢" if rate > 80 else "🟡" if rate > 50 else "🔴"
                    lines.append(f"* {month}: {hours}시간 ({rate}%) {icon}")
            
            if total:
                lines.append(f"\n[총 가동률]")
                lines.append(f"* 기간: {total.get('period_months', 0)}개월")
                lines.append(f"* 총 가동 시간: {total.get('runtime_hours', 0)}시간")
                lines.append(f"* 평균 가동률: {total.get('operating_rate', 0)}%")
        
        # 기간별 Stats 포맷팅
        if "daily_stats" in data:
            daily_data = data["daily_stats"]
            daily_list = daily_data.get("daily", [])
            total = daily_data.get("total", {})
            
            if daily_list:
                lines.append(f"\n[일별 {total.get('field', 'Load')} 통계 (최근 {total.get('period_days', len(daily_list))}일)]")
                for d in daily_list:
                    day = d.get("day", "N/A")
                    mean = d.get("mean", 0) or 0
                    max_v = d.get("max", 0) or 0
                    min_v = d.get("min", 0) or 0
                    lines.append(f"* {day}: 평균 {mean}, 최대 {max_v}, 최소 {min_v}")
            
            if total:
                lines.append(f"\n[전체 통계]")
                lines.append(f"* 평균: {total.get('mean', 0)}, 최대: {total.get('max', 0)}, 최소: {total.get('min', 0)}")
        
        if "weekly_stats" in data:
            weekly_data = data["weekly_stats"]
            weekly_list = weekly_data.get("weekly", [])
            total = weekly_data.get("total", {})
            
            if weekly_list:
                lines.append(f"\n[주별 {total.get('field', 'Load')} 통계 (최근 {total.get('period_weeks', len(weekly_list))}주)]")
                for w in weekly_list:
                    week = w.get("week", "N/A")
                    mean = w.get("mean", 0) or 0
                    max_v = w.get("max", 0) or 0
                    min_v = w.get("min", 0) or 0
                    lines.append(f"* {week}: 평균 {mean}, 최대 {max_v}, 최소 {min_v}")
            
            if total:
                lines.append(f"\n[전체 통계]")
                lines.append(f"* 평균: {total.get('mean', 0)}, 최대: {total.get('max', 0)}, 최소: {total.get('min', 0)}")
        
        if "monthly_stats" in data:
            monthly_data = data["monthly_stats"]
            monthly_list = monthly_data.get("monthly", [])
            total = monthly_data.get("total", {})
            
            if monthly_list:
                lines.append(f"\n[월별 {total.get('field', 'Load')} 통계 (최근 {total.get('period_months', len(monthly_list))}개월)]")
                for m in monthly_list:
                    month = m.get("month", "N/A")
                    mean = m.get("mean", 0) or 0
                    max_v = m.get("max", 0) or 0
                    min_v = m.get("min", 0) or 0
                    lines.append(f"* {month}: 평균 {mean}, 최대 {max_v}, 최소 {min_v}")
            
            if total:
                lines.append(f"\n[전체 통계]")
                lines.append(f"* 평균: {total.get('mean', 0)}, 최대: {total.get('max', 0)}, 최소: {total.get('min', 0)}")

        # CT 통계 포맷팅
        if "daily_ct_stats" in data:
            ct_data = data["daily_ct_stats"]
            daily_list = ct_data.get("daily", [])
            total = ct_data.get("total", {})
            
            if daily_list:
                lines.append(f"\n[일별 CT(Cycle Time) 통계 (최근 {total.get('period_days', len(daily_list))}일)]")
                for d in daily_list:
                    lines.append(f"* {d.get('date')}: 평균 {d.get('mean_ct', 0)}초")
            
            if total:
                lines.append(f"\n[전체 CT 통계]")
                lines.append(f"* 평균 CT: {total.get('mean', 0)}초")
                lines.append(f"* 최대 CT: {total.get('max', 0)}초, 최소 CT: {total.get('min', 0)}초")
                lines.append(f"* 총 생산 횟수(집계): {total.get('count', 0)}회")

        if "weekly_ct_stats" in data:
            ct_data = data["weekly_ct_stats"]
            weekly_list = ct_data.get("weekly", [])
            total = ct_data.get("total", {})
            
            if weekly_list:
                lines.append(f"\n[주별 CT(Cycle Time) 통계 (최근 {total.get('period_weeks', len(weekly_list))}주)]")
                for w in weekly_list:
                    lines.append(f"* {w.get('week_start')} 주: 평균 {w.get('mean_ct', 0)}초")
            
            if total:
                lines.append(f"\n[전체 CT 통계]")
                lines.append(f"* 평균 CT: {total.get('mean', 0)}초")
                lines.append(f"* 최대 CT: {total.get('max', 0)}초, 최소 CT: {total.get('min', 0)}초")
                lines.append(f"* 총 생산 횟수(집계): {total.get('count', 0)}회")

        if "monthly_ct_stats" in data:
            ct_data = data["monthly_ct_stats"]
            monthly_list = ct_data.get("monthly", [])
            total = ct_data.get("total", {})
            
            if monthly_list:
                lines.append(f"\n[월별 CT(Cycle Time) 통계 (최근 {total.get('period_months', len(monthly_list))}개월)]")
                for m in monthly_list:
                    lines.append(f"* {m.get('month')}: 평균 {m.get('mean_ct', 0)}초")
            
            if total:
                lines.append(f"\n[전체 CT 통계]")
                lines.append(f"* 평균 CT: {total.get('mean', 0)}초")
                lines.append(f"* 최대 CT: {total.get('max', 0)}초, 최소 CT: {total.get('min', 0)}초")
                lines.append(f"* 총 생산 횟수(집계): {total.get('count', 0)}회")
        
        return "\n".join(lines)

    async def _handle_general_conversation(self, original_q: str, aware_q: str, hist_key: str, start_time: float) -> QueryResponse:
        """데이터 소스가 없을 시 일반 대화로 응답"""
        answer = await self.answer_generator.generate_general_conversation_response(aware_q)
        self.conversation_manager.add_to_history(hist_key, original_q, answer, [], 0.8)
        return QueryResponse(answer=answer, sources=[], confidence=0.8, processing_time=time.time() - start_time)

    def _create_simple_response(self, text: str, req: QueryRequest, key: str, start: float) -> QueryResponse:
        """간단한 응답 생성 및 히스토리 기록"""
        self.conversation_manager.add_to_history(key, req.question, text, [], 1.0)
        return QueryResponse(answer=text, sources=[], confidence=1.0, processing_time=time.time() - start)

    async def health_check(self) -> Dict[str, Any]:
        """모든 컴포넌트 상태 확인"""
        return {
            "status": "healthy" if self._initialized else "initializing",
            "mongodb": await self.mongodb.health_check(),
            "influxdb": await self.influxdb.health_check(),
            "retriever": await self.retriever.vector_store.health_check()
        }

    async def cleanup(self):
        """자원 정리"""
        await asyncio.gather(
            self.mongodb.close(),
            self.retriever.vector_store.cleanup(),
            self.gemini.cleanup(),
            return_exceptions=True
        )
        self._initialized = False


# 전역 싱글톤 인스턴스
hybrid_rag_engine = HybridRAGEngine()
