"""
질문 분류 및 분석 모듈
사용자 질문을 분석하여 의도, 엔티티, 시간 범위를 통합 추출합니다.
"""
import logging
import asyncio
import json
from typing import Optional, List, Dict, Any
from enum import Enum
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


class QuestionType(str, Enum):
    """질문 유형 열거형"""
    GREETING = "GREETING"
    DOCUMENT_LIST = "DOCUMENT_LIST"
    SYSTEM_STATUS = "SYSTEM_STATUS"
    DOCUMENT_QUERY = "DOCUMENT_QUERY"
    PRODUCTION_QUERY = "PRODUCTION_QUERY"
    ABNORMAL_QUERY = "ABNORMAL_QUERY"
    MACHINE_QUERY = "MACHINE_QUERY"
    TOOL_QUERY = "TOOL_QUERY"
    RAW_SENSOR_QUERY = "RAW_SENSOR_QUERY"
    HYBRID_QUERY = "HYBRID_QUERY"


class SensorQueryType(str, Enum):
    """센서 쿼리 유형 열거형"""
    CURRENT_STATUS = "CURRENT_STATUS"      # 현재 상태 조회
    RUNNING_STATS = "RUNNING_STATS"        # 가동 중 통계
    RAW_STATS = "RAW_STATS"                # 전체 통계 (평균/최대/최소)
    TREND = "TREND"                        # 트렌드 조회
    RUNTIME = "RUNTIME"                    # 가동 시간/률
    CT_STATS = "CT_STATS"                  # CT(Cycle Time/사이클타임) 통계


@dataclass
class QuestionAnalysisResult:
    """질문 분석 통합 결과"""
    primary_type: QuestionType
    secondary_types: List[QuestionType] = field(default_factory=list)
    entities: Dict[str, Any] = field(default_factory=dict)
    time_range: str = "24h"  # 기본값 "24h"
    sensor_query_type: Optional[SensorQueryType] = None  # 센서 쿼리 세부 유형
    target_field: Optional[str] = None  # 조회 대상 필드 (예: "CT", "Load", "Run")


class QuestionClassifier:
    """질문 유형을 분류하고 메타 정보를 추출하는 클래스"""
    
    def __init__(self, gemini_service, vector_store):
        self.gemini_service = gemini_service
        self.vector_store = vector_store
    
    async def handle_meta_questions(self, question: str, user_id: str) -> Optional[str]:
        """메타 질문 처리"""
        try:
            # 메타 질문 판단은 가볍게 처리하거나 분석 결과 이용
            # 여기서는 classify_question 호출 비용이 크므로, 간단한 키워드나 별도 로직을 쓸 수도 있지만
            # 일단 통합 분석 결과를 사용
            result = await self.classify_question(question)
            
            if result.primary_type == QuestionType.GREETING:
                return await self._handle_greeting(question)
            elif result.primary_type == QuestionType.DOCUMENT_LIST:
                return await self._handle_document_list_request(user_id)
            elif result.primary_type == QuestionType.SYSTEM_STATUS:
                return await self._handle_system_status_request()
            
            return None
                
        except Exception as e:
            logger.warning(f"메타 질문 처리 실패: {e}")
            return None
    
    async def classify_question(self, question: str) -> QuestionAnalysisResult:
        """LLM을 사용하여 질문 분석 (의도, 엔티티, 시간, 타겟 필드 통합 추출)"""
        
        prompt = f"""사용자의 질문을 심층 분석하여 다음 정보를 JSON 형식으로 추출하세요.
질문: "{question}"

1. **질문 유형 (Query Types)**:
   - PRODUCTION_QUERY: 생산 수량, 생산 이력, 제품 정보 조회. ("가동 시간", "CT"는 제외)
   - ABNORMAL_QUERY: 알람, 에러, 불량, 이상 내역.
   - RAW_SENSOR_QUERY: 설비의 센서 데이터, 가동 상태, 부하, 가동 시간, CT(Cycle Time).
   - TOOL_QUERY: 공구 정보, 공구 사용량, 공구 수명, 공구 교체, 툴 라이프.
   - MACHINE_QUERY: 설비 정보, 설비 목록, **임계치**, 설정값, 설비 스펙.
   - **중요 규칙**: 
     1. "가동 시간", "CT", "부하" 등 센서/상태 관련 질문 -> **RAW_SENSOR_QUERY**.
     2. "생산량", "몇 개 만들었어" -> **PRODUCTION_QUERY**.
     3. "공구 사용량", "툴 수명", "공구 교체" -> **TOOL_QUERY**.
     4. "임계치", "threshold", "설정값", "설비 정보" -> **MACHINE_QUERY**.

2. **엔티티 (Entities)**:
   - machine_id: 설비 코드/번호 (예: 10-1, CNC_01). "설비", "장비" 등의 단어 제외하고 코드만 추출.
   - product_no: 생산 번호/제품 번호.
   - tool_code: 공구 번호 (예: T1, 505).
   - workshop_id, line_id, op_code: 언급되었을 경우만 추출.
   - abnormal_code: 언급된 이상 유형.
   - "전체", "모든" -> "ALL" 등의 값으로 매핑 가능.

3. **시간 범위 (Time Range)**:
   - 질문에 언급된 기간을 찾아 표준 형식으로 변환 (Default: "24h").
   - 1년->"365d", 1개월->"30d", 1주->"7d", 어제->"48h" 등.

4. **센서 쿼리 유형 (Sensor Query Type)**:
   - CURRENT_STATUS: 현재 상태
   - RAW_STATS: 통계 (평균/최대/최소), 부하, CT 등 데이터값 조회
   - TREND: 트렌드/추이
   - RUNTIME: 가동 시간/률

5. **조회 대상 필드 (target_field)**:
   - 질문이 무엇을 묻는지에 따라 표준화된 필드명 추출.
   - **"CT", "사이클타임" -> "CT"**
   - **"부하", "로드" -> "Load"**
   - **"가동", "운전" -> "Run"**
   - **"속도", "피드" -> "Feed"**
   - **"회전수", "RPM" -> "SpindleSpeed"** (또는 적절한 필드명)
   - 언급 없으면 null.
   - **중요**: 여러 항목 조회 시 콤마(,)로 구분하여 반환 (예: "CT, Load"). "CT랑 부하" -> "CT, Load".

**출력 형식 (JSON Only):**
{{
  "primary_type": "QUERY_TYPE",
  "secondary_types": ["QUERY_TYPE", ...],
  "entities": {{ ... }},
  "time_range": "24h",
  "sensor_query_type": "RAW_STATS",
  "target_field": "CT"
}}
"""

        try:
            if hasattr(self.gemini_service, 'model') and self.gemini_service.model:
                def analyze():
                    response = self.gemini_service.model.generate_content(prompt)
                    text = response.text.replace("```json", "").replace("```", "").strip()
                    try:
                        data = json.loads(text)
                        
                        # Types
                        p_str = data.get("primary_type", "DOCUMENT_QUERY")
                        if p_str not in QuestionType.__members__: p_str = "DOCUMENT_QUERY"
                        primary = QuestionType(p_str)
                        
                        secondaries = []
                        for t in data.get("secondary_types", []):
                            if t in QuestionType.__members__:
                                secondaries.append(QuestionType(t))
                        
                        # Entities
                        entities = data.get("entities", {}) or {}
                        cleaned_entities = {k: v for k, v in entities.items() if v}
                        
                        # Time Range
                        time_range = data.get("time_range", "24h")
                        
                        # Sensor Query Type
                        sensor_query_type = None
                        sqt_str = data.get("sensor_query_type")
                        if sqt_str and sqt_str in SensorQueryType.__members__:
                            sensor_query_type = SensorQueryType(sqt_str)
                            
                        # Target Field
                        target_field = data.get("target_field")
                        
                        return QuestionAnalysisResult(
                            primary_type=primary,
                            secondary_types=secondaries,
                            entities=cleaned_entities,
                            time_range=time_range,
                            sensor_query_type=sensor_query_type,
                            target_field=target_field
                        )
                    except Exception as e:
                        logger.warning(f"JSON 파싱 실패: {e}, Raw: {text}")
                        return QuestionAnalysisResult(primary_type=QuestionType.DOCUMENT_QUERY)
                
                return await asyncio.to_thread(analyze)
        except Exception as e:
            logger.warning(f"질문 분석 실패: {e}")
        
        return QuestionAnalysisResult(primary_type=QuestionType.DOCUMENT_QUERY)

    async def _handle_greeting(self, question: str) -> str:
        prompt = f"""사용자("{question}")에게 제조 현장 AI 어시스턴트로서 짧고 친근하게 인사하고, 
        생산, 설비, 이상감지 등에 대해 도움을 줄 수 있음을 안내하세요."""
        try:
            return await asyncio.to_thread(lambda: self.gemini_service.model.generate_content(prompt).text.strip())
        except:
            return "안녕하세요! 생산 현황이나 설비 상태에 대해 궁금한 점이 있으신가요?"

    async def _handle_document_list_request(self, user_id: str) -> str:
        try:
            collections = await self.vector_store.list_user_documents(user_id)
            if not collections: return "업로드된 문서가 없습니다."
            return "문서 목록:\n" + "\n".join([f"- {d.get('file_name','Unknown')}" for d in collections])
        except:
            return "문서 목록 조회 실패."

    async def _handle_system_status_request(self) -> str:
        return "✅ 시스템 정상 작동 중 (생산/이상/설비/공구/센서 데이터 조회 가능)"


def create_question_classifier(gemini_service, vector_store) -> QuestionClassifier:
    return QuestionClassifier(gemini_service, vector_store)
