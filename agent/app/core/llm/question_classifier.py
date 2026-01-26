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
    USER_QUERY = "USER_QUERY"
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
        
        system_instruction = """당신은 제조 현장(Smart Factory)의 데이터 분석 전문가입니다. 
사용자의 질문을 분석하여 JSON 형식으로 의도를 추출해야 합니다.

질문 유형(primary_type):
- PRODUCTION_QUERY: 생산량, 생산 이력, 제품 정보, 사이클 타임(CT) 등 생산 관련
- ABNORMAL_QUERY: 알람, 에러, 불량, 이상 징후, 고장 내역 등
- RAW_SENSOR_QUERY: 실시간 센서(부하, 속도), 가동 상태, 가동 시간/률 등
- TOOL_QUERY: 공구(Tool) 정보, 공구 수명, 공구 사용량 등
- MACHINE_QUERY: 설비 마스터 정보, 임계치(Threshold) 설정 등
- USER_QUERY: 사용자 계정, 권한 정보 등
- GREETING: 인사말, 자기소개 요청 등
- DOCUMENT_QUERY: 위의 범주에 해당하지 않는 일반적인 매뉴얼이나 가이드 질문

시간 범위(time_range):
- 질문에 언급된 시간을 "24h", "7d", "30d", "1h" 등의 형식으로 추출하세요.
- 언급이 없으면 "24h"를 기본값으로 사용하세요.

엔티티(entities):
- machine_id: 설비 코드 (예: M01, CNC-01)
- product_no: 제품 번호 또는 로트 번호
- tool_code: 공구 번호 (예: T01)

반드시 다음 JSON 구조로만 응답하세요:
{
  "primary_type": "유형",
  "secondary_types": ["보조유형"],
  "entities": {"machine_id": null, "product_no": null, "tool_code": null},
  "time_range": "24h",
  "sensor_query_type": "CURRENT_STATUS | RUNNING_STATS | RAW_STATS | TREND | RUNTIME | CT_STATS | null",
  "target_field": "조회대상필드명(예: Load, Run, CT, Feed)"
}"""

        user_prompt = f"질문: \"{question}\""

        

        try:
            if hasattr(self.gemini_service, 'client') and self.gemini_service.client:
                def analyze():
                    # JSON 모드 및 시스템 프롬프트 적용을 위한 설정
                    from google.genai import types
                    
                    # 새로운 모델 인스턴스 생성 (시스템 지침 포함) 또는 
                    # 현재 SDK 버전에 맞게 컨텐츠 구성
                    # 시스템 프롬프트를 텍스트 파트 앞에 추가하는 방식이 호환성이 높음
                    full_prompt = f"{system_instruction}\n\n{user_prompt}"
                    
                    config = types.GenerateContentConfig(
                        temperature=0.1,
                        response_mime_type="application/json"
                    )
                    
                    response = self.gemini_service.client.models.generate_content(
                        model=self.gemini_service.model_name,
                        contents=full_prompt,
                        config=config
                    )
                    
                    # Gemini 응답 텍스트 추출 (candidates 처리)
                    try:
                        text = self.gemini_service._extract_text_from_response(response)
                        if not text:
                            logger.warning("Gemini 응답 텍스트가 비어있음")
                            return None
                    except Exception as e:
                        logger.warning(f"Gemini 응답 추출 실패: {e}")
                        return None
                    
                    text = text.replace("```json", "").replace("```", "").strip()
                    
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
                        logger.warning(f"JSON 파싱 실패: {e}, Raw: {text[:200]}...")
                        return None
                
                result = await asyncio.to_thread(analyze)
                if result:
                    return result
                    
        except Exception as e:
            logger.warning(f"질문 분석 실패: {e}")
        
        # Fallback: 키워드 기반 분류
        logger.info("LLM 분석 실패 - 키워드 기반 분류로 전환")
        return self._keyword_based_classification(question)
    
    def _keyword_based_classification(self, question: str) -> QuestionAnalysisResult:
        """키워드 기반 질문 분류 (LLM 실패시 폴백)"""
        q_lower = question.lower()
        
        # 키워드 매칭
        abnormal_keywords = ["이상", "알람", "에러", "불량", "문제", "고장", "abnormal", "alarm", "error"]
        production_keywords = ["생산", "제품", "생산량", "제조", "생산품", "product"]
        machine_keywords = ["설비", "장비", "기계", "임계치", "threshold", "machine"]
        tool_keywords = ["공구", "툴", "tool"]
        sensor_keywords = ["부하", "load", "ct", "가동", "센서", "운전", "속도"]
        user_keywords = ["사용자", "유저", "계정", "등록된", "user"]
        
        # 우선 순위 기반 매칭
        primary = QuestionType.DOCUMENT_QUERY  # 기본값
        secondaries = []
        
        # 이상감지 - 최우선
        if any(kw in q_lower for kw in abnormal_keywords):
            primary = QuestionType.ABNORMAL_QUERY
            if any(kw in q_lower for kw in production_keywords):
                secondaries.append(QuestionType.PRODUCTION_QUERY)
        
        # 생산 (이상이 없을 때)
        elif any(kw in q_lower for kw in production_keywords):
            primary = QuestionType.PRODUCTION_QUERY
        
        # 센서
        elif any(kw in q_lower for kw in sensor_keywords):
            primary = QuestionType.RAW_SENSOR_QUERY
        
        # 공구
        elif any(kw in q_lower for kw in tool_keywords):
            primary = QuestionType.TOOL_QUERY
        
        # 설비
        elif any(kw in q_lower for kw in machine_keywords):
            primary = QuestionType.MACHINE_QUERY
        
        # 사용자
        elif any(kw in q_lower for kw in user_keywords):
            primary = QuestionType.USER_QUERY
        
        logger.info(f"키워드 기반 분류: {primary}, 보조: {secondaries}")
        
        return QuestionAnalysisResult(
            primary_type=primary,
            secondary_types=secondaries,
            entities={},
            time_range="24h"
        )


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
