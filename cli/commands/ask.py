"""
질문 관련 명령어
"""
import requests
import json
import re
from config import BASE_URL, API_ENDPOINT, STREAM_ENDPOINT

try:
    from tabulate import tabulate
    HAS_TABULATE = True
except ImportError:
    HAS_TABULATE = False

def format_answer_for_cli(answer: str) -> str:
    """CLI 표시용으로 답변 포맷팅 (표 변환 등)"""
    if not HAS_TABULATE:
        return answer
    
    # Markdown 표를 찾아서 ASCII 표로 변환
    lines = answer.split('\n')
    formatted_lines = []
    table_lines = []
    in_table = False
    
    for line in lines:
        stripped = line.strip()
        
        # 표 시작 감지 (|로 시작하고 |가 2개 이상)
        if stripped.startswith('|') and stripped.count('|') >= 2 and not in_table:
            in_table = True
            table_lines = [line]
        elif in_table and stripped.startswith('|'):
            table_lines.append(line)
        elif in_table and not stripped.startswith('|'):
            # 표 끝
            in_table = False
            if table_lines:
                ascii_table = convert_markdown_table_to_ascii(table_lines)
                formatted_lines.append(ascii_table)
                formatted_lines.append("")  # 빈 줄 추가
            formatted_lines.append(line)
        else:
            formatted_lines.append(line)
    
    # 마지막 표 처리
    if in_table and table_lines:
        ascii_table = convert_markdown_table_to_ascii(table_lines)
        formatted_lines.append(ascii_table)
    
    return '\n'.join(formatted_lines)

def convert_markdown_table_to_ascii(table_lines: list) -> str:
    """Markdown 표 라인들을 ASCII 표로 변환"""
    if not HAS_TABULATE or not table_lines:
        return '\n'.join(table_lines)
    
    try:
        # 헤더와 데이터 분리
        headers = []
        data = []
        
        for i, line in enumerate(table_lines):
            cells = [cell.strip() for cell in line.split('|')[1:-1]]  # | 제거
            
            if i == 0:
                headers = cells
            elif i == 1 and all('-' in cell for cell in cells):
                # 구분선은 무시
                continue
            else:
                data.append(cells)
        
        if headers and data:
            # tabulate로 ASCII 표 생성
            return tabulate(data, headers=headers, tablefmt="grid")
        else:
            return '\n'.join(table_lines)
            
    except Exception as e:
        # 변환 실패시 원본 반환
        return '\n'.join(table_lines)

def get_confidence_description(confidence: float) -> str:
    """신뢰도 점수에 따른 설명 반환"""
    if confidence >= 0.8:
        return "높음 (문서 기반 답변 또는 일반 대화)"
    elif confidence >= 0.6:
        return "보통 (관련 문서 다수 발견)"
    elif confidence >= 0.3:
        return "낮음 (관련성이 낮은 문서만 발견)"
    else:
        return "매우 낮음 (관련 문서 없음)"

def ask_question(question: str, stream: bool = False, user_id: str = "cli_user") -> None:
    """질문을 RAG Agent에 전송하고 응답을 출력합니다"""
    
    payload = {
        "question": question,
        "user_id": user_id
    }
    
    try:
        if stream:
            # 스트리밍 모드
            print("🔄 처리 중...")
            print("=" * 50)
            
            response = requests.post(
                STREAM_ENDPOINT,
                json=payload,
                headers={"Content-Type": "application/json"},
                stream=True
            )
            
            if response.status_code != 200:
                print(f"❌ 오류: {response.status_code} - {response.text}")
                return
            
            answer_parts = []
            
            for line in response.iter_lines(decode_unicode=True):
                if line.startswith('data: '):
                    data_str = line[6:]  # 'data: ' 제거
                    
                    if data_str == '[DONE]':
                        break
                    
                    try:
                        data = json.loads(data_str)
                        
                        if data['type'] == 'start':
                            print(f"🔄 {data['message']}")
                        elif data['type'] == 'progress':
                            print(f"⏳ {data['message']}")
                        elif data['type'] == 'chunk':
                            print(data['content'], end='', flush=True)
                            answer_parts.append(data['content'])
                        elif data['type'] == 'sources':
                            if data['sources']:
                                print(f"\n\n📚 참조 문서 ({len(data['sources'])}개):")
                                for i, source in enumerate(data['sources'][:3], 1):
                                    print(f"  {i}. {source.get('file_path', 'Unknown')} (관련도: {source.get('relevance_score', 0):.2f})")
                        elif data['type'] == 'complete':
                            confidence = data['confidence']
                            confidence_desc = get_confidence_description(confidence)
                            print(f"\n\n✅ 완료 (신룰도: {confidence:.2f} - {confidence_desc})")
                        elif data['type'] == 'error':
                            print(f"\n❌ 오류: {data['message']}")
                            
                    except json.JSONDecodeError:
                        continue
        
        else:
            # 일반 모드
            print("🔄 처리 중...")
            
            response = requests.post(
                API_ENDPOINT,
                json=payload,
                headers={"Content-Type": "application/json"}
            )
            
            if response.status_code == 200:
                data = response.json()
                print("=" * 50)
                formatted_answer = format_answer_for_cli(data['answer'])
                print(f"💬 답변:\n{formatted_answer}")
                
                if data['sources']:
                    print(f"\n📚 참조 문서 ({len(data['sources'])}개):")
                    for i, source in enumerate(data['sources'][:3], 1):
                        print(f"  {i}. {source.get('file_path', 'Unknown')} (관련도: {source.get('relevance_score', 0):.2f})")
                
                # 신뢰도 설명과 함께 출력
                confidence = data['confidence']
                confidence_desc = get_confidence_description(confidence)
                
                print(f"\n✅ 신뢰도: {confidence:.2f} ({confidence_desc})")
                
            else:
                print(f"❌ 오류: {response.status_code} - {response.text}")
                
    except requests.exceptions.ConnectionError:
        print("❌ 연결 오류: RAG Agent 서비스가 실행 중인지 확인하세요.")
        print("   서비스 시작: python -m uvicorn app.main:app --reload")
    except Exception as e:
        print(f"❌ 예상치 못한 오류: {e}")
