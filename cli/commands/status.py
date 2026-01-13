"""
시스템 상태 확인 관련 명령어
"""
import requests
from config import BASE_URL

def check_vector_status() -> None:
    """벡터 DB(Qdrant) 상태를 확인합니다"""
    
    try:
        print("🔍 벡터 DB 상태 확인 중...")
        
        response = requests.get(f"{BASE_URL}/api/v1/vector-status")
        
        if response.status_code == 200:
            data = response.json()
            print("=" * 50)
            print(f"🟢 Qdrant 상태: {data['qdrant_status']}")
            
            if data['collection_info']:
                info = data['collection_info']
                print(f"📊 컬렉션 정보:")
                for key, value in info.items():
                    print(f"   {key}: {value}")
            
            print(f"💬 메시지: {data['message']}")
        else:
            print(f"❌ 상태 확인 실패: {response.status_code} - {response.text}")
            
    except requests.exceptions.ConnectionError:
        print("❌ 연결 오류: RAG Agent 서비스가 실행 중인지 확인하세요.")
    except Exception as e:
        print(f"❌ 예상치 못한 오류: {e}")

def test_vector_search(query: str = "테스트") -> None:
    """벡터 검색 테스트를 수행합니다"""
    
    try:
        print(f"🔍 벡터 검색 테스트: '{query}'")
        
        response = requests.get(
            f"{BASE_URL}/api/v1/search-test",
            params={"query": query}
        )
        
        if response.status_code == 200:
            data = response.json()
            print("=" * 50)
            print(f"📋 검색 결과: {data['results_count']}개 발견")
            
            if data['results']:
                for i, result in enumerate(data['results'], 1):
                    print(f"\n{i}. 점수: {result['score']:.3f}")
                    print(f"   내용: {result['content']}")
                    if result['metadata'].get('original_filename'):
                        print(f"   파일: {result['metadata']['original_filename']}")
            else:
                print("📭 검색 결과가 없습니다. 먼저 문서를 업로드해주세요.")
        else:
            print(f"❌ 검색 테스트 실패: {response.status_code} - {response.text}")
            
    except requests.exceptions.ConnectionError:
        print("❌ 연결 오류: RAG Agent 서비스가 실행 중인지 확인하세요.")
    except Exception as e:
        print(f"❌ 예상치 못한 오류: {e}")
