"""
문서 관리 관련 명령어
"""
import requests
from config import BASE_URL, DEFAULT_TIMEOUT, SUPPORTED_EXTENSIONS

def delete_document(document_id: str, user_id: str = "cli_user") -> None:
    """문서를 벡터 DB에서 삭제합니다 (문서 ID 또는 파일명으로)"""
    
    try:
        print(f"🗑️  문서 삭제 중: {document_id}")
        print("=" * 40)
        
        # 파일명으로 삭제 시도 (확장자가 있는 경우)
        if ('.' in document_id and document_id.count('.') == 1 and 
            any(document_id.lower().endswith(ext) for ext in SUPPORTED_EXTENSIONS)):
            
            print("📄 파일명으로 삭제를 시도합니다...")
            response = requests.delete(
                f"{BASE_URL}/api/v1/documents/delete-by-name/{document_id}",
                params={"user_id": user_id},
                timeout=DEFAULT_TIMEOUT
            )
        else:
            print("🆔 문서 ID로 삭제를 시도합니다...")
            response = requests.delete(
                f"{BASE_URL}/api/v1/documents/delete/{document_id}",
                params={"user_id": user_id},
                timeout=DEFAULT_TIMEOUT
            )
        
        if response.status_code == 200:
            data = response.json()
            print("✅ 문서 삭제 완료!")
            print(f"📋 결과: {data.get('message', '삭제되었습니다.')}")
            
            deleted_chunks = data.get('deleted_chunks', 0)
            success = data.get('success', True)
            
            if success and deleted_chunks > 0:
                print(f"🎉 {deleted_chunks}개의 텍스트 청크가 성공적으로 삭제되었습니다!")
            elif deleted_chunks == 0:
                print("\n💡 해당 이름/ID로 문서를 찾을 수 없습니다.")
                print("   다음을 확인해보세요:")
                print("   1. '/list' 명령어로 업로드된 파일 목록 확인")
                print("   2. 정확한 파일명인지 확인 (대소문자, 확장자 포함)")
                print("   3. 파일이 실제로 업로드되었는지 확인")
        else:
            print("❌ 문서 삭제 실패!")
            try:
                error_data = response.json()
                print(f"   오류: {error_data.get('detail', response.text)}")
            except:
                print(f"   오류: {response.text}")
                
    except requests.exceptions.ConnectionError:
        print("❌ 연결 오류: RAG Agent 서비스가 실행 중인지 확인하세요.")
    except Exception as e:
        print(f"❌ 예상치 못한 오류: {e}")

def clear_all_documents(user_id: str = "cli_user") -> None:
    """모든 문서를 벡터 DB에서 삭제합니다 (주의: 되돌릴 수 없음)"""
    
    try:
        print("⚠️  경고: 모든 문서를 삭제하려고 합니다!")
        print("🗑️  이 작업은 되돌릴 수 없습니다.")
        print("="*50)
        
        # 사용자 확인
        confirm = input("정말로 모든 문서를 삭제하시겠습니까? (yes/no): ").strip().lower()
        
        if confirm not in ['yes', 'y', '예']:
            print("❌ 삭제가 취소되었습니다.")
            return
        
        print("🗑️  모든 문서 삭제 중...")
        
        response = requests.delete(
            f"{BASE_URL}/api/v1/documents/clear-all",
            params={"user_id": user_id},
            timeout=60  # 전체 삭제는 시간이 걸릴 수 있음
        )
        
        if response.status_code == 200:
            data = response.json()
            print("✅ 전체 문서 삭제 완료!")
            print(f"📋 결과: {data.get('message', '모든 문서가 삭제되었습니다.')}") 
            
            success = data.get('success', True)
            if success:
                print("🎉 벡터 데이터베이스가 완전히 초기화되었습니다!")
                print("💡 이제 새로운 문서를 업로드할 수 있습니다.")
            else:
                print("ℹ️  삭제할 문서가 없었거나 이미 비어있습니다.")
        else:
            print("❌ 전체 문서 삭제 실패!")
            try:
                error_data = response.json()
                print(f"   오류: {error_data.get('detail', response.text)}")
            except:
                print(f"   오류: {response.text}")
                
    except requests.exceptions.ConnectionError:
        print("❌ 연결 오류: RAG Agent 서비스가 실행 중인지 확인하세요.")
    except Exception as e:
        print(f"❌ 예상치 못한 오류: {e}")

def list_files(user_id: str = "cli_user") -> None:
    """업로드된 파일 목록을 조회합니다"""
    
    try:
        print("📂 업로드된 파일 목록 조회 중...")
        
        response = requests.get(
            f"{BASE_URL}/api/v1/documents/list",
            params={"user_id": user_id}
        )
        
        if response.status_code == 200:
            data = response.json()
            files = data['files']
            
            if not files:
                print("📭 업로드된 파일이 없습니다.")
                return
                
            print("=" * 50)
            print(f"📁 총 {data['total_count']}개 파일")
            if 'total_chunks' in data:
                print(f"📊 총 청크 수: {data['total_chunks']}개")
            print()
            
            for i, file_info in enumerate(files, 1):
                print(f"{i}. {file_info['original_name']}")
                
                if 'chunks' in file_info:
                    print(f"   📦 청크 수: {file_info['chunks']}개")
                if 'content_length' in file_info:
                    print(f"   📝 콘텐츠 길이: {file_info['content_length']:,} 자")
                if 'document_id' in file_info:
                    print(f"   🆔 문서 ID: {file_info['document_id'][:12]}...")
                
                print(f"   📄 형식: {file_info['file_type']}")
                if file_info.get('uploaded_at'):
                    print(f"   📅 업로드: {file_info['uploaded_at'][:10]}")
                print()
        else:
            print(f"❌ 목록 조회 실패: {response.status_code} - {response.text}")
            
    except requests.exceptions.ConnectionError:
        print("❌ 연결 오류: RAG Agent 서비스가 실행 중인지 확인하세요.")
    except Exception as e:
        print(f"❌ 예상치 못한 오류: {e}")
