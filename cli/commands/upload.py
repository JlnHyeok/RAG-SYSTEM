"""
파일 업로드 관련 명령어
"""
import os
import time
import threading
import requests
from pathlib import Path
from config import BASE_URL, UPLOAD_TIMEOUT
from utils.progress import (
    ProgressFileWrapper, 
    create_upload_progress_tracker,
    format_file_size
)
from utils.websocket_monitor import await_processing_completion

def _upload_file_to_server(file_path, user_id, progress_callback):
    """서버에 파일을 업로드하는 내부 함수"""
    with open(file_path, 'rb') as file:
        progress_file = ProgressFileWrapper(file, progress_callback)
        files = {'file': (os.path.basename(file_path), progress_file, 'application/octet-stream')}
        data = {'user_id': user_id}
        
        upload_url = f"{BASE_URL}/api/v1/documents/upload"
        
        response = requests.post(
            upload_url,
            files=files,
            data=data,
            timeout=UPLOAD_TIMEOUT
        )
        
        return response

def upload_file(file_path: str, user_id: str = "cli_user") -> bool:
    """파일을 RAG Agent에 업로드합니다"""
    
    if not os.path.exists(file_path):
        print(f"❌ 파일을 찾을 수 없습니다: {file_path}")
        return False
    
    try:
        # 파일 정보 수집
        file_size = Path(file_path).stat().st_size
        file_name = Path(file_path).name
        
        print(f"📤 파일 업로드 시작: {file_name} ({format_file_size(file_size)})")
        print("=" * 50)
        
        # 진행률 추적 설정
        upload_complete = threading.Event()
        bytes_sent = [0]
        
        def update_progress(bytes_read):
            bytes_sent[0] = bytes_read
        
        # 진행률 표시 스레드 시작
        progress_tracker = create_upload_progress_tracker(file_size, bytes_sent, upload_complete)
        progress_thread = threading.Thread(target=progress_tracker, daemon=True)
        progress_thread.start()
        
        # 파일 업로드 실행
        response = _upload_file_to_server(file_path, user_id, update_progress)
        upload_complete.set()
        time.sleep(0.1)  # 진행률 업데이트 완료 대기
        
        # 업로드 결과 확인
        if response.status_code != 200:
            print(f"\n❌ 업로드 실패: {response.status_code}")
            try:
                error_data = response.json()
                print(f"오류: {error_data.get('detail', response.text)}")
            except:
                print(f"오류: {response.text}")
            return False
        
        # 업로드 성공 - 문서 ID 추출
        try:
            data = response.json()
            document_id = data.get('document_id')
            status = data.get('status', 'unknown')
            message = data.get('message', '')
            
            if not document_id:
                print("❌ 문서 ID를 받지 못했습니다.")
                return False
            
            print(f"\n✅ 파일 업로드 완료!")
            print(f"📋 문서 ID: {document_id[:12]}...")
            print(f"📝 메시지: {message}")
            
            # 상태에 따른 처리
            if status == "processing":
                print("\n🔗 문서 처리 상태 모니터링 시작...")
                # 웹소켓으로 처리 완료 대기
                success = await_processing_completion(document_id)
                return success
                
            elif status == "completed":
                # 이미 처리 완료된 경우 (작은 파일들)
                text_chunks = data.get('text_chunks', 0)
                image_chunks = data.get('image_chunks', 0)
                total_embeddings = data.get('total_embeddings', 0)
                processing_time = data.get('processing_time', 0)
                
                print(f"\n✅ 문서 처리 완료!")
                print(f"📝 텍스트 청크: {text_chunks}개")
                print(f"🖼️ 이미지 청크: {image_chunks}개")
                print(f"🧠 임베딩: {total_embeddings}개")
                print(f"⏱️ 처리 시간: {processing_time:.2f}초")
                return True
                
            else:
                print(f"⚠️ 알 수 없는 상태: {status}")
                # 웹소켓으로 확인
                success = await_processing_completion(document_id)
                return success
            
        except Exception as e:
            print(f"❌ 응답 처리 오류: {e}")
            return False
        
    except requests.exceptions.Timeout:
        print(f"\n❌ 타임아웃: 파일 처리 시간이 너무 오래 걸립니다.")
        return False
    except requests.exceptions.ConnectionError:
        print("❌ 연결 오류: RAG Agent 서비스가 실행 중인지 확인하세요.")
        return False
    except Exception as e:
        print(f"❌ 예상치 못한 오류: {e}")
        return False
