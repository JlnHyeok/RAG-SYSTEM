import time
import json
import websocket
import requests
import threading
from typing import Optional, Dict, Any

def await_processing_completion(document_id: str, timeout: int = 600) -> bool:
    """WebSocket을 통해 문서 처리 진행 상황을 실시간으로 모니터링합니다."""
    print(f"📡 문서 처리 모니터링 시작...")
    
    websocket_working = False
    processing_completed = False
    progress_data = {}
    api_url = "http://localhost:8000"
    last_message_time = time.time()
    ws = None
    last_printed_ws_progress = {}  # WebSocket 메시지 중복 출력 방지

    def on_message(ws, message):
        nonlocal websocket_working, progress_data, processing_completed, last_message_time, last_printed_ws_progress
        try:
            last_message_time = time.time()
            # print(f"\n📨 WebSocket 메시지 수신: {message}", flush=True)  # 디버깅용
            data = json.loads(message)
            
            # 진행률 데이터 업데이트
            if 'step' in data:
                progress_data['step'] = data['step']
            if 'overall_progress' in data:
                progress_data['overall_progress'] = data['overall_progress']
            if 'step_progress' in data:
                progress_data['step_progress'] = data['step_progress']
            if 'status' in data:
                progress_data['status'] = data['status']
            
            # 진행률 표시 (WebSocket 메시지)
            step = data.get('step', 'N/A')
            overall_progress = data.get('overall_progress', 0)
            step_progress = data.get('step_progress', 0)
            
            # 진행률이 변경되었을 때만 출력 (중복 방지)
            current_ws_progress = {
                'step': step,
                'step_progress': step_progress,
                'overall_progress': overall_progress
            }
            
            if current_ws_progress != last_printed_ws_progress:
                print(f"\r🔄 {step}: {step_progress:.1f}% | 전체: {overall_progress:.1f}%", end="", flush=True)
                last_printed_ws_progress = current_ws_progress.copy()
            
            if data.get('status') == 'completed' or data.get('type') == 'completion':
                if not processing_completed:  # 중복 방지
                    print(f"\n✅ 문서 처리 완료!")
                    
                    # 완료 데이터는 result 객체 안에 있을 수 있음
                    result_data = data.get('result', {})
                    
                    text_chunks = result_data.get('text_chunks', data.get('text_chunks', 0))
                    image_chunks = result_data.get('image_chunks', data.get('image_chunks', 0))
                    total_embeddings = result_data.get('total_embeddings', data.get('total_embeddings', 0))
                    processing_time = result_data.get('processing_time', data.get('processing_time', 0))
                    
                    print(f"📝 텍스트 청크: {text_chunks}개")
                    print(f"🖼️ 이미지 청크: {image_chunks}개")
                    print(f"🧠 임베딩: {total_embeddings}개")
                    print(f"⏱️ 처리 시간: {processing_time:.2f}초")
                    
                processing_completed = True
                websocket_working = False
            elif data.get('status') == 'failed':
                print(f"\n❌ 문서 처리 실패: {data.get('error', '알 수 없는 오류')}")
                processing_completed = True
                websocket_working = False
                
        except json.JSONDecodeError as e:
            print(f"\n❌ WebSocket 메시지 파싱 실패: {e}")
        except Exception as e:
            print(f"\n❌ WebSocket 메시지 처리 오류: {e}")

    def on_error(ws, error):
        nonlocal websocket_working
        # ping 관련 에러는 무시 (더 포괄적으로)
        error_str = str(error).lower()
        if any(term in error_str for term in ["ping", "pong", "heartbeat", "websocketapp"]):
            return
        print(f"⚠️ WebSocket 오류: {error}", flush=True)
        websocket_working = False

    def on_close(ws, close_status_code, close_msg):
        nonlocal websocket_working
        # print(f"📡 WebSocket 연결 종료", flush=True)  # 사용자 요청으로 비활성화 (프롬프트 간섭 방지)
        websocket_working = False

    def on_open(ws):
        nonlocal websocket_working
        print("✅ WebSocket 연결 성공!")
        websocket_working = True
        print(f"📡 WebSocket URL: {websocket_url}")  # 연결된 URL 확인

    # WebSocket 연결 시도
    websocket_url = f"ws://localhost:8000/api/v1/documents/ws/progress/{document_id}"
    
    try:
        ws = websocket.WebSocketApp(websocket_url,
                                  on_message=on_message,
                                  on_error=on_error,
                                  on_close=on_close,
                                  on_open=on_open)
        
        # 백그라운드에서 WebSocket 실행
        ws_thread = threading.Thread(target=lambda: ws.run_forever(), daemon=True)
        ws_thread.start()
        
        # 연결이 될 때까지 대기
        connection_timeout = 10
        connection_start = time.time()
        while not websocket_working and (time.time() - connection_start) < connection_timeout:
            time.sleep(0.1)
        
    except Exception as e:
        print(f"WebSocket 연결 실패: {e}")
        websocket_working = False
    
    # 처리 완료까지 대기
    start_time = time.time()
    check_interval = 5  # API 체크 간격 (5초)
    last_api_check = time.time()
    last_printed_progress = {}  # 중복 출력 방지용
    
    while not processing_completed and (time.time() - start_time) < timeout:
        current_time = time.time()
        
        # WebSocket 메시지가 5초 이상 없을 때만 API로 체크
        should_check_api = (
            (current_time - last_message_time) > 5 and  # 5초 동안 WebSocket 메시지 없음
            (current_time - last_api_check) >= check_interval  # 정기 체크 간격
        )
        
        if should_check_api:
            print(f"\n🔍 API로 상태 확인 중... (WebSocket 메시지 간격: {current_time - last_message_time:.1f}초)", flush=True)
            try:
                response = requests.get(f"{api_url}/api/v1/documents/upload/{document_id}/status", timeout=5)
                if response.status_code == 200:
                    api_data = response.json()
                    api_status = api_data.get('status', 'unknown')
                    api_progress = api_data.get('overall_progress', 0)
                    current_step = api_data.get('current_step', 'N/A')
                    step_progress = api_data.get('step_progress', 0)
                    
                    print(f"📊 API 응답: {api_status}, {current_step}, 단계:{step_progress:.1f}%, 전체:{api_progress:.1f}%", flush=True)
                    
                    # 진행률이 변경되었을 때만 출력 (중복 방지)
                    current_progress = {
                        'step': current_step,
                        'step_progress': step_progress,
                        'overall_progress': api_progress
                    }
                    
                    if current_progress != last_printed_progress:
                        print(f"\r🔄 {current_step}: {step_progress:.1f}% | 전체: {api_progress:.1f}%", end="", flush=True)
                        last_printed_progress = current_progress.copy()
                    
                    if api_status == 'completed':
                        if not processing_completed:
                            print(f"\n✅ 문서 처리 완료!")
                            # API에서 결과 데이터 가져오기
                            result_data = api_data.get('result_data', {})
                            text_chunks = result_data.get('text_chunks', 0)
                            image_chunks = result_data.get('image_chunks', 0)
                            total_embeddings = result_data.get('total_embeddings', 0)
                            processing_time = api_data.get('elapsed_time', 0)
                            
                            print(f"📝 텍스트 청크: {text_chunks}개")
                            print(f"🖼️ 이미지 청크: {image_chunks}개")
                            print(f"🧠 임베딩: {total_embeddings}개")
                            print(f"⏱️ 처리 시간: {processing_time:.2f}초")
                        
                        processing_completed = True
                        break
                    elif api_status == 'failed':
                        print(f"\n❌ 문서 처리 실패!")
                        processing_completed = True
                        break
                        
            except requests.exceptions.RequestException as e:
                pass  # API 요청 실패는 조용히 처리
            
            last_api_check = current_time
        
        time.sleep(1.0)  # 체크 간격
    
    # WebSocket 연결 정리
    try:
        if ws:
            ws.close()
    except:
        pass
    
    if not processing_completed:
        print(f"\n⏰ 처리 타임아웃 ({timeout}초)")
        return False
    
    return True
