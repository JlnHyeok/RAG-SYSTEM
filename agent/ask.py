#!/usr/bin/env python3
"""
RAG Agent CLI - 터미널에서 바로 질문하기
사용법: python ask.py "질문내용"
"""
import sys
import requests
import json
import argparse
import os
from typing import Optional

# 기본 설정
BASE_URL = "http://localhost:8000"
API_ENDPOINT = f"{BASE_URL}/api/v1/query"
STREAM_ENDPOINT = f"{BASE_URL}/api/v1/query/stream"


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
    
    # 질문 표시 (모든 모드에서 동일하게)
    # print(f"🤖 질문: {question}")
    
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
                print(f"💬 답변:\n{data['answer']}")
                
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

def upload_file(file_path: str, user_id: str = "cli_user") -> None:
    """파일을 RAG Agent에 업로드합니다 - 실시간 진행률 표시"""
    
    if not os.path.exists(file_path):
        print(f"❌ 파일을 찾을 수 없습니다: {file_path}")
        return
    
    try:
        import time
        import threading
        import websocket
        import json
        from pathlib import Path
        
        # 파일 크기 확인
        file_size = Path(file_path).stat().st_size
        file_size_mb = file_size / (1024 * 1024)
        
        print(f"📤 파일 업로드 시작: {Path(file_path).name} ({file_size_mb:.1f}MB)")
        print("=" * 50)
        
        # WebSocket 연결 상태 및 메시지 저장
        ws_connected = threading.Event()
        ws_messages = []
        document_id = None
        
        def on_websocket_message(ws, message):
            """WebSocket 메시지 수신"""
            try:
                data = json.loads(message)
                ws_messages.append(data)
                
                if data['type'] == 'progress':
                    step = data['step']
                    progress = data['progress']
                    message = data.get('message', '')
                    
                    # 진행률 바 생성
                    bar_width = 20
                    filled = int(bar_width * progress / 100)
                    bar = "█" * filled + "░" * (bar_width - filled)
                    
                    print(f"\r🔄 {step} [{bar}] {progress:.1f}% - {message}     ", 
                          end="", flush=True)
                    
                elif data['type'] == 'completion':
                    print(f"\n✅ 처리 완료: {data['status']}")
                    
            except Exception as e:
                pass  # JSON 파싱 에러 무시
        
        def on_websocket_error(ws, error):
            """WebSocket 에러"""
            pass  # 에러 무시 (서버 처리가 완료되면 자동으로 끊어짐)
        
        def on_websocket_close(ws, close_status_code, close_msg):
            """WebSocket 연결 종료"""
            pass
        
        def on_websocket_open(ws):
            """WebSocket 연결 성공"""
            ws_connected.set()
        
        # 1단계: 파일 업로드 (기존 방식)
        upload_complete = threading.Event()
        bytes_sent = [0]
        
        def show_upload_progress():
            """업로드 진행 상태 표시"""
            animation = ["|", "/", "-", "\\"]
            i = 0
            start_time = time.time()
            
            while not upload_complete.is_set():
                current_time = time.time()
                elapsed = current_time - start_time
                current_bytes = bytes_sent[0]
                
                progress = min((current_bytes / file_size) * 100, 100) if file_size > 0 else 0
                
                # 속도 계산
                if elapsed > 0:
                    speed = current_bytes / elapsed
                    if speed >= 1024 * 1024:
                        speed_str = f"{speed / (1024 * 1024):.1f}MB/s"
                    elif speed >= 1024:
                        speed_str = f"{speed / 1024:.1f}KB/s"
                    else:
                        speed_str = f"{speed:.0f}B/s"
                else:
                    speed_str = "0B/s"
                
                # 프로그레스 바
                bar_width = 30
                filled = int(bar_width * progress / 100)
                bar = "█" * filled + "░" * (bar_width - filled)
                
                size_mb = current_bytes / (1024 * 1024)
                print(f"\r📤 업로드: [{bar}] {progress:.1f}% ({size_mb:.1f}MB) {speed_str}     ", 
                      end="", flush=True)
                
                time.sleep(0.2)
                i += 1
        
        class ProgressFileWrapper:
            """파일 읽기를 모니터링하는 래퍼 클래스"""
            def __init__(self, file, callback, chunk_size=8192):
                self.file = file
                self.callback = callback
                self.bytes_read = 0
                self.chunk_size = chunk_size
                self.last_update = time.time()
            
            def read(self, size=-1):
                """파일 읽기 - requests 호환 (청크 단위)"""
                if size is None or size <= 0:
                    # 전체 파일을 청크 단위로 읽기
                    data = b''
                    while True:
                        chunk = self.file.read(self.chunk_size)
                        if not chunk:
                            break
                        data += chunk
                        self.bytes_read += len(chunk)
                        
                        # 100ms마다 진행률 업데이트
                        now = time.time()
                        if now - self.last_update >= 0.1:
                            self.callback(self.bytes_read)
                            self.last_update = now
                    
                    self.callback(self.bytes_read)  # 최종 업데이트
                    return data
                else:
                    # 지정된 크기만큼 읽기
                    data = self.file.read(size)
                    self.bytes_read += len(data)
                    
                    # 진행률 업데이트
                    now = time.time()
                    if now - self.last_update >= 0.1 or len(data) == 0:
                        self.callback(self.bytes_read)
                        self.last_update = now
                    
                    return data
            
            def readline(self, size=-1):
                """라인 읽기 - requests 호환"""
                if size is None or size < 0:
                    data = self.file.readline()
                else:
                    data = self.file.readline(size)
                
                self.bytes_read += len(data)
                self.callback(self.bytes_read)
                return data
            
            def readlines(self, hint=-1):
                """모든 라인 읽기 - requests 호환"""
                lines = self.file.readlines(hint)
                for line in lines:
                    self.bytes_read += len(line)
                self.callback(self.bytes_read)
                return lines
            
            def seek(self, offset, whence=0):
                """파일 위치 변경"""
                result = self.file.seek(offset, whence)
                # seek 후 현재 위치로 bytes_read 조정
                self.bytes_read = self.file.tell()
                self.callback(self.bytes_read)
                return result
            
            def tell(self):
                """현재 파일 위치 반환"""
                return self.file.tell()
            
            def __getattr__(self, name):
                """나머지 속성들은 원본 파일 객체에 위임"""
                return getattr(self.file, name)
        
        def update_progress(bytes_read):
            """진행률 업데이트 콜백"""
            bytes_sent[0] = bytes_read
        
        # 진행 상태 스레드 시작
        progress_thread = threading.Thread(target=show_upload_progress, daemon=True)
        progress_thread.start()
        
        # 실제 업로드 (진행률 모니터링과 함께)
        with open(file_path, 'rb') as file:
            # 파일을 래퍼로 감싸서 진행률 추적
            progress_file = ProgressFileWrapper(file, update_progress)
            files = {'file': (os.path.basename(file_path), progress_file, 'application/octet-stream')}
            data = {'user_id': user_id}
            
            upload_start = time.time()
            response = requests.post(
                f"{BASE_URL}/api/v1/upload",
                files=files,
                data=data,
                timeout=1800  # 30분 타임아웃 (대용량 파일 + OCR 처리)
            )
            upload_time = time.time() - upload_start
        
        # 업로드 완료 신호
        upload_complete.set()
        time.sleep(0.2)  # 마지막 진행률 업데이트 대기
        
        # 최종 완료 메시지
        final_speed = file_size / upload_time if upload_time > 0 else 0
        if final_speed >= 1024 * 1024:  # MB/s
            speed_str = f"{final_speed / (1024 * 1024):.1f}MB/s"
        elif final_speed >= 1024:  # KB/s
            speed_str = f"{final_speed / 1024:.1f}KB/s"
        else:  # B/s
            speed_str = f"{final_speed:.0f}B/s"
        
        bar_filled = "█" * 20  # 100% 진행률 바
        print(f"\r✅ 업로드 완료! [{bar_filled}] 100.0% ({file_size_mb:.1f}MB) 평균 {speed_str} ({upload_time:.1f}s)" + " " * 10)
        
        if response.status_code == 200:
            data = response.json()
            print("\n🤖 문서 처리 중...")
            print("-" * 40)
            
            processing_steps = [
                "📖 PDF 파싱...",
                "✂️ 텍스트 추출 및 청킹...", 
                "🖼️ 이미지 추출...",
                "👁️ OCR 처리...",
                "🧠 임베딩 생성...",
                "💾 벡터 저장..."
            ]
            
            # 간단한 처리 상황 시뮬레이션
            for i, step in enumerate(processing_steps):
                print(f"{step}")
                if i < 3:
                    time.sleep(1)  # 빠른 단계들
                else:
                    time.sleep(0.5)  # 이미 완료된 상태이므로 빠르게
            
            print("\n" + "=" * 40)
            
            # 결과 표시
            if data['status'] == 'completed':
                print(f"✅ 문서 처리 완료!")
                print(f"📄 문서 ID: {data['document_id'][:12]}...")
                print(f"📝 텍스트 청크: {data['text_chunks']}개")
                if data.get('image_chunks', 0) > 0:
                    print(f"🖼️ 이미지 청크: {data['image_chunks']}개 (OCR 처리)")
                print(f"💾 총 임베딩: {data['total_embeddings']}개")
                print(f"⏱️ 처리 시간: {data['processing_time']:.1f}초")
                print("\n🎉 이제 이 문서에 대해 질문할 수 있습니다!")
            else:
                print(f"❌ 처리 실패: {data.get('status', 'unknown')}")
        else:
            print(f"\n❌ 업로드 실패: {response.status_code}")
            try:
                error_data = response.json()
                print(f"오류: {error_data.get('detail', response.text)}")
            except:
                print(f"오류: {response.text}")
                
    except requests.exceptions.Timeout:
        print(f"\n❌ 타임아웃: 파일 처리 시간이 너무 오래 걸립니다.")
        print("   대용량 파일이나 이미지가 많은 PDF의 경우 시간이 오래 걸릴 수 있습니다.")
    except requests.exceptions.ConnectionError:
        print("❌ 연결 오류: RAG Agent 서비스가 실행 중인지 확인하세요.")
    except Exception as e:
        print(f"❌ 오류: {e}")


def delete_document(document_id: str, user_id: str = "cli_user") -> None:
    """문서를 벡터 DB에서 삭제합니다 (문서 ID 또는 파일명으로)"""
    
    try:
        print(f"🗑️  문서 삭제 중: {document_id}")
        print("=" * 40)
        
        # 먼저 파일명으로 삭제 시도 (확장자가 있거나 일반적인 파일명인 경우)
        if ('.' in document_id and document_id.count('.') == 1 and 
            any(document_id.lower().endswith(ext) for ext in ['.txt', '.pdf', '.md', '.doc', '.docx'])):
            
            print("📄 파일명으로 삭제를 시도합니다...")
            response = requests.delete(
                f"{BASE_URL}/api/v1/delete-by-name/{document_id}",
                params={"user_id": user_id},
                timeout=30
            )
        else:
            print("🆔 문서 ID로 삭제를 시도합니다...")
            response = requests.delete(
                f"{BASE_URL}/api/v1/delete/{document_id}",
                params={"user_id": user_id},
                timeout=30
            )
        
        if response.status_code == 200:
            data = response.json()
            print("✅ 문서 삭제 완료!")
            print(f"📋 결과: {data.get('message', '삭제되었습니다.')}")
            
            # 삭제 성공/실패 여부 표시
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
        print("   서비스 시작: python -m uvicorn app.main:app --reload")
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
            f"{BASE_URL}/api/v1/clear-all",
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
        print("   서비스 시작: python -m uvicorn app.main:app --reload")
    except Exception as e:
        print(f"❌ 예상치 못한 오류: {e}")


def list_files(user_id: str = "cli_user") -> None:
    """업로드된 파일 목록을 조회합니다"""
    
    try:
        print("📂 업로드된 파일 목록 조회 중...")
        
        response = requests.get(
            f"{BASE_URL}/api/v1/list",
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
                
                # 새로운 응답 구조에 맞게 수정
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


def show_help():
    """사용법 도움말을 출력합니다"""
    print("💡 RAG Agent CLI 사용법:")
    print("  - 질문을 입력하면 답변을 받을 수 있습니다")
    print("  - '/upload <파일경로>' 로 파일을 업로드할 수 있습니다")
    print("  - '/delete <문서ID|파일명>' 로 업로드된 문서를 삭제할 수 있습니다")
    print("    예: /delete test_document.txt 또는 /delete abc123")
    print("  - '/clear' 로 모든 문서를 삭제할 수 있습니다 (주의!)")
    print("  - '/list' 로 업로드된 파일 목록을 확인할 수 있습니다")
    print("  - '/status' 로 벡터 DB 상태를 확인할 수 있습니다")
    print("  - '/search <검색어>' 로 벡터 검색을 테스트할 수 있습니다")
    print("  - '/stream on/off' 로 스트리밍 모드를 전환할 수 있습니다")
    print("  - '/사용법', '/help' 로 이 도움말을 다시 볼 수 있습니다")
    print("  - '/exit', '/quit', 'quit', 'exit' 로 종료합니다")
    print("="* 50)


def show_current_files():
    """현재 디렉토리의 파일 목록을 표시"""
    import glob
    
    # 일반적인 문서 파일 확장자들
    patterns = ['*.txt', '*.pdf', '*.md', '*.doc', '*.docx']
    
    print("📁 현재 디렉토리의 파일들:")
    found_files = []
    
    for pattern in patterns:
        files = glob.glob(pattern)
        found_files.extend(files)
    
    if found_files:
        for i, file in enumerate(sorted(found_files), 1):
            file_size = os.path.getsize(file) / 1024  # KB 단위
            if file_size < 1024:
                size_str = f"{file_size:.1f}KB"
            else:
                size_str = f"{file_size/1024:.1f}MB"
            print(f"  {i}. {file} ({size_str})")
    else:
        print("  📭 지원되는 문서 파일이 없습니다.")
        print("     지원 형식: .txt, .pdf, .md, .doc, .docx")
    
    print("💡 사용법: /upload <파일명> (예: /upload document.pdf)")
    print()


def interactive_mode():
    """대화형 모드"""
    print("🤖 RAG Agent 대화형 모드")
    show_help()
    
    stream_mode = False
    
    try:
        while True:
            question = input("\n💭 질문: ").strip()
            
            if question.lower() in ['quit', 'exit', 'q', '/quit', '/exit']:
                print("👋 안녕히 가세요!")
                break
            
            if question == '/stream on':
                stream_mode = True
                print("✅ 스트리밍 모드 활성화")
                continue
            elif question == '/stream off':
                stream_mode = False
                print("✅ 스트리밍 모드 비활성화")
                continue
            elif question == '/list':
                list_files()
                continue
            elif question == '/status':
                check_vector_status()
                continue
            elif question.startswith('/search '):
                search_query = question[8:].strip()
                if search_query:
                    test_vector_search(search_query)
                else:
                    test_vector_search()
                continue
            elif question.startswith('/upload '):
                file_path = question[8:].strip()
                if file_path:
                    print(f"📋 업로드 요청: {file_path}")
                    if not os.path.exists(file_path):
                        print(f"❌ 파일을 찾을 수 없습니다: {file_path}")
                        print("💡 현재 디렉토리의 파일들을 확인하세요:")
                        show_current_files()
                    else:
                        upload_file(file_path)
                else:
                    print("❌ 파일 경로를 입력해주세요.")
                    print()
                    show_current_files()
                continue
            elif question.startswith('/delete '):
                document_id = question[8:].strip()
                if document_id:
                    delete_document(document_id)
                else:
                    print("❌ 문서 ID를 입력해주세요.")
                    print("💡 사용법: /delete <문서ID|파일명>")
                continue
            elif question == '/clear':
                clear_all_documents()
                continue
            elif question == '/upload':
                print("❌ 파일 경로를 입력해주세요.")
                print()
                show_current_files()
                continue
            elif question in ['/사용법', '/help', '/도움말']:
                show_help()
                continue
            
            if not question:
                continue
            
            ask_question(question, stream=stream_mode)  # 질문 표시는 ask_question 내부에서
            
    except KeyboardInterrupt:
        print("\n👋 안녕히 가세요!")

def main():
    parser = argparse.ArgumentParser(
        description="RAG Agent CLI - 터미널에서 바로 질문하기"
    )
    parser.add_argument(
        "question", 
        nargs="?", 
        help="질문 내용 (없으면 대화형 모드)"
    )
    parser.add_argument(
        "-s", "--stream", 
        action="store_true", 
        help="스트리밍 모드 사용"
    )
    parser.add_argument(
        "-u", "--user-id", 
        default="cli_user", 
        help="사용자 ID (기본값: cli_user)"
    )
    parser.add_argument(
        "--url", 
        default="http://localhost:8000", 
        help="RAG Agent 서버 URL (기본값: http://localhost:8000)"
    )
    
    args = parser.parse_args()
    
    # 전역 변수 업데이트
    global BASE_URL, API_ENDPOINT, STREAM_ENDPOINT
    BASE_URL = args.url
    API_ENDPOINT = f"{BASE_URL}/api/v1/query"
    STREAM_ENDPOINT = f"{BASE_URL}/api/v1/query/stream"
    
    if args.question:
        # 단일 질문 모드
        ask_question(args.question, stream=args.stream, user_id=args.user_id)
    else:
        # 대화형 모드
        interactive_mode()

if __name__ == "__main__":
    main()