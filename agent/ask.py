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
    """파일을 RAG Agent에 업로드합니다 - 진행 상태 표시"""
    
    if not os.path.exists(file_path):
        print(f"❌ 파일을 찾을 수 없습니다: {file_path}")
        return
    
    try:
        import time
        import threading
        from pathlib import Path
        
        # 파일 크기 확인
        file_size = Path(file_path).stat().st_size
        file_size_mb = file_size / (1024 * 1024)
        
        print(f"📤 파일 업로드 중: {Path(file_path).name} ({file_size_mb:.1f}MB)")
        print("=" * 50)
        
        # 진행 상태 표시를 위한 플래그
        upload_complete = threading.Event()
        
        def show_upload_progress():
            """업로드 진행 상태 애니메이션"""
            animation = ["|", "/", "-", "\\"]
            i = 0
            start_time = time.time()
            
            while not upload_complete.is_set():
                elapsed = time.time() - start_time
                print(f"\r🔄 파일 전송 중... {animation[i % len(animation)]} ({elapsed:.1f}s)", end="", flush=True)
                time.sleep(0.2)
                i += 1
        
        # 진행 상태 스레드 시작
        progress_thread = threading.Thread(target=show_upload_progress, daemon=True)
        progress_thread.start()
        
        # 실제 업로드
        with open(file_path, 'rb') as file:
            files = {'file': (os.path.basename(file_path), file, 'application/octet-stream')}
            data = {'user_id': user_id}
            
            upload_start = time.time()
            response = requests.post(
                f"{BASE_URL}/api/v1/upload",
                files=files,
                data=data,
                timeout=300  # 5분 타임아웃
            )
            upload_time = time.time() - upload_start
        
        # 업로드 완료 신호
        upload_complete.set()
        time.sleep(0.3)  # 애니메이션 정리 시간
        print(f"\r✅ 파일 전송 완료! ({upload_time:.1f}s)                    ")
        
        if response.status_code == 200:
            data = response.json()
            print("\n🤖 Agent 처리 중...")
            print("=" * 50)
            
            # 상태별 메시지
            if data['status'] == 'completed':
                print(f"✅ 문서 처리 완료!")
                print(f"📄 문서 ID: {data['document_id'][:12]}...")
                print(f"📝 텍스트 청크: {data['text_chunks']}개")
                if data.get('image_chunks', 0) > 0:
                    print(f"🖼️ 이미지 청크: {data['image_chunks']}개 (OCR 처리됨)")
                print(f"💾 총 벡터 임베딩: {data['total_embeddings']}개")
                print(f"⏱️ 총 처리 시간: {data['processing_time']:.2f}초")
                print("\n🎉 이제 이 문서에 대해 질문할 수 있습니다!")
            elif data['status'] == 'failed':
                print(f"❌ 문서 처리 실패")
                print(f"📄 문서 ID: {data['document_id'][:12]}...")
                print(f"⏱️ 처리 시간: {data['processing_time']:.2f}초")
                print("💡 파일이 업로드되었지만 벡터 처리에 실패했습니다.")
                print("   다른 파일을 시도하거나 관리자에게 문의하세요.")
        else:
            print(f"\n❌ 업로드 실패: {response.status_code}")
            try:
                error_data = response.json()
                print(f"   오류: {error_data.get('detail', response.text)}")
            except:
                print(f"   오류: {response.text}")
            
    except requests.exceptions.Timeout:
        print(f"\n❌ 업로드 타임아웃: 파일이 너무 크거나 서버 응답이 느립니다.")
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