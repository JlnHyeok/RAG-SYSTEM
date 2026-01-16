"""
대화형 모드 관련 기능
"""
import os
from commands.ask import ask_question
from commands.upload import upload_file
from commands.documents import list_files, delete_document, clear_all_documents
from commands.status import check_vector_status, test_vector_search
from utils.file_utils import show_current_files

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
                        result = upload_file(file_path)
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
            
            ask_question(question, stream=stream_mode)
            
    except KeyboardInterrupt:
        print("\n👋 안녕히 가세요!")
