#!/usr/bin/env python3
"""
RAG Agent CLI - 메인 진입점
사용법: python main.py "질문내용"
"""
import sys
import argparse
from pathlib import Path

# readline 라이브러리 활성화 (한국어 입력 처리 개선)
try:
    import readline
    # macOS에서 libedit 사용 시 처리
    if 'libedit' in readline.__doc__:
        readline.parse_and_bind("bind ^I rl_complete")
    else:
        readline.parse_and_bind("tab: complete")
except ImportError:
    pass  # Windows에서는 readline이 없을 수 있음

# 현재 디렉토리를 Python 경로에 추가
sys.path.insert(0, str(Path(__file__).parent))

from commands.ask import ask_question
from commands.interactive import interactive_mode
from config import BASE_URL, API_ENDPOINT, STREAM_ENDPOINT

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
    import config
    config.BASE_URL = args.url
    config.API_ENDPOINT = f"{args.url}/api/v1/query"
    config.STREAM_ENDPOINT = f"{args.url}/api/v1/query/stream"
    
    if args.question:
        # 단일 질문 모드
        ask_question(args.question, stream=args.stream, user_id=args.user_id)
    else:
        # 대화형 모드
        interactive_mode()

if __name__ == "__main__":
    main()
