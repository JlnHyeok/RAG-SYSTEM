"""
파일 관련 유틸리티
"""
import os
import glob
from config import SUPPORTED_EXTENSIONS

def show_current_files():
    """현재 디렉토리의 파일 목록을 표시"""
    # 일반적인 문서 파일 확장자들
    patterns = [f'*{ext}' for ext in SUPPORTED_EXTENSIONS]
    
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
        print(f"     지원 형식: {', '.join(SUPPORTED_EXTENSIONS)}")
    
    print("💡 사용법: /upload <파일명> (예: /upload document.pdf)")
    print()
