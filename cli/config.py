"""
설정 파일 - CLI 전역 설정
"""

# 기본 설정
BASE_URL = "http://localhost:8000"
API_ENDPOINT = f"{BASE_URL}/api/v1/query"
STREAM_ENDPOINT = f"{BASE_URL}/api/v1/query/stream"

# 지원 파일 형식
SUPPORTED_EXTENSIONS = ['.txt', '.pdf', '.md', '.doc', '.docx']

# 기본 타임아웃 설정
DEFAULT_TIMEOUT = 30
UPLOAD_TIMEOUT = 1800  # 30분
API_TIMEOUT = 10
