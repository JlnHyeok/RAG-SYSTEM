"""
진행률 추적 관련 유틸리티
"""
import time

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

def create_upload_progress_tracker(file_size, bytes_sent_container, upload_complete_event):
    """업로드 진행률 추적 함수를 생성합니다"""
    def show_upload_progress():
        """업로드 진행 상태 표시"""
        start_time = time.time()
        
        while not upload_complete_event.is_set():
            current_time = time.time()
            elapsed = current_time - start_time
            current_bytes = bytes_sent_container[0]
            
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
    
    return show_upload_progress

def format_speed(bytes_per_second):
    """속도를 적절한 단위로 포맷팅합니다"""
    if bytes_per_second >= 1024 * 1024:
        return f"{bytes_per_second / (1024 * 1024):.1f}MB/s"
    elif bytes_per_second >= 1024:
        return f"{bytes_per_second / 1024:.1f}KB/s"
    else:
        return f"{bytes_per_second:.0f}B/s"

def format_file_size(bytes_size):
    """파일 크기를 적절한 단위로 포맷팅합니다"""
    mb_size = bytes_size / (1024 * 1024)
    if mb_size >= 1:
        return f"{mb_size:.1f}MB"
    else:
        kb_size = bytes_size / 1024
        return f"{kb_size:.1f}KB"

