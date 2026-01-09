"""
멀티모달 문서 처리기
PDF, 이미지, 텍스트 파일을 처리하여 텍스트를 추출하고 청킹합니다.
"""
import io
import logging
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
import tempfile
import os

# 파일 처리
import PyPDF2
import fitz  # PyMuPDF
from PIL import Image
import cv2
import numpy as np

# OCR
import pytesseract
try:
    import easyocr
    EASYOCR_AVAILABLE = True
except ImportError:
    EASYOCR_AVAILABLE = False
    
try:
    from paddleocr import PaddleOCR
    PADDLEOCR_AVAILABLE = True
except ImportError:
    PADDLEOCR_AVAILABLE = False

from app.models.schemas import DocumentChunk, ProcessingResult, OCREngine
from app.models.enums import FileType

logger = logging.getLogger(__name__)

class MultiModalDocumentProcessor:
    """멀티모달 문서 처리기 클래스"""
    
    def __init__(self):
        """초기화 - OCR 엔진들을 설정합니다"""
        self.ocr_engines = {}
        
        # EasyOCR 초기화
        if EASYOCR_AVAILABLE:
            try:
                self.ocr_engines['easyocr'] = easyocr.Reader(['ko', 'en'])
                logger.info("EasyOCR 엔진 초기화 완료")
            except Exception as e:
                logger.warning(f"EasyOCR 초기화 실패: {e}")
        
        # PaddleOCR 초기화
        if PADDLEOCR_AVAILABLE:
            try:
                self.ocr_engines['paddleocr'] = PaddleOCR(
                    use_angle_cls=True,
                    lang='korean'
                )
                logger.info("PaddleOCR 엔진 초기화 완료")
            except Exception as e:
                logger.warning(f"PaddleOCR 초기화 실패: {e}")
        
        logger.info(f"사용 가능한 OCR 엔진: {list(self.ocr_engines.keys())}")

    async def process_document(
        self, 
        file_content: bytes, 
        filename: str, 
        metadata: Optional[Dict[str, Any]] = None
    ) -> ProcessingResult:
        """
        문서를 처리하여 텍스트를 추출하고 청킹합니다.
        
        Args:
            file_content: 파일 내용 (바이트)
            filename: 파일명
            metadata: 추가 메타데이터
            
        Returns:
            ProcessingResult: 처리된 결과
        """
        try:
            file_type = self._detect_file_type(filename)
            logger.info(f"파일 처리 시작: {filename} (타입: {file_type})")
            
            # 파일 타입에 따른 텍스트 추출
            if file_type == FileType.PDF:
                extracted_text = await self._process_pdf(file_content)
            elif file_type in [FileType.IMAGE_PNG, FileType.IMAGE_JPEG, FileType.IMAGE_JPG]:
                extracted_text = await self._process_image(file_content)
            elif file_type in [FileType.TEXT, FileType.MARKDOWN]:
                extracted_text = file_content.decode('utf-8')
            else:
                raise ValueError(f"지원하지 않는 파일 형식: {file_type}")
            
            # 텍스트 청킹
            chunks = self._chunk_text(
                extracted_text, 
                filename, 
                metadata or {}
            )
            
            result = ProcessingResult(
                filename=filename,
                file_type=file_type,
                total_chunks=len(chunks),
                chunks=chunks,
                metadata=metadata or {},
                processing_time=0.0  # 실제 구현시 시간 측정
            )
            
            logger.info(f"문서 처리 완료: {filename}, 청크 수: {len(chunks)}")
            return result
            
        except Exception as e:
            logger.error(f"문서 처리 중 오류 발생: {filename}, 오류: {str(e)}")
            raise

    def _detect_file_type(self, filename: str) -> FileType:
        """파일 확장자를 기반으로 파일 타입을 감지합니다"""
        suffix = Path(filename).suffix.lower()
        
        type_mapping = {
            '.pdf': FileType.PDF,
            '.txt': FileType.TEXT,
            '.md': FileType.MARKDOWN,
            '.png': FileType.IMAGE_PNG,
            '.jpg': FileType.IMAGE_JPEG,
            '.jpeg': FileType.IMAGE_JPEG,
            '.tiff': FileType.IMAGE_TIFF,
            '.tif': FileType.IMAGE_TIFF,
            '.bmp': FileType.IMAGE_BMP
        }
        
        if suffix not in type_mapping:
            raise ValueError(f"지원하지 않는 파일 확장자: {suffix}")
        
        return type_mapping[suffix]

    async def _process_pdf(self, file_content: bytes) -> str:
        """PDF 파일에서 텍스트와 이미지를 추출합니다"""
        extracted_texts = []
        
        try:
            # PyMuPDF를 사용한 텍스트 및 이미지 추출
            pdf_document = fitz.open(stream=file_content, filetype="pdf")
            
            for page_num in range(len(pdf_document)):
                page = pdf_document.load_page(page_num)
                
                # 텍스트 추출
                page_text = page.get_text()
                if page_text.strip():
                    extracted_texts.append(f"[페이지 {page_num + 1}]\n{page_text}")
                
                # 이미지 추출 및 OCR 처리
                image_list = page.get_images()
                for img_index, img in enumerate(image_list):
                    try:
                        # 이미지 데이터 추출
                        xref = img[0]
                        pix = fitz.Pixmap(pdf_document, xref)
                        
                        if pix.n - pix.alpha < 4:  # GRAY 또는 RGB
                            img_data = pix.tobytes("png")
                            
                            # OCR 처리
                            ocr_text = await self._extract_text_from_image(img_data)
                            if ocr_text.strip():
                                extracted_texts.append(
                                    f"[페이지 {page_num + 1} 이미지 {img_index + 1}]\n{ocr_text}"
                                )
                        
                        pix = None
                    except Exception as e:
                        logger.warning(f"PDF 이미지 처리 중 오류: {e}")
                        continue
            
            pdf_document.close()
            
        except Exception as e:
            logger.warning(f"PyMuPDF 처리 실패, PyPDF2로 fallback: {e}")
            
            # PyPDF2 fallback
            try:
                pdf_reader = PyPDF2.PdfReader(io.BytesIO(file_content))
                for page_num, page in enumerate(pdf_reader.pages):
                    page_text = page.extract_text()
                    if page_text.strip():
                        extracted_texts.append(f"[페이지 {page_num + 1}]\n{page_text}")
            except Exception as e2:
                logger.error(f"PyPDF2 처리도 실패: {e2}")
                raise
        
        return "\n\n".join(extracted_texts)

    async def _process_image(self, file_content: bytes) -> str:
        """이미지 파일에서 OCR을 통해 텍스트를 추출합니다"""
        return await self._extract_text_from_image(file_content)

    async def _extract_text_from_image(self, image_data: bytes) -> str:
        """이미지에서 OCR을 통해 텍스트를 추출합니다"""
        try:
            # 이미지 전처리
            processed_image = self._preprocess_image(image_data)
            
            # 여러 OCR 엔진으로 텍스트 추출 시도
            ocr_results = []
            
            # Tesseract OCR
            try:
                tesseract_text = pytesseract.image_to_string(
                    processed_image, 
                    lang='kor+eng',
                    config='--psm 6'
                )
                if tesseract_text.strip():
                    ocr_results.append(('tesseract', tesseract_text.strip()))
            except Exception as e:
                logger.warning(f"Tesseract OCR 실패: {e}")
            
            # EasyOCR
            if 'easyocr' in self.ocr_engines:
                try:
                    # PIL Image를 numpy array로 변환
                    img_array = np.array(processed_image)
                    results = self.ocr_engines['easyocr'].readtext(img_array)
                    
                    easyocr_text = ' '.join([result[1] for result in results if result[2] > 0.5])
                    if easyocr_text.strip():
                        ocr_results.append(('easyocr', easyocr_text.strip()))
                except Exception as e:
                    logger.warning(f"EasyOCR 실패: {e}")
            
            # PaddleOCR
            if 'paddleocr' in self.ocr_engines:
                try:
                    img_array = np.array(processed_image)
                    results = self.ocr_engines['paddleocr'].ocr(img_array, cls=True)
                    
                    paddle_texts = []
                    for line in results[0] or []:
                        if line and len(line) >= 2 and line[1][1] > 0.5:
                            paddle_texts.append(line[1][0])
                    
                    if paddle_texts:
                        paddle_text = ' '.join(paddle_texts)
                        ocr_results.append(('paddleocr', paddle_text.strip()))
                except Exception as e:
                    logger.warning(f"PaddleOCR 실패: {e}")
            
            # 가장 긴 결과 선택 (일반적으로 더 정확함)
            if ocr_results:
                best_result = max(ocr_results, key=lambda x: len(x[1]))
                logger.info(f"OCR 엔진 '{best_result[0]}' 사용, 텍스트 길이: {len(best_result[1])}")
                return best_result[1]
            
            logger.warning("모든 OCR 엔진에서 텍스트 추출 실패")
            return ""
            
        except Exception as e:
            logger.error(f"이미지 OCR 처리 중 오류: {e}")
            return ""

    def _preprocess_image(self, image_data: bytes) -> Image.Image:
        """이미지 전처리를 통해 OCR 성능을 향상시킵니다"""
        try:
            # PIL Image로 로드
            image = Image.open(io.BytesIO(image_data))
            
            # RGB로 변환 (필요시)
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            # numpy array로 변환
            img_array = np.array(image)
            
            # OpenCV를 사용한 이미지 전처리
            # 그레이스케일 변환
            gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
            
            # 노이즈 제거
            denoised = cv2.medianBlur(gray, 3)
            
            # 대비 향상 (CLAHE)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
            enhanced = clahe.apply(denoised)
            
            # 이진화 (적응형 임계값)
            binary = cv2.adaptiveThreshold(
                enhanced, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                cv2.THRESH_BINARY, 11, 2
            )
            
            # PIL Image로 다시 변환
            processed_image = Image.fromarray(binary)
            
            return processed_image
            
        except Exception as e:
            logger.warning(f"이미지 전처리 실패, 원본 사용: {e}")
            return Image.open(io.BytesIO(image_data))

    def _chunk_text(
        self, 
        text: str, 
        filename: str, 
        metadata: Dict[str, Any]
    ) -> List[DocumentChunk]:
        """텍스트를 청크로 분할합니다"""
        if not text.strip():
            return []
        
        chunks = []
        chunk_size = 1000  # 기본값
        chunk_overlap = 200  # 기본값
        
        # 간단한 텍스트 청킹 (실제로는 더 정교한 알고리즘 사용 가능)
        words = text.split()
        
        for i in range(0, len(words), chunk_size - chunk_overlap):
            chunk_words = words[i:i + chunk_size]
            chunk_text = ' '.join(chunk_words)
            
            if chunk_text.strip():
                chunk = DocumentChunk(
                    chunk_id=f"{filename}_{len(chunks)}",
                    content=chunk_text,
                    metadata={
                        **metadata,
                        'filename': filename,
                        'chunk_index': len(chunks),
                        'start_word': i,
                        'end_word': min(i + chunk_size, len(words))
                    }
                )
                chunks.append(chunk)
        
        return chunks


# 전역 인스턴스
document_processor = MultiModalDocumentProcessor()