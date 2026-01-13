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
        """텍스트를 청크로 분할합니다 (문장/문단 경계 고려)"""
        if not text.strip():
            return []
        
        chunks = []
        max_chunk_size = 1500  # 최대 문자 수
        min_chunk_size = 500   # 최소 문자 수
        overlap_size = 200     # 중첩 문자 수
        
        # 문단 단위로 먼저 분할
        paragraphs = text.split('\n\n')
        current_chunk = ""
        chunk_index = 0
        
        for para_idx, paragraph in enumerate(paragraphs):
            paragraph = paragraph.strip()
            if not paragraph:
                continue
                
            # 문단이 너무 긴 경우 문장 단위로 분할
            if len(paragraph) > max_chunk_size:
                sentences = self._split_sentences(paragraph)
                
                for sentence in sentences:
                    sentence = sentence.strip()
                    if not sentence:
                        continue
                        
                    # 현재 청크에 문장을 추가했을 때 크기 확인
                    potential_chunk = current_chunk + ('\n' if current_chunk else '') + sentence
                    
                    if len(potential_chunk) <= max_chunk_size:
                        current_chunk = potential_chunk
                    else:
                        # 현재 청크가 최소 크기 이상이면 저장
                        if len(current_chunk) >= min_chunk_size:
                            chunk = self._create_chunk(current_chunk, filename, chunk_index, metadata)
                            chunks.append(chunk)
                            chunk_index += 1
                            
                            # 중첩을 위해 마지막 부분 유지
                            current_chunk = self._get_overlap_text(current_chunk, overlap_size) + '\n' + sentence
                        else:
                            current_chunk = potential_chunk
            else:
                # 문단이 적당한 크기인 경우
                potential_chunk = current_chunk + ('\n\n' if current_chunk else '') + paragraph
                
                if len(potential_chunk) <= max_chunk_size:
                    current_chunk = potential_chunk
                else:
                    # 현재 청크 저장
                    if len(current_chunk) >= min_chunk_size:
                        chunk = self._create_chunk(current_chunk, filename, chunk_index, metadata)
                        chunks.append(chunk)
                        chunk_index += 1
                        
                        # 중첩을 위해 마지막 부분 유지
                        current_chunk = self._get_overlap_text(current_chunk, overlap_size) + '\n\n' + paragraph
                    else:
                        current_chunk = potential_chunk
        
        # 마지막 청크 처리
        if current_chunk.strip():
            chunk = self._create_chunk(current_chunk, filename, chunk_index, metadata)
            chunks.append(chunk)
        
        return chunks
    
    def _split_sentences(self, text: str) -> List[str]:
        """텍스트를 문장 단위로 분할합니다"""
        import re
        
        # 한국어와 영어 문장 끝 패턴
        sentence_endings = r'[.!?]+\s+|[.!?]+$|[。！？]+\s+|[。！？]+$'
        sentences = re.split(sentence_endings, text)
        
        # 빈 문장 제거 및 정리
        sentences = [s.strip() for s in sentences if s.strip()]
        
        # 너무 짧은 문장들은 합치기
        merged_sentences = []
        current_sentence = ""
        
        for sentence in sentences:
            if len(current_sentence + sentence) < 100:  # 100자 미만이면 합치기
                current_sentence = current_sentence + (' ' if current_sentence else '') + sentence
            else:
                if current_sentence:
                    merged_sentences.append(current_sentence)
                current_sentence = sentence
        
        if current_sentence:
            merged_sentences.append(current_sentence)
            
        return merged_sentences
    
    def _get_overlap_text(self, text: str, overlap_size: int) -> str:
        """텍스트의 마지막 부분에서 중첩할 텍스트를 추출합니다"""
        if len(text) <= overlap_size:
            return text
        
        # 마지막 overlap_size 문자에서 문장 경계 찾기
        overlap_text = text[-overlap_size:]
        
        # 문장의 시작점 찾기 (완전한 문장을 포함하기 위해)
        sentences = self._split_sentences(overlap_text)
        if sentences:
            return sentences[-1] if len(sentences) == 1 else ' '.join(sentences[-2:])
        
        return overlap_text
    
    def _create_chunk(self, content: str, filename: str, chunk_index: int, metadata: Dict[str, Any]) -> DocumentChunk:
        """DocumentChunk 객체를 생성합니다"""
        return DocumentChunk(
            chunk_id=f"{filename}_{chunk_index}",
            content=content.strip(),
            metadata={
                **metadata,
                'filename': filename,
                'chunk_index': chunk_index,
                'char_count': len(content),
                'word_count': len(content.split())
            }
        )


# 전역 인스턴스
document_processor = MultiModalDocumentProcessor()