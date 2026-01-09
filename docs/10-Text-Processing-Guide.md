# 텍스트 전처리 및 청킹 전략 가이드

## 1. 텍스트 전처리 개요

텍스트 전처리는 RAG 시스템에서 **문서의 품질을 향상시키고 검색 정확도를 높이는 핵심 과정**입니다. 원시 텍스트를 AI 모델이 더 잘 이해할 수 있도록 정제하고 구조화합니다.

### 텍스트 전처리의 이론적 배경

**왜 전처리가 필요한가?**

1. **토큰화의 일관성**: AI 모델은 텍스트를 토큰 단위로 처리하는데, 불일치한 형식은 같은 의미의 단어를 다른 토큰으로 인식하게 만듭니다.

   - 예: "AI", "A.I.", "인공지능" → 모두 같은 개념이지만 다른 토큰으로 처리

2. **벡터 공간의 효율성**: 임베딩 모델이 생성하는 벡터 공간에서 유사한 의미의 텍스트가 가까운 위치에 배치되려면 일관된 형식이 필요합니다.

3. **노이즈 제거**: HTML 태그, 특수문자, 불필요한 공백 등은 의미적 유사성 계산을 방해하는 노이즈로 작용합니다.

4. **검색 정확도**: 전처리된 텍스트는 사용자 쿼리와 문서 간의 매칭 정확도를 크게 향상시킵니다.

**전처리의 품질이 RAG 성능에 미치는 영향**

- **Recall**: 관련 문서를 얼마나 많이 찾아내는가
- **Precision**: 찾아낸 문서 중 실제로 관련된 것의 비율
- **Response Quality**: LLM이 생성하는 답변의 품질

### 1.1 전처리의 중요성

```python
# 전처리 전후 비교
raw_text = """
    안녕하세요!!!    Python은    정말 좋은 프로그래밍 언어입니다.

        하지만 배우기가 쉽지않아요ㅠㅠ   그래도 계속 공부해야겠어요!

HTML 태그가 있네요: <div>이런 것들</div>

    email@example.com 이런 이메일도 있고...
"""

processed_text = """
안녕하세요! Python은 정말 좋은 프로그래밍 언어입니다.
하지만 배우기가 쉽지 않아요. 그래도 계속 공부해야겠어요!
HTML 태그가 있네요: 이런 것들
[EMAIL] 이런 이메일도 있고...
"""
```

### 1.2 전처리의 주요 목표

1. **노이즈 제거**: 불필요한 문자, 태그, 특수기호 제거
2. **정규화**: 일관된 형식으로 변환
3. **구조화**: 의미 있는 단위로 분할
4. **개체 보호**: 중요한 정보 마스킹 또는 변환

## 2. 기본 텍스트 정제

### 텍스트 정제의 단계별 접근법

텍스트 정제는 다음과 같은 계층적 접근 방식을 따릅니다:

**1단계: 구조적 정제**

- HTML/XML 태그 제거
- 문서 구조 파싱 (헤더, 푸터, 메타데이터 분리)
- 인코딩 문제 해결

**2단계: 문자 수준 정제**

- 유니코드 정규화 (NFD ↔ NFC)
- 특수문자 및 제어문자 처리
- 공백 문자 통일

**3단계: 토큰 수준 정제**

- 개인정보 마스킹
- URL, 이메일 등 특수 패턴 처리
- 반복 문자 정규화

**4단계: 의미 수준 정제**

- 맞춤법 검사 및 교정
- 약어 확장
- 동의어 통일

### 정제 수준에 따른 트레이드오프

- **Conservative**: 원본 정보 최대 보존, 노이즈는 일부 남음
- **Moderate**: 균형잡힌 접근, 대부분의 실무 환경에 적합
- **Aggressive**: 최대한 정제, 일부 정보 손실 가능성

### 2.1 문자 및 형식 정제

```python
import re
import unicodedata
from bs4 import BeautifulSoup

class BasicTextCleaner:
    def __init__(self):
        # 정규식 패턴들
        self.email_pattern = re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b')
        self.phone_pattern = re.compile(r'(\d{2,3}[-\.\s]?\d{3,4}[-\.\s]?\d{4})')
        self.url_pattern = re.compile(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+')
        self.excessive_whitespace = re.compile(r'\s+')
        self.special_chars = re.compile(r'[^\w\s\.\,\!\?\:\;\"\'()가-힣]')

    def remove_html_tags(self, text):
        """HTML 태그 제거"""
        soup = BeautifulSoup(text, 'html.parser')
        return soup.get_text()

    def normalize_whitespace(self, text):
        """공백 정규화"""
        # 탭, 개행문자를 공백으로 변환
        text = re.sub(r'[\t\r\n]', ' ', text)
        # 연속된 공백을 하나로 축소
        text = self.excessive_whitespace.sub(' ', text)
        return text.strip()

    def remove_excessive_punctuation(self, text):
        """과도한 문장부호 정리"""
        # 연속된 느낌표, 물음표 정리
        text = re.sub(r'[!]{2,}', '!', text)
        text = re.sub(r'[?]{2,}', '?', text)
        text = re.sub(r'[\.]{2,}', '...', text)
        return text

    def mask_personal_info(self, text, mask_emails=True, mask_phones=True):
        """개인정보 마스킹"""
        if mask_emails:
            text = self.email_pattern.sub('[EMAIL]', text)
        if mask_phones:
            text = self.phone_pattern.sub('[PHONE]', text)
        return text

    def normalize_unicode(self, text):
        """유니코드 정규화"""
        # NFD 정규화 후 NFC로 재정규화
        text = unicodedata.normalize('NFD', text)
        text = unicodedata.normalize('NFC', text)
        return text

    def clean_text(self, text, remove_html=True, mask_pii=True):
        """전체 텍스트 정제 파이프라인"""
        if remove_html:
            text = self.remove_html_tags(text)

        text = self.normalize_unicode(text)
        text = self.normalize_whitespace(text)
        text = self.remove_excessive_punctuation(text)

        if mask_pii:
            text = self.mask_personal_info(text)

        return text

# 사용 예시
cleaner = BasicTextCleaner()

dirty_text = """
<p>안녕하세요!!!    제 이메일은 test@example.com 이고
전화번호는 010-1234-5678 입니다.</p>

    <div>    Python은 좋은 언어예요ㅠㅠㅠ    </div>
"""

clean_text = cleaner.clean_text(dirty_text)
print("정제된 텍스트:", clean_text)
# 출력: 안녕하세요! 제 이메일은 [EMAIL] 이고 전화번호는 [PHONE] 입니다. Python은 좋은 언어예요.
```

### 2.2 언어별 특화 전처리

#### 한국어 전처리

```python
import re
from konlpy.tag import Okt, Kkma
from soynlp.normalizer import repeat_normalize

class KoreanTextPreprocessor:
    def __init__(self):
        self.okt = Okt()
        self.kkma = Kkma()

        # 한국어 특수 패턴
        self.jamo_pattern = re.compile(r'[ㄱ-ㅎㅏ-ㅣ]+')  # 자모음 단독
        self.repeated_char = re.compile(r'(.)\1{2,}')  # 반복 문자
        self.emoticon_pattern = re.compile(r'[ㅋㅎㅠㅜㅡ]{2,}')  # 이모티콘

    def normalize_repeats(self, text):
        """반복 문자 정규화 (ㅋㅋㅋ -> ㅋㅋ)"""
        text = repeat_normalize(text, num_repeats=2)
        return text

    def remove_jamo(self, text):
        """단독 자모음 제거"""
        return self.jamo_pattern.sub('', text)

    def normalize_emoticons(self, text):
        """이모티콘 정규화"""
        # ㅋㅋㅋㅋ -> ㅋㅋ
        text = re.sub(r'ㅋ{2,}', 'ㅋㅋ', text)
        text = re.sub(r'ㅎ{2,}', 'ㅎㅎ', text)
        text = re.sub(r'ㅠ{2,}', 'ㅠㅠ', text)
        return text

    def extract_morphemes(self, text, pos_filter=None):
        """형태소 분석"""
        morphemes = self.okt.pos(text, norm=True, stem=True)

        if pos_filter:
            # 특정 품사만 필터링 (명사, 동사, 형용사 등)
            morphemes = [(word, pos) for word, pos in morphemes if pos in pos_filter]

        return morphemes

    def preprocess_korean(self, text):
        """한국어 특화 전처리"""
        text = self.normalize_repeats(text)
        text = self.normalize_emoticons(text)
        text = self.remove_jamo(text)

        # 공백 정규화
        text = re.sub(r'\s+', ' ', text).strip()

        return text

# 사용 예시
korean_processor = KoreanTextPreprocessor()

korean_text = "안녕하세요ㅋㅋㅋㅋㅋㅋ 오늘 날씨가 정말정말정말 좋네요ㅠㅠㅠㅠ ㅎㅎㅎ"
processed = korean_processor.preprocess_korean(korean_text)
print("전처리된 텍스트:", processed)

# 형태소 분석
morphemes = korean_processor.extract_morphemes(processed, pos_filter=['Noun', 'Verb', 'Adjective'])
print("형태소:", morphemes)
```

#### 영어 전처리

```python
import nltk
import string
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.tokenize import word_tokenize, sent_tokenize

class EnglishTextPreprocessor:
    def __init__(self):
        # NLTK 데이터 다운로드 (필요시)
        nltk.download('punkt', quiet=True)
        nltk.download('stopwords', quiet=True)
        nltk.download('wordnet', quiet=True)

        self.stemmer = PorterStemmer()
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))

    def expand_contractions(self, text):
        """축약형 확장"""
        contractions = {
            "won't": "will not",
            "can't": "cannot",
            "n't": " not",
            "'re": " are",
            "'ve": " have",
            "'ll": " will",
            "'d": " would",
            "'m": " am"
        }

        for contraction, expansion in contractions.items():
            text = text.replace(contraction, expansion)

        return text

    def remove_punctuation(self, text, keep_sentence_endings=True):
        """구두점 제거"""
        if keep_sentence_endings:
            # 문장 끝 구두점 유지
            translator = str.maketrans('', '', string.punctuation.replace('.', '').replace('!', '').replace('?', ''))
        else:
            translator = str.maketrans('', '', string.punctuation)

        return text.translate(translator)

    def tokenize_and_clean(self, text, remove_stopwords=True, use_stemming=False, use_lemmatization=True):
        """토큰화 및 정제"""
        # 소문자 변환
        text = text.lower()

        # 축약형 확장
        text = self.expand_contractions(text)

        # 토큰화
        tokens = word_tokenize(text)

        # 구두점 제거
        tokens = [token for token in tokens if token.isalpha()]

        # 불용어 제거
        if remove_stopwords:
            tokens = [token for token in tokens if token not in self.stop_words]

        # 어간 추출 또는 표제어 추출
        if use_stemming:
            tokens = [self.stemmer.stem(token) for token in tokens]
        elif use_lemmatization:
            tokens = [self.lemmatizer.lemmatize(token) for token in tokens]

        return tokens

    def preprocess_english(self, text, join_result=True):
        """영어 전체 전처리"""
        tokens = self.tokenize_and_clean(text)

        if join_result:
            return ' '.join(tokens)
        else:
            return tokens

# 사용 예시
english_processor = EnglishTextPreprocessor()

english_text = "I can't believe it's working! The results are amazing, aren't they?"
processed = english_processor.preprocess_english(english_text)
print("전처리된 텍스트:", processed)
# 출력: believe work result amazing
```

## 3. 고급 전처리 기법

### 3.1 도메인 특화 전처리

````python
class DomainSpecificPreprocessor:
    def __init__(self, domain='general'):
        self.domain = domain
        self.setup_domain_patterns()

    def setup_domain_patterns(self):
        """도메인별 패턴 설정"""
        self.patterns = {}

        if self.domain == 'legal':
            self.patterns.update({
                'case_citation': re.compile(r'\d+\s+[A-Za-z\.]+\s+\d+'),
                'statute': re.compile(r'§\s*\d+(?:\.\d+)*'),
                'legal_term': re.compile(r'\b(?:plaintiff|defendant|whereas|hereby)\b', re.IGNORECASE)
            })

        elif self.domain == 'medical':
            self.patterns.update({
                'medication': re.compile(r'\b[A-Za-z]+(?:mg|ml|mcg|g|l)\b'),
                'diagnosis_code': re.compile(r'\b[A-Z]\d{2}\.\d+\b'),
                'medical_term': re.compile(r'\b(?:diagnosis|prescription|symptom|treatment)\b', re.IGNORECASE)
            })

        elif self.domain == 'technical':
            self.patterns.update({
                'version': re.compile(r'v?\d+\.\d+(?:\.\d+)?'),
                'code_block': re.compile(r'```[\s\S]*?```'),
                'api_endpoint': re.compile(r'/api/v\d+/[a-zA-Z0-9/_]+')
            })

    def extract_domain_entities(self, text):
        """도메인 특화 개체 추출"""
        entities = {}

        for entity_type, pattern in self.patterns.items():
            matches = pattern.findall(text)
            if matches:
                entities[entity_type] = matches

        return entities

    def mask_domain_entities(self, text):
        """도메인 개체 마스킹"""
        for entity_type, pattern in self.patterns.items():
            mask_token = f'[{entity_type.upper()}]'
            text = pattern.sub(mask_token, text)

        return text

    def preprocess_domain_text(self, text, extract_entities=True, mask_entities=False):
        """도메인 특화 전처리"""
        result = {
            'processed_text': text,
            'entities': {}
        }

        if extract_entities:
            result['entities'] = self.extract_domain_entities(text)

        if mask_entities:
            result['processed_text'] = self.mask_domain_entities(text)

        return result

# 사용 예시
medical_processor = DomainSpecificPreprocessor('medical')

medical_text = """
Patient diagnosed with A01.1 condition.
Prescribed 500mg acetaminophen twice daily.
Treatment shows positive results.
"""

result = medical_processor.preprocess_domain_text(medical_text, extract_entities=True)
print("추출된 의료 개체:", result['entities'])
````

### 3.2 노이즈 감지 및 처리

```python
import numpy as np
from collections import Counter

class NoiseDetector:
    def __init__(self):
        self.noise_patterns = {
            'repetitive_chars': re.compile(r'(.)\1{5,}'),  # 5회 이상 반복
            'random_strings': re.compile(r'\b[a-zA-Z]{20,}\b'),  # 20자 이상 랜덤 문자열
            'excessive_caps': re.compile(r'[A-Z]{10,}'),  # 10자 이상 대문자
            'garbage_unicode': re.compile(r'[\u2000-\u206F\u2E00-\u2E7F\uFE30-\uFE4F\uFE50-\uFE6F]'),
        }

    def calculate_text_quality_score(self, text):
        """텍스트 품질 점수 계산"""
        if not text or len(text.strip()) < 10:
            return 0.0

        score = 1.0
        penalties = {}

        # 1. 문자 다양성 체크
        char_diversity = len(set(text)) / len(text)
        if char_diversity < 0.1:  # 문자 다양성이 10% 미만
            penalties['low_diversity'] = 0.3

        # 2. 반복 패턴 체크
        for pattern_name, pattern in self.noise_patterns.items():
            matches = len(pattern.findall(text))
            if matches > 0:
                penalties[pattern_name] = min(0.5, matches * 0.1)

        # 3. 언어 일관성 체크
        ascii_ratio = sum(1 for c in text if ord(c) < 128) / len(text)
        if 0.3 < ascii_ratio < 0.7:  # 혼재된 언어
            penalties['mixed_language'] = 0.2

        # 4. 문장 구조 체크
        sentences = re.split(r'[.!?]', text)
        avg_sentence_length = np.mean([len(s.split()) for s in sentences if s.strip()])
        if avg_sentence_length < 3 or avg_sentence_length > 50:
            penalties['abnormal_sentence_length'] = 0.2

        # 총 점수 계산
        total_penalty = sum(penalties.values())
        final_score = max(0.0, score - total_penalty)

        return {
            'quality_score': final_score,
            'penalties': penalties,
            'is_high_quality': final_score >= 0.7
        }

    def filter_noisy_chunks(self, text_chunks, min_quality=0.5):
        """품질 기준으로 노이즈 청크 필터링"""
        filtered_chunks = []

        for chunk in text_chunks:
            quality = self.calculate_text_quality_score(chunk)
            if quality['quality_score'] >= min_quality:
                filtered_chunks.append({
                    'text': chunk,
                    'quality': quality['quality_score']
                })

        return filtered_chunks

# 사용 예시
noise_detector = NoiseDetector()

chunks = [
    "이것은 정상적인 텍스트입니다. 의미 있는 내용을 담고 있습니다.",
    "aaaaaaaaaaaaaaaaaa",  # 반복 문자
    "THISISALLCAPSTEXT",   # 전체 대문자
    "ㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋ",  # 반복 자모
    "Python은 프로그래밍 언어입니다."
]

filtered_chunks = noise_detector.filter_noisy_chunks(chunks)
print("필터링된 고품질 청크들:")
for chunk_info in filtered_chunks:
    print(f"점수: {chunk_info['quality']:.2f} - {chunk_info['text'][:50]}")
```

## 4. 청킹(Chunking) 전략

### 청킹의 이론적 배경

청킹은 긴 문서를 **의미적으로 일관된 작은 단위로 분할**하는 과정입니다. RAG 시스템에서 청킹의 품질은 검색 성능과 답변 품질에 결정적 영향을 미칩니다.

**청킹이 중요한 이유:**

1. **벡터 DB 제약**: 대부분의 벡터 DB에는 벡터 차원 수 제한이 있음

   - OpenAI text-embedding-ada-002: 1536 차원
   - 일반적으로 512-2048 토큰 길이의 텍스트가 최적

2. **맥락 유지**: 너무 작으면 맥락 손실, 너무 크면 노이즈 증가

3. **검색 정밀도**: 사용자 쿼리와 가장 관련 있는 내용만 추출

4. **LLM 컨텍스트 창 효율**: 입력 토큰 수 제한 내에서 최대 정보 활용

### 청킹 전략 비교

| 전략            | 장점              | 단점             | 적용 사례            |
| --------------- | ----------------- | ---------------- | -------------------- |
| **고정 크기**   | 간단, 일관된 크기 | 의미 단위 무시   | 규격화된 문서        |
| **문장 단위**   | 자연스러운 경계   | 크기 편차 심함   | 논문, 법률 문서      |
| **단락 단위**   | 의미 일관성 보장  | 의존성 발생 가능 | 블로그, 뉴스 기사    |
| **의미적 청킹** | 최적의 정보 밀도  | 복잡한 처리      | 전문 서적, 기술 문서 |
| **계층적 청킹** | 유연한 규모       | 구현 복잡성      | 구조화된 대용량 문서 |

### 청킹 품질 지표

1. **Coherence (일관성)**: 각 청크 내에서 내용이 논리적으로 연결되는 정도
2. **Completeness (완성도)**: 핵심 정보가 빠지지 않고 포함되는 정도
3. **Distinctiveness (차별성)**: 청크 간 내용 중복을 최소화하는 정도
4. **Retrievability (검색성)**: 쿼리와 매칭되어 검색되기 쉬운 정도

### 4.1 기본 청킹 방법

```python
from abc import ABC, abstractmethod

class BaseChunker(ABC):
    def __init__(self, chunk_size=1000, chunk_overlap=200):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    @abstractmethod
    def chunk_text(self, text):
        pass

    def create_chunks_with_metadata(self, text, source_metadata=None):
        """메타데이터가 포함된 청크 생성"""
        chunks = self.chunk_text(text)
        chunk_objects = []

        for i, chunk_text in enumerate(chunks):
            chunk_metadata = {
                'chunk_id': i,
                'chunk_size': len(chunk_text),
                'total_chunks': len(chunks),
                'overlap_start': max(0, i * self.chunk_size - self.chunk_overlap),
                'overlap_end': min(len(text), (i + 1) * self.chunk_size + self.chunk_overlap)
            }

            if source_metadata:
                chunk_metadata.update(source_metadata)

            chunk_objects.append({
                'text': chunk_text,
                'metadata': chunk_metadata
            })

        return chunk_objects

class FixedSizeChunker(BaseChunker):
    """고정 크기 청킹"""

    def chunk_text(self, text):
        chunks = []
        text_length = len(text)

        for i in range(0, text_length, self.chunk_size - self.chunk_overlap):
            chunk_end = min(i + self.chunk_size, text_length)
            chunk = text[i:chunk_end]

            if len(chunk.strip()) > 0:
                chunks.append(chunk)

            if chunk_end >= text_length:
                break

        return chunks

class SentenceAwareChunker(BaseChunker):
    """문장 경계 인식 청킹"""

    def __init__(self, chunk_size=1000, chunk_overlap=200, language='ko'):
        super().__init__(chunk_size, chunk_overlap)
        self.language = language
        self._setup_sentence_splitter()

    def _setup_sentence_splitter(self):
        if self.language == 'ko':
            # 한국어 문장 분리 패턴
            self.sentence_pattern = re.compile(r'[.!?]+\s+')
        else:
            # 영어 문장 분리 패턴
            self.sentence_pattern = re.compile(r'[.!?]+\s+')

    def split_sentences(self, text):
        """문장 분리"""
        sentences = self.sentence_pattern.split(text)
        # 마지막 빈 문장 제거
        return [s.strip() for s in sentences if s.strip()]

    def chunk_text(self, text):
        sentences = self.split_sentences(text)
        chunks = []
        current_chunk = ""

        for sentence in sentences:
            # 현재 청크에 문장을 추가했을 때 크기 확인
            potential_chunk = current_chunk + " " + sentence if current_chunk else sentence

            if len(potential_chunk) <= self.chunk_size:
                current_chunk = potential_chunk
            else:
                # 현재 청크를 저장하고 새 청크 시작
                if current_chunk:
                    chunks.append(current_chunk)

                # 오버랩 처리
                if chunks and self.chunk_overlap > 0:
                    overlap_text = self._get_overlap_text(current_chunk)
                    current_chunk = overlap_text + " " + sentence
                else:
                    current_chunk = sentence

        # 마지막 청크 추가
        if current_chunk:
            chunks.append(current_chunk)

        return chunks

    def _get_overlap_text(self, text):
        """오버랩 텍스트 추출"""
        if len(text) <= self.chunk_overlap:
            return text
        return text[-self.chunk_overlap:]

class SemanticChunker(BaseChunker):
    """의미론적 청킹"""

    def __init__(self, embedding_model, similarity_threshold=0.7, chunk_size=1000):
        super().__init__(chunk_size)
        self.embedding_model = embedding_model
        self.similarity_threshold = similarity_threshold

    def chunk_text(self, text):
        sentences = self._split_into_sentences(text)

        if not sentences:
            return []

        # 문장별 임베딩 생성
        sentence_embeddings = self.embedding_model.encode(sentences)

        chunks = []
        current_chunk_sentences = [sentences[0]]
        current_chunk_embedding = sentence_embeddings[0:1]

        for i in range(1, len(sentences)):
            sentence = sentences[i]
            sentence_embedding = sentence_embeddings[i:i+1]

            # 현재 청크와의 유사도 계산
            similarity = self._cosine_similarity(
                current_chunk_embedding.mean(axis=0, keepdims=True),
                sentence_embedding
            )[0][0]

            current_chunk_text = " ".join(current_chunk_sentences + [sentence])

            # 유사도가 높고 크기 제한 내에 있으면 추가
            if similarity >= self.similarity_threshold and len(current_chunk_text) <= self.chunk_size:
                current_chunk_sentences.append(sentence)
                current_chunk_embedding = np.vstack([current_chunk_embedding, sentence_embedding])
            else:
                # 현재 청크 완료, 새 청크 시작
                chunks.append(" ".join(current_chunk_sentences))
                current_chunk_sentences = [sentence]
                current_chunk_embedding = sentence_embedding

        # 마지막 청크 추가
        if current_chunk_sentences:
            chunks.append(" ".join(current_chunk_sentences))

        return chunks

    def _split_into_sentences(self, text):
        """문장 분리 (간단한 버전)"""
        return [s.strip() for s in re.split(r'[.!?]+', text) if s.strip()]

    def _cosine_similarity(self, a, b):
        """코사인 유사도 계산"""
        from sklearn.metrics.pairwise import cosine_similarity
        return cosine_similarity(a, b)

# 사용 예시
text = """
Python은 1991년 귀도 반 로섬이 개발한 프로그래밍 언어입니다.
간단하고 읽기 쉬운 문법으로 유명합니다.
많은 개발자들이 Python을 선호하는 이유입니다.
데이터 과학과 웹 개발에 널리 사용됩니다.
머신러닝과 AI 분야에서도 인기가 높습니다.
"""

# 고정 크기 청킹
fixed_chunker = FixedSizeChunker(chunk_size=100, chunk_overlap=20)
fixed_chunks = fixed_chunker.create_chunks_with_metadata(text)

print("고정 크기 청킹:")
for chunk in fixed_chunks:
    print(f"청크 {chunk['metadata']['chunk_id']}: {chunk['text'][:50]}...")

# 문장 인식 청킹
sentence_chunker = SentenceAwareChunker(chunk_size=150, chunk_overlap=30)
sentence_chunks = sentence_chunker.create_chunks_with_metadata(text)

print("\n문장 인식 청킹:")
for chunk in sentence_chunks:
    print(f"청크 {chunk['metadata']['chunk_id']}: {chunk['text']}")
```

### 4.2 계층적 청킹

```python
class HierarchicalChunker:
    """계층적 문서 청킹"""

    def __init__(self):
        self.section_patterns = {
            'title': re.compile(r'^#\s+(.+)$', re.MULTILINE),
            'subtitle': re.compile(r'^##\s+(.+)$', re.MULTILINE),
            'subsubtitle': re.compile(r'^###\s+(.+)$', re.MULTILINE),
        }
        self.paragraph_pattern = re.compile(r'\n\s*\n')

    def extract_document_structure(self, text):
        """문서 구조 추출"""
        structure = {
            'sections': [],
            'metadata': {
                'total_length': len(text),
                'has_titles': bool(self.section_patterns['title'].search(text)),
                'paragraph_count': len(self.paragraph_pattern.split(text))
            }
        }

        # 제목별로 섹션 분리
        title_matches = list(self.section_patterns['title'].finditer(text))

        if not title_matches:
            # 제목이 없는 경우 전체를 하나의 섹션으로 처리
            structure['sections'].append({
                'title': 'Document',
                'level': 0,
                'start_pos': 0,
                'end_pos': len(text),
                'content': text
            })
            return structure

        for i, match in enumerate(title_matches):
            title = match.group(1)
            start_pos = match.start()
            end_pos = title_matches[i + 1].start() if i + 1 < len(title_matches) else len(text)

            section_content = text[start_pos:end_pos]

            structure['sections'].append({
                'title': title,
                'level': 1,
                'start_pos': start_pos,
                'end_pos': end_pos,
                'content': section_content,
                'subsections': self._extract_subsections(section_content)
            })

        return structure

    def _extract_subsections(self, section_text):
        """하위 섹션 추출"""
        subsections = []
        subtitle_matches = list(self.section_patterns['subtitle'].finditer(section_text))

        for i, match in enumerate(subtitle_matches):
            title = match.group(1)
            start_pos = match.start()
            end_pos = subtitle_matches[i + 1].start() if i + 1 < len(subtitle_matches) else len(section_text)

            subsections.append({
                'title': title,
                'level': 2,
                'start_pos': start_pos,
                'end_pos': end_pos,
                'content': section_text[start_pos:end_pos]
            })

        return subsections

    def create_hierarchical_chunks(self, document_structure, max_chunk_size=1000):
        """계층적 청크 생성"""
        chunks = []

        for section in document_structure['sections']:
            section_chunks = self._chunk_section(section, max_chunk_size)
            chunks.extend(section_chunks)

        return chunks

    def _chunk_section(self, section, max_chunk_size):
        """섹션별 청킹"""
        chunks = []

        # 섹션이 충분히 작으면 그대로 사용
        if len(section['content']) <= max_chunk_size:
            chunks.append({
                'text': section['content'],
                'metadata': {
                    'section_title': section['title'],
                    'section_level': section['level'],
                    'chunk_type': 'complete_section'
                }
            })
        else:
            # 하위 섹션이 있으면 하위 섹션별로 청킹
            if 'subsections' in section and section['subsections']:
                for subsection in section['subsections']:
                    subsection_chunks = self._chunk_section(subsection, max_chunk_size)
                    chunks.extend(subsection_chunks)
            else:
                # 문단 단위로 청킹
                paragraphs = self.paragraph_pattern.split(section['content'])
                current_chunk = ""

                for paragraph in paragraphs:
                    if not paragraph.strip():
                        continue

                    potential_chunk = current_chunk + "\n\n" + paragraph if current_chunk else paragraph

                    if len(potential_chunk) <= max_chunk_size:
                        current_chunk = potential_chunk
                    else:
                        if current_chunk:
                            chunks.append({
                                'text': current_chunk,
                                'metadata': {
                                    'section_title': section['title'],
                                    'section_level': section['level'],
                                    'chunk_type': 'partial_section'
                                }
                            })
                        current_chunk = paragraph

                if current_chunk:
                    chunks.append({
                        'text': current_chunk,
                        'metadata': {
                            'section_title': section['title'],
                            'section_level': section['level'],
                            'chunk_type': 'partial_section'
                        }
                    })

        return chunks

# 사용 예시
markdown_text = """
# Python 기초

Python은 프로그래밍 언어입니다.

## 변수와 데이터 타입

Python에서 변수를 선언하는 방법을 알아봅시다.
숫자, 문자열, 리스트 등 다양한 데이터 타입이 있습니다.

## 함수 정의

함수는 코드의 재사용성을 높입니다.
def 키워드를 사용하여 함수를 정의할 수 있습니다.

# 고급 Python

## 객체지향 프로그래밍

클래스와 객체의 개념을 배워봅시다.
상속, 캡슐화, 다형성 등이 핵심 개념입니다.
"""

hierarchical_chunker = HierarchicalChunker()
doc_structure = hierarchical_chunker.extract_document_structure(markdown_text)
hierarchical_chunks = hierarchical_chunker.create_hierarchical_chunks(doc_structure)

print("계층적 청킹 결과:")
for i, chunk in enumerate(hierarchical_chunks):
    print(f"청크 {i+1}: {chunk['metadata']['section_title']} (레벨 {chunk['metadata']['section_level']})")
    print(f"내용: {chunk['text'][:100]}...")
    print()
```

### 4.3 청킹 품질 평가

```python
class ChunkQualityEvaluator:
    def __init__(self, embedding_model=None):
        self.embedding_model = embedding_model

    def evaluate_chunk_coherence(self, chunks):
        """청크 간 일관성 평가"""
        if not self.embedding_model:
            raise ValueError("임베딩 모델이 필요합니다.")

        chunk_texts = [chunk['text'] if isinstance(chunk, dict) else chunk for chunk in chunks]
        embeddings = self.embedding_model.encode(chunk_texts)

        # 인접 청크 간 유사도 계산
        coherence_scores = []
        for i in range(len(embeddings) - 1):
            similarity = self._cosine_similarity(
                embeddings[i:i+1],
                embeddings[i+1:i+2]
            )[0][0]
            coherence_scores.append(similarity)

        return {
            'average_coherence': np.mean(coherence_scores),
            'coherence_scores': coherence_scores,
            'min_coherence': np.min(coherence_scores),
            'max_coherence': np.max(coherence_scores)
        }

    def evaluate_chunk_sizes(self, chunks):
        """청크 크기 분포 평가"""
        sizes = []
        for chunk in chunks:
            if isinstance(chunk, dict):
                text = chunk.get('text', '')
            else:
                text = chunk
            sizes.append(len(text))

        return {
            'average_size': np.mean(sizes),
            'size_std': np.std(sizes),
            'min_size': np.min(sizes),
            'max_size': np.max(sizes),
            'size_distribution': np.histogram(sizes, bins=10)
        }

    def evaluate_information_density(self, chunks):
        """정보 밀도 평가"""
        densities = []

        for chunk in chunks:
            text = chunk['text'] if isinstance(chunk, dict) else chunk

            # 단어 수
            word_count = len(text.split())

            # 고유 단어 수
            unique_words = len(set(text.lower().split()))

            # 문장 수
            sentence_count = len(re.split(r'[.!?]+', text))

            # 정보 밀도 계산
            if word_count > 0:
                lexical_diversity = unique_words / word_count
                avg_sentence_length = word_count / max(1, sentence_count)

                # 0-1 스케일의 밀도 점수
                density = min(1.0, lexical_diversity * (avg_sentence_length / 20))
                densities.append(density)
            else:
                densities.append(0.0)

        return {
            'average_density': np.mean(densities),
            'density_scores': densities,
            'low_density_count': sum(1 for d in densities if d < 0.3)
        }

    def comprehensive_evaluation(self, chunks):
        """종합 평가"""
        evaluation = {}

        try:
            evaluation['coherence'] = self.evaluate_chunk_coherence(chunks)
        except:
            evaluation['coherence'] = None

        evaluation['size_analysis'] = self.evaluate_chunk_sizes(chunks)
        evaluation['density_analysis'] = self.evaluate_information_density(chunks)

        # 종합 점수 계산
        size_score = self._calculate_size_score(evaluation['size_analysis'])
        density_score = evaluation['density_analysis']['average_density']

        if evaluation['coherence']:
            coherence_score = evaluation['coherence']['average_coherence']
            overall_score = (size_score + density_score + coherence_score) / 3
        else:
            overall_score = (size_score + density_score) / 2

        evaluation['overall_quality'] = {
            'score': overall_score,
            'grade': self._assign_grade(overall_score)
        }

        return evaluation

    def _calculate_size_score(self, size_analysis):
        """크기 기반 점수 계산"""
        avg_size = size_analysis['average_size']
        size_std = size_analysis['size_std']

        # 적절한 크기 범위 (500-1500자)
        optimal_min, optimal_max = 500, 1500

        if optimal_min <= avg_size <= optimal_max:
            size_score = 1.0 - (size_std / avg_size)  # 표준편차가 작을수록 좋음
        else:
            # 범위를 벗어나면 감점
            deviation = min(abs(avg_size - optimal_min), abs(avg_size - optimal_max))
            size_score = max(0.0, 1.0 - deviation / 1000)

        return max(0.0, min(1.0, size_score))

    def _assign_grade(self, score):
        """점수에 따른 등급 부여"""
        if score >= 0.8:
            return 'A'
        elif score >= 0.6:
            return 'B'
        elif score >= 0.4:
            return 'C'
        else:
            return 'D'

    def _cosine_similarity(self, a, b):
        """코사인 유사도 계산"""
        from sklearn.metrics.pairwise import cosine_similarity
        return cosine_similarity(a, b)

# 사용 예시 (가상의 임베딩 모델 사용)
class MockEmbeddingModel:
    def encode(self, texts):
        # 실제로는 sentence-transformers 등 사용
        return np.random.randn(len(texts), 384)

evaluator = ChunkQualityEvaluator(MockEmbeddingModel())
evaluation_result = evaluator.comprehensive_evaluation(sentence_chunks)

print("청킹 품질 평가 결과:")
print(f"전체 품질 점수: {evaluation_result['overall_quality']['score']:.2f}")
print(f"품질 등급: {evaluation_result['overall_quality']['grade']}")
print(f"평균 청크 크기: {evaluation_result['size_analysis']['average_size']:.0f}자")
print(f"정보 밀도: {evaluation_result['density_analysis']['average_density']:.2f}")
```

텍스트 전처리와 청킹은 RAG 시스템의 성능을 크게 좌우하는 핵심 과정입니다. 도메인과 데이터 특성에 맞는 적절한 전처리와 청킹 전략을 선택하여 최적의 결과를 얻을 수 있습니다.
