# 임베딩(Embedding) 완전 가이드

## 1. 임베딩이란 무엇인가?

**임베딩(Embedding)**은 텍스트, 이미지, 음성 등의 비구조화 데이터를 **고정된 크기의 벡터로 변환하는 기술**입니다. 이 벡터는 원본 데이터의 **의미적 정보를 수치로 압축**하여 표현합니다.

### 임베딩의 수학적 원리

**차원 축소와 정보 보존**

임베딩의 핵심은 **고차원 데이터를 저차원 공간으로 매핑하면서 중요한 정보를 보존**하는 것입니다. 이는 정보 이론의 차원 축소 원리에 기반합니다.

```python
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

# 고차원 데이터에서 의미 구조 보존 예시
def demonstrate_embedding_principle():
    # 10차원 샘플 데이터 (세 개의 군집)
    np.random.seed(42)

    # 군집 1: 과학 관련
    science_docs = np.random.multivariate_normal([2, 3, 1, 0, 0, 0, 0, 0, 0, 0],
                                               np.eye(10) * 0.5, 50)

    # 군집 2: 예술 관련
    art_docs = np.random.multivariate_normal([0, 0, 0, 2, 3, 1, 0, 0, 0, 0],
                                           np.eye(10) * 0.5, 50)

    # 군집 3: 스포츠 관련
    sports_docs = np.random.multivariate_normal([0, 0, 0, 0, 0, 0, 2, 3, 1, 0],
                                              np.eye(10) * 0.5, 50)

    # 전체 데이터
    all_docs = np.vstack([science_docs, art_docs, sports_docs])
    labels = ['Science']*50 + ['Art']*50 + ['Sports']*50

    # 2차원으로 차원 축소 (임베딩 시뮬레이션)
    tsne = TSNE(n_components=2, random_state=42)
    embedded_2d = tsne.fit_transform(all_docs)

    # 시각화
    colors = ['red', 'blue', 'green']
    for i, label in enumerate(['Science', 'Art', 'Sports']):
        mask = np.array(labels) == label
        plt.scatter(embedded_2d[mask, 0], embedded_2d[mask, 1],
                   c=colors[i], label=label, alpha=0.7)

    plt.title('고차원 데이터의 2D 임베딩 결과')
    plt.xlabel('Dimension 1')
    plt.ylabel('Dimension 2')
    plt.legend()
    plt.show()

    # 중요한 관찰: 유사한 내용의 문서들이 가까운 곳에 위치
    return embedded_2d, labels

# demonstrate_embedding_principle()
```

**주요 수학적 개념:**

1. **매니폴드 학습 (Manifold Learning)**

   - 고차원 데이터가 실제로는 저차원 매니폴드 상에 존재
   - 임베딩은 이 숨겨진 구조를 발견하는 과정

2. **문서-단어 행렬 분해**

   - SVD, NMF 등을 통한 차원 축소
   - 잠재 주제(Latent Topic) 발견

3. **코사인 유사도와 내적 (Dot Product)**

   - 두 벡터 사이의 각도를 통한 유사성 측정
   - 정규화된 벡터에서 cosine similarity = dot product

4. **분포 어정 가설 (Distributional Hypothesis)**
   - "비슷한 맥락에서 나타나는 단어들은 비슷한 의미"
   - Word2Vec, GloVe의 이론적 기초

### 1.1 임베딩의 기본 개념

```python
# 임베딩 예시
text = "Python은 프로그래밍 언어입니다"
embedding_vector = [0.23, -0.41, 0.87, 0.12, ..., 0.56]  # 384차원 벡터

# 유사한 의미의 텍스트는 유사한 벡터를 가짐
text1 = "Python은 프로그래밍 언어다"      # 벡터1
text2 = "파이썬은 코딩 언어이다"          # 벡터2 (벡터1과 유사)
text3 = "사과는 빨간 과일이다"           # 벡터3 (벡터1,2와 상이)
```

### 1.2 왜 임베딩이 필요한가?

#### 컴퓨터의 한계

- 컴퓨터는 텍스트를 **직접 이해할 수 없음**
- 수치 데이터만 처리 가능
- 단순한 원-핫 인코딩은 의미 정보 손실

#### 임베딩의 장점

```python
# 원-핫 인코딩 (의미 정보 없음)
vocab = ["사과", "바나나", "컴퓨터", "프로그래밍"]
apple_onehot = [1, 0, 0, 0]
banana_onehot = [0, 1, 0, 0]
computer_onehot = [0, 0, 1, 0]

# 임베딩 (의미 정보 포함)
apple_embedding = [0.8, 0.2, 0.1, 0.0]      # 과일 관련 높은 값
banana_embedding = [0.7, 0.3, 0.1, 0.0]     # 사과와 유사
computer_embedding = [0.1, 0.0, 0.8, 0.9]   # 기술 관련 높은 값
```

## 2. 임베딩의 종류와 발전사

### 임베딩 기술의 혁신적 발전

임베딩 기술은 **통계적 방법 → 신경망 기반 → 트랜스포머 기반**으로 진화해왔으며, 각 단계마다 의미 표현 능력이 크게 향상되었습니다.

**발전 단계별 핵심 혁신:**

1. **1세대 (2000년대)**: 통계적 공기동(Co-occurrence) 기반

   - LSA, TF-IDF: 문서-단어 행렬의 차원 축소
   - 한계: 단어 순서 무시, 의미적 관계 부족

2. **2세대 (2010년대 초)**: 신경망 언어 모델

   - Word2Vec, GloVe: 분산 표현(Distributed Representation)
   - 혁신: "King - Man + Woman = Queen" 같은 의미 연산 가능

3. **3세대 (2010년대 후반)**: 문맥 인식 임베딩

   - ELMo, BERT: 같은 단어도 문맥에 따라 다른 벡터
   - 혁신: 다의성 문제 해결, 양방향 문맥 이해

4. **4세대 (2020년대)**: 대규모 멀티모달 임베딩
   - GPT, CLIP: 텍스트-이미지 통합 임베딩
   - 혁신: 모달리티 간 의미 공간 통합

### 2.1 초기 임베딩 기법

#### 정보 이론 기반 접근법

초기 임베딩은 **정보 이론**과 **선형 대수**에 기반했습니다. 핵심 아이디어는 고차원의 희소한(Sparse) 표현을 저차원의 밀집한(Dense) 표현으로 변환하는 것이었습니다.

#### Bag of Words (BoW) - 기초적 접근

```python
from collections import Counter
import numpy as np

class BagOfWordsEmbedding:
    """가장 기본적인 텍스트 벡터화 방법"""

    def __init__(self):
        self.vocab = {}
        self.vocab_size = 0

    def fit(self, documents):
        """어휘 사전 구축"""
        all_words = set()
        for doc in documents:
            all_words.update(doc.lower().split())

        self.vocab = {word: idx for idx, word in enumerate(sorted(all_words))}
        self.vocab_size = len(self.vocab)

        print(f"어휘 사전 크기: {self.vocab_size}")
        print(f"샘플 어휘: {list(self.vocab.keys())[:10]}")

    def transform(self, document):
        """문서를 BoW 벡터로 변환"""
        vector = np.zeros(self.vocab_size)
        words = document.lower().split()
        word_count = Counter(words)

        for word, count in word_count.items():
            if word in self.vocab:
                vector[self.vocab[word]] = count

        return vector

    def fit_transform(self, documents):
        """학습과 변환을 동시에 수행"""
        self.fit(documents)
        return np.array([self.transform(doc) for doc in documents])

# BoW의 한계 시연
documents = [
    "Python is a programming language",
    "Java is also a programming language",
    "I love Python programming",
    "Machine learning uses Python"
]

bow = BagOfWordsEmbedding()
vectors = bow.fit_transform(documents)

print("\\nBoW 벡터 예시:")
for i, doc in enumerate(documents):
    print(f"문서 {i+1}: {doc}")
    print(f"벡터: {vectors[i][:10]}...")  # 처음 10개 차원만 출력
    print()

# BoW의 문제점 분석
def analyze_bow_limitations():
    # 1. 단어 순서 무시
    sent1 = "Dog bites man"
    sent2 = "Man bites dog"

    vec1 = bow.transform(sent1)
    vec2 = bow.transform(sent2)

    # 두 문장이 완전히 다른 의미임에도 벡터가 동일
    print(f"'{sent1}' 벡터: {vec1[:5]}")
    print(f"'{sent2}' 벡터: {vec2[:5]}")
    print(f"벡터 동일성: {np.array_equal(vec1, vec2)}")

    # 2. 희소성 문제
    sparsity = np.sum(vectors == 0) / vectors.size
    print(f"\\n희소성 비율: {sparsity:.2%}")

    # 3. 의미적 관계 부족
    similar_docs = ["Python programming", "Java coding"]
    vec_py = bow.transform(similar_docs[0])
    vec_java = bow.transform(similar_docs[1])

    # 코사인 유사도 계산
    cosine_sim = np.dot(vec_py, vec_java) / (np.linalg.norm(vec_py) * np.linalg.norm(vec_java))
    print(f"유사한 의미 문서들의 유사도: {cosine_sim:.3f}")  # 매우 낮음

# analyze_bow_limitations()
```

#### TF-IDF - 통계적 가중치 도입

TF-IDF는 **단어의 중요도를 통계적으로 측정**하여 BoW의 한계를 부분적으로 해결합니다.

```python
import math
from collections import Counter, defaultdict

class TFIDFEmbedding:
    """TF-IDF 기반 문서 임베딩"""

    def __init__(self):
        self.vocab = {}
        self.idf_scores = {}
        self.document_count = 0

    def fit(self, documents):
        """TF-IDF 가중치 학습"""
        self.document_count = len(documents)

        # 1. 어휘 사전 구축
        all_words = set()
        for doc in documents:
            all_words.update(doc.lower().split())
        self.vocab = {word: idx for idx, word in enumerate(sorted(all_words))}

        # 2. IDF 점수 계산
        doc_frequency = defaultdict(int)
        for doc in documents:
            unique_words = set(doc.lower().split())
            for word in unique_words:
                doc_frequency[word] += 1

        # IDF = log(전체 문서 수 / 단어가 등장하는 문서 수)
        for word in self.vocab:
            df = doc_frequency[word]
            self.idf_scores[word] = math.log(self.document_count / df) if df > 0 else 0

        # IDF 점수 분석
        print("\\n=== IDF 점수 분석 ===")
        sorted_idf = sorted(self.idf_scores.items(), key=lambda x: x[1], reverse=True)
        print("높은 IDF (희귀한 단어):", sorted_idf[:5])
        print("낮은 IDF (흔한 단어):", sorted_idf[-5:])

    def transform(self, document):
        """문서를 TF-IDF 벡터로 변환"""
        words = document.lower().split()
        word_count = Counter(words)
        doc_length = len(words)

        vector = np.zeros(len(self.vocab))

        for word, count in word_count.items():
            if word in self.vocab:
                # TF = (단어 빈도) / (문서 내 총 단어 수)
                tf = count / doc_length
                # IDF = 미리 계산된 값
                idf = self.idf_scores[word]
                # TF-IDF = TF * IDF
                tfidf = tf * idf

                vector[self.vocab[word]] = tfidf

        return vector

    def get_top_terms(self, document, top_k=5):
        """문서의 가장 중요한 단어들 반환"""
        vector = self.transform(document)

        # 단어별 TF-IDF 점수
        word_scores = []
        for word, idx in self.vocab.items():
            if vector[idx] > 0:
                word_scores.append((word, vector[idx]))

        word_scores.sort(key=lambda x: x[1], reverse=True)
        return word_scores[:top_k]

# TF-IDF 개선 효과 시연
documents = [
    "Python is a great programming language for beginners",
    "Java is also a popular programming language",
    "Machine learning algorithms use Python extensively",
    "Deep learning frameworks like TensorFlow use Python",
    "The cat sat on the mat",  # 다른 도메인 문서
    "A quick brown fox jumps over the lazy dog"  # 다른 도메인 문서
]

tfidf = TFIDFEmbedding()
tfidf.fit(documents)

print("\\n=== TF-IDF 중요 단어 분석 ===")
for i, doc in enumerate(documents[:3]):  # 처음 3개 문서만
    print(f"\\n문서 {i+1}: {doc}")
    top_terms = tfidf.get_top_terms(doc, top_k=3)
    print(f"중요 단어: {top_terms}")
```

#### LSA (Latent Semantic Analysis) - 차원 축소의 시작

LSA는 **특이값 분해(SVD)**를 사용해 TF-IDF 행렬을 저차원으로 축소하여 **잠재적 의미 구조**를 발견합니다.

```python
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.pipeline import Pipeline
import numpy as np

class LSAEmbedding:
    """LSA (Latent Semantic Analysis) 임베딩"""

    def __init__(self, n_components=100, max_features=10000):
        self.n_components = n_components
        self.pipeline = Pipeline([
            ('tfidf', TfidfVectorizer(max_features=max_features, stop_words='english')),
            ('svd', TruncatedSVD(n_components=n_components, random_state=42))
        ])

    def fit_transform(self, documents):
        """문서들을 LSA 벡터로 변환"""
        return self.pipeline.fit_transform(documents)

    def get_topics(self, feature_names, top_k=10):
        """주요 토픽 추출"""
        svd = self.pipeline.named_steps['svd']

        topics = []
        for i, component in enumerate(svd.components_):
            # 각 컴포넌트에서 가장 중요한 단어들
            top_indices = component.argsort()[-top_k:][::-1]
            top_words = [feature_names[idx] for idx in top_indices]
            top_weights = component[top_indices]

            topics.append(list(zip(top_words, top_weights)))

        return topics

# LSA의 의미 발견 능력 시연
tech_documents = [
    "Python programming language syntax is clean and readable",
    "Java object oriented programming concepts are powerful",
    "Machine learning algorithms require statistical knowledge",
    "Deep neural networks learn complex patterns in data",
    "Natural language processing analyzes human communication",
    "Computer vision processes and interprets visual information",
    "Database systems store and retrieve structured data efficiently",
    "Web development frameworks simplify application building",
    "Mobile app development targets smartphone platforms",
    "Cloud computing provides scalable infrastructure services"
]

# LSA 학습
lsa = LSAEmbedding(n_components=5)
lsa_vectors = lsa.fit_transform(tech_documents)

# TF-IDF 특성 이름 가져오기
tfidf_vectorizer = lsa.pipeline.named_steps['tfidf']
feature_names = tfidf_vectorizer.get_feature_names_out()

# 발견된 토픽 분석
topics = lsa.get_topics(feature_names, top_k=5)

print("\\n=== LSA로 발견된 잠재 토픽 ===")
topic_names = ["프로그래밍 언어", "머신러닝/AI", "웹/모바일 개발", "데이터 시스템", "기타"]
for i, (topic_name, topic_words) in enumerate(zip(topic_names[:len(topics)], topics)):
    print(f"\\n토픽 {i+1} ({topic_name}):")
    for word, weight in topic_words:
        print(f"  {word}: {weight:.3f}")

# 문서 간 의미적 유사성 비교
print("\\n=== 문서 간 의미적 유사성 (LSA) ===")
similar_pairs = [
    (0, 1),  # Python vs Java (프로그래밍 언어)
    (2, 3),  # ML vs Deep Learning (AI 관련)
    (7, 8),  # Web vs Mobile (개발 관련)
]

for doc1_idx, doc2_idx in similar_pairs:
    vec1, vec2 = lsa_vectors[doc1_idx], lsa_vectors[doc2_idx]
    similarity = np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

    print(f"문서 {doc1_idx+1} vs 문서 {doc2_idx+1}:")
    print(f"  '{tech_documents[doc1_idx][:40]}...'")
    print(f"  '{tech_documents[doc2_idx][:40]}...'")
    print(f"  유사도: {similarity:.3f}\\n")
```

**초기 임베딩 기법들의 혁신과 한계:**

**혁신점:**

- **차원 축소**: 고차원 희소 벡터를 저차원 밀집 벡터로 변환
- **의미 발견**: 단어 동시 출현 패턴에서 잠재적 의미 구조 추출
- **계산 효율성**: 선형 대수 연산으로 대규모 처리 가능

**한계점:**

- **정적 표현**: 하나의 단어는 항상 같은 벡터 (문맥 무시)
- **단어 순서**: 문법적 구조와 어순 정보 손실
- **의미 연산 한계**: 복잡한 의미 관계 표현 어려움

### 2.2 신경망 기반 임베딩 - 혁명의 시작

        vector = [0] * len(self.vocab)
        for word, count in word_count.items():
            if word in self.vocab:
                tf = count / len(words)  # Term Frequency
                idf = self.idf_scores[word]  # Inverse Document Frequency
                vector[self.vocab[word]] = tf * idf

        return vector

# 사용 예시

tfidf = TFIDFEmbedding()
documents = [
"python programming language",
"java programming language",
"machine learning python"
]
tfidf.fit(documents)
vector = tfidf.transform("python programming")

````

### 2.2 Word Embeddings

#### Word2Vec (2013)
```python
# Word2Vec 개념 구현
class Word2Vec:
    def __init__(self, vector_size=100, window=5, min_count=1):
        self.vector_size = vector_size
        self.window = window
        self.min_count = min_count
        self.vocab = {}
        self.word_vectors = {}

    def build_vocab(self, sentences):
        """어휘 사전 구축"""
        word_freq = Counter()
        for sentence in sentences:
            words = sentence.split()
            word_freq.update(words)

        # 최소 빈도 이상의 단어만 포함
        self.vocab = {
            word: i for i, (word, freq) in enumerate(word_freq.items())
            if freq >= self.min_count
        }

    def train_skipgram(self, sentences, epochs=5):
        """Skip-gram 모델 학습 (개념적 구현)"""
        import numpy as np

        # 가중치 초기화
        vocab_size = len(self.vocab)
        W1 = np.random.randn(vocab_size, self.vector_size) * 0.1
        W2 = np.random.randn(self.vector_size, vocab_size) * 0.1

        for epoch in range(epochs):
            for sentence in sentences:
                words = sentence.split()
                for i, center_word in enumerate(words):
                    if center_word not in self.vocab:
                        continue

                    # 컨텍스트 윈도우 내 단어들
                    for j in range(max(0, i-self.window),
                                 min(len(words), i+self.window+1)):
                        if i != j and words[j] in self.vocab:
                            context_word = words[j]
                            # 여기서 실제 학습 (gradient descent) 수행
                            pass

        # 학습된 가중치를 단어 벡터로 저장
        for word, idx in self.vocab.items():
            self.word_vectors[word] = W1[idx]

    def get_vector(self, word):
        """단어의 벡터 반환"""
        return self.word_vectors.get(word, None)

    def most_similar(self, word, top_k=10):
        """가장 유사한 단어들 반환"""
        if word not in self.word_vectors:
            return []

        word_vector = self.word_vectors[word]
        similarities = []

        for other_word, other_vector in self.word_vectors.items():
            if word != other_word:
                similarity = cosine_similarity(word_vector, other_vector)
                similarities.append((other_word, similarity))

        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:top_k]

# 실제 사용 (gensim 라이브러리)
from gensim.models import Word2Vec

sentences = [
    ["python", "programming", "language"],
    ["java", "programming", "language"],
    ["machine", "learning", "python"]
]

model = Word2Vec(sentences, vector_size=100, window=5, min_count=1, workers=4)

# 단어 벡터 얻기
python_vector = model.wv['python']

# 유사한 단어 찾기
similar_words = model.wv.most_similar('python', topn=5)
````

#### GloVe (Global Vectors for Word Representation)

```python
class GloVe:
    def __init__(self, vector_size=100, learning_rate=0.05, x_max=100, alpha=0.75):
        self.vector_size = vector_size
        self.learning_rate = learning_rate
        self.x_max = x_max
        self.alpha = alpha

    def build_cooccurrence_matrix(self, corpus, window_size=5):
        """동시 출현 행렬 구축"""
        vocab = set()
        for sentence in corpus:
            vocab.update(sentence.split())

        vocab_to_idx = {word: i for i, word in enumerate(vocab)}
        cooccurrence = defaultdict(lambda: defaultdict(int))

        for sentence in corpus:
            words = sentence.split()
            for i, center_word in enumerate(words):
                for j in range(max(0, i-window_size),
                              min(len(words), i+window_size+1)):
                    if i != j:
                        context_word = words[j]
                        distance = abs(i - j)
                        cooccurrence[center_word][context_word] += 1.0 / distance

        return cooccurrence, vocab_to_idx

    def weighting_function(self, x):
        """GloVe 가중치 함수"""
        if x < self.x_max:
            return (x / self.x_max) ** self.alpha
        else:
            return 1.0

    def train(self, corpus, epochs=50):
        """GloVe 모델 학습"""
        import numpy as np

        cooccurrence, vocab_to_idx = self.build_cooccurrence_matrix(corpus)
        vocab_size = len(vocab_to_idx)

        # 매개변수 초기화
        W = np.random.randn(vocab_size, self.vector_size) * 0.1
        W_tilde = np.random.randn(vocab_size, self.vector_size) * 0.1
        b = np.random.randn(vocab_size) * 0.1
        b_tilde = np.random.randn(vocab_size) * 0.1

        for epoch in range(epochs):
            cost = 0
            for word_i, contexts in cooccurrence.items():
                i = vocab_to_idx[word_i]
                for word_j, x_ij in contexts.items():
                    j = vocab_to_idx[word_j]

                    weight = self.weighting_function(x_ij)
                    cost_inner = (W[i].dot(W_tilde[j]) + b[i] + b_tilde[j] - np.log(x_ij)) ** 2
                    cost += weight * cost_inner

                    # 기울기 계산 및 매개변수 업데이트 (생략)

        # 최종 단어 벡터
        self.word_vectors = W + W_tilde
        self.vocab_to_idx = vocab_to_idx
```

### 2.3 최신 임베딩 기법

#### BERT Embeddings

```python
from transformers import BertTokenizer, BertModel
import torch

class BERTEmbedding:
    def __init__(self, model_name='bert-base-multilingual-cased'):
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.model = BertModel.from_pretrained(model_name)
        self.model.eval()

    def encode(self, texts, pooling='cls'):
        """텍스트를 BERT 임베딩으로 변환"""
        if isinstance(texts, str):
            texts = [texts]

        # 토크나이징
        encoded = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors='pt'
        )

        with torch.no_grad():
            outputs = self.model(**encoded)

            if pooling == 'cls':
                # [CLS] 토큰 사용
                embeddings = outputs.last_hidden_state[:, 0, :]
            elif pooling == 'mean':
                # 평균 풀링
                attention_mask = encoded['attention_mask']
                embeddings = (outputs.last_hidden_state * attention_mask.unsqueeze(-1)).mean(dim=1)

            return embeddings.numpy()

# 사용 예시
bert_embedding = BERTEmbedding()
texts = [
    "Python은 프로그래밍 언어입니다",
    "파이썬은 코딩에 사용되는 언어입니다"
]
embeddings = bert_embedding.encode(texts)
print(f"Embedding shape: {embeddings.shape}")  # (2, 768)
```

#### Sentence Transformers

```python
from sentence_transformers import SentenceTransformer

class SentenceEmbedding:
    def __init__(self, model_name='distiluse-base-multilingual-cased'):
        self.model = SentenceTransformer(model_name)

    def encode(self, texts):
        """문장을 임베딩으로 변환"""
        return self.model.encode(texts)

    def semantic_similarity(self, text1, text2):
        """두 텍스트 간 의미적 유사도"""
        embeddings = self.model.encode([text1, text2])
        similarity = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]
        return similarity

# 사용 예시
sentence_model = SentenceEmbedding()

texts = [
    "Python은 프로그래밍 언어입니다",
    "자바는 객체지향 언어입니다",
    "오늘 날씨가 좋습니다"
]

embeddings = sentence_model.encode(texts)
print(f"Embeddings shape: {embeddings.shape}")

# 유사도 계산
similarity = sentence_model.semantic_similarity(texts[0], texts[1])
print(f"Similarity: {similarity}")  # 높은 유사도
```

#### OpenAI Embeddings

```python
import openai
import numpy as np

class OpenAIEmbedding:
    def __init__(self, api_key, model="text-embedding-ada-002"):
        openai.api_key = api_key
        self.model = model

    def encode(self, texts, batch_size=100):
        """OpenAI API를 사용한 임베딩 생성"""
        if isinstance(texts, str):
            texts = [texts]

        embeddings = []

        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]

            response = openai.Embedding.create(
                input=batch,
                model=self.model
            )

            batch_embeddings = [item['embedding'] for item in response['data']]
            embeddings.extend(batch_embeddings)

        return np.array(embeddings)

    def get_embedding_dimension(self):
        """임베딩 차원 반환"""
        if self.model == "text-embedding-ada-002":
            return 1536
        elif self.model == "text-embedding-3-small":
            return 1536
        elif self.model == "text-embedding-3-large":
            return 3072
        else:
            return 1536  # 기본값

# 사용 예시
openai_embedding = OpenAIEmbedding(api_key="your-api-key")

texts = [
    "Python 프로그래밍 언어 학습",
    "머신러닝과 딥러닝 기초",
    "웹 개발 프레임워크"
]

embeddings = openai_embedding.encode(texts)
print(f"Embeddings shape: {embeddings.shape}")  # (3, 1536)
```

## 3. 임베딩 품질 평가

### 3.1 내재적 평가 (Intrinsic Evaluation)

#### 단어 유사도 테스트

```python
class WordSimilarityEvaluator:
    def __init__(self, embedding_model):
        self.model = embedding_model

    def evaluate_word_similarity(self, word_pairs_with_scores):
        """
        단어 쌍과 인간이 평가한 유사도 점수를 사용한 평가

        word_pairs_with_scores: [("dog", "cat", 8.5), ("car", "tree", 1.2), ...]
        """
        predicted_similarities = []
        human_similarities = []

        for word1, word2, human_score in word_pairs_with_scores:
            if word1 in self.model and word2 in self.model:
                vec1 = self.model[word1]
                vec2 = self.model[word2]

                predicted_sim = cosine_similarity([vec1], [vec2])[0][0]
                predicted_similarities.append(predicted_sim)
                human_similarities.append(human_score / 10.0)  # 0-1 스케일로 정규화

        # 스피어만 상관계수 계산
        from scipy.stats import spearmanr
        correlation, p_value = spearmanr(predicted_similarities, human_similarities)

        return {
            "spearman_correlation": correlation,
            "p_value": p_value,
            "num_pairs": len(predicted_similarities)
        }

# 사용 예시
word_pairs = [
    ("개", "고양이", 7.5),
    ("자동차", "나무", 1.0),
    ("컴퓨터", "프로그래밍", 8.8),
    ("사과", "바나나", 6.2)
]

evaluator = WordSimilarityEvaluator(word2vec_model)
results = evaluator.evaluate_word_similarity(word_pairs)
```

#### 단어 유추 테스트

```python
class WordAnalogyEvaluator:
    def __init__(self, embedding_model):
        self.model = embedding_model

    def evaluate_analogy(self, analogies):
        """
        단어 유추 평가: A:B = C:D
        예: 왕:남자 = 여왕:?  (정답: 여자)
        """
        correct = 0
        total = 0

        for a, b, c, expected_d in analogies:
            if all(word in self.model for word in [a, b, c, expected_d]):
                # 벡터 연산: D = B - A + C
                vec_a = self.model[a]
                vec_b = self.model[b]
                vec_c = self.model[c]

                target_vector = vec_b - vec_a + vec_c

                # 가장 유사한 단어 찾기 (A, B, C 제외)
                similarities = []
                for word, vector in self.model.items():
                    if word not in [a, b, c]:
                        sim = cosine_similarity([target_vector], [vector])[0][0]
                        similarities.append((word, sim))

                similarities.sort(key=lambda x: x[1], reverse=True)
                predicted_d = similarities[0][0]

                if predicted_d == expected_d:
                    correct += 1

                total += 1

        accuracy = correct / total if total > 0 else 0
        return {
            "accuracy": accuracy,
            "correct": correct,
            "total": total
        }

# 사용 예시 (영어)
analogies = [
    ("king", "man", "queen", "woman"),
    ("paris", "france", "london", "england"),
    ("walk", "walked", "go", "went")
]

analogy_evaluator = WordAnalogyEvaluator(word2vec_model)
results = analogy_evaluator.evaluate_analogy(analogies)
```

### 3.2 외재적 평가 (Extrinsic Evaluation)

#### 텍스트 분류 성능

```python
class TextClassificationEvaluator:
    def __init__(self, embedding_model):
        self.embedding_model = embedding_model

    def evaluate_on_classification(self, texts, labels, test_size=0.2):
        """텍스트 분류 태스크에서 임베딩 성능 평가"""
        from sklearn.model_selection import train_test_split
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import accuracy_score, f1_score

        # 텍스트를 임베딩으로 변환
        embeddings = []
        for text in texts:
            embedding = self.embedding_model.encode(text)
            embeddings.append(embedding)

        X = np.array(embeddings)
        y = np.array(labels)

        # 훈련/테스트 분할
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42
        )

        # 로지스틱 회귀 학습
        classifier = LogisticRegression(random_state=42)
        classifier.fit(X_train, y_train)

        # 예측 및 평가
        y_pred = classifier.predict(X_test)

        return {
            "accuracy": accuracy_score(y_test, y_pred),
            "f1_score": f1_score(y_test, y_pred, average='weighted'),
            "test_samples": len(y_test)
        }

# 사용 예시
texts = [
    "이 영화 정말 재미있어요!",
    "최악의 영화였습니다.",
    "그럭저럭 볼만한 영화네요.",
    "완전히 시간 낭비였어요."
]
labels = [1, 0, 1, 0]  # 1: 긍정, 0: 부정

evaluator = TextClassificationEvaluator(sentence_transformer_model)
results = evaluator.evaluate_on_classification(texts, labels)
```

## 4. 임베딩 최적화 기법

### 4.1 차원 축소

#### PCA (Principal Component Analysis)

```python
from sklearn.decomposition import PCA
import numpy as np

class EmbeddingDimensionReducer:
    def __init__(self, target_dimension=128):
        self.target_dimension = target_dimension
        self.pca = None

    def fit_pca(self, embeddings):
        """PCA로 차원 축소 모델 학습"""
        self.pca = PCA(n_components=self.target_dimension)
        self.pca.fit(embeddings)

        # 설명된 분산 비율 출력
        explained_variance_ratio = np.sum(self.pca.explained_variance_ratio_)
        print(f"설명된 분산 비율: {explained_variance_ratio:.3f}")

        return self.pca

    def transform(self, embeddings):
        """임베딩 차원 축소"""
        if self.pca is None:
            raise ValueError("PCA 모델이 학습되지 않았습니다.")

        return self.pca.transform(embeddings)

    def fit_transform(self, embeddings):
        """학습과 변환을 동시에 수행"""
        self.fit_pca(embeddings)
        return self.transform(embeddings)

# 사용 예시
original_embeddings = np.random.randn(1000, 768)  # BERT 임베딩

reducer = EmbeddingDimensionReducer(target_dimension=256)
reduced_embeddings = reducer.fit_transform(original_embeddings)

print(f"원본 차원: {original_embeddings.shape}")
print(f"축소된 차원: {reduced_embeddings.shape}")
```

#### t-SNE 시각화

```python
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

class EmbeddingVisualizer:
    def __init__(self):
        pass

    def visualize_tsne(self, embeddings, labels=None, title="Embedding Visualization"):
        """t-SNE를 사용한 임베딩 시각화"""
        # 2차원으로 축소
        tsne = TSNE(n_components=2, random_state=42, perplexity=30)
        embeddings_2d = tsne.fit_transform(embeddings)

        plt.figure(figsize=(12, 8))

        if labels is not None:
            unique_labels = np.unique(labels)
            colors = plt.cm.Set3(np.linspace(0, 1, len(unique_labels)))

            for label, color in zip(unique_labels, colors):
                mask = labels == label
                plt.scatter(embeddings_2d[mask, 0], embeddings_2d[mask, 1],
                          c=[color], label=str(label), alpha=0.7)
            plt.legend()
        else:
            plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], alpha=0.7)

        plt.title(title)
        plt.xlabel('t-SNE 1')
        plt.ylabel('t-SNE 2')
        plt.grid(True, alpha=0.3)
        plt.show()

        return embeddings_2d

# 사용 예시
visualizer = EmbeddingVisualizer()
embeddings_2d = visualizer.visualize_tsne(
    embeddings=sentence_embeddings,
    labels=category_labels,
    title="문장 임베딩 시각화"
)
```

### 4.2 임베딩 후처리

#### Mean Centering & Whitening

```python
class EmbeddingPostProcessor:
    def __init__(self):
        self.mean = None
        self.whitening_matrix = None

    def fit(self, embeddings):
        """임베딩 통계 계산"""
        self.mean = np.mean(embeddings, axis=0)

        # 공분산 행렬 계산
        centered_embeddings = embeddings - self.mean
        covariance_matrix = np.cov(centered_embeddings.T)

        # 화이트닝 행렬 계산 (SVD 사용)
        U, s, Vt = np.linalg.svd(covariance_matrix)
        self.whitening_matrix = U @ np.diag(1.0 / np.sqrt(s + 1e-8)) @ U.T

    def transform(self, embeddings):
        """임베딩 후처리 적용"""
        # Mean centering
        centered_embeddings = embeddings - self.mean

        # Whitening (선택적)
        if self.whitening_matrix is not None:
            whitened_embeddings = centered_embeddings @ self.whitening_matrix.T
            return whitened_embeddings
        else:
            return centered_embeddings

    def fit_transform(self, embeddings):
        """학습과 변환을 동시에 수행"""
        self.fit(embeddings)
        return self.transform(embeddings)

# 사용 예시
processor = EmbeddingPostProcessor()
processed_embeddings = processor.fit_transform(original_embeddings)

# 성능 비교
original_sim = cosine_similarity(original_embeddings[:10])
processed_sim = cosine_similarity(processed_embeddings[:10])

print("후처리 전후 유사도 분포 변화:")
print(f"원본 평균 유사도: {np.mean(original_sim):.3f}")
print(f"후처리된 평균 유사도: {np.mean(processed_sim):.3f}")
```

### 4.3 다국어 임베딩 정렬

#### Procrustes Analysis

```python
class CrossLingualEmbeddingAligner:
    def __init__(self):
        self.transformation_matrix = None

    def align_embeddings(self, source_embeddings, target_embeddings):
        """
        Procrustes Analysis를 사용한 임베딩 공간 정렬

        source_embeddings: 소스 언어 임베딩
        target_embeddings: 타겟 언어 임베딩
        """
        # 중심화
        source_centered = source_embeddings - np.mean(source_embeddings, axis=0)
        target_centered = target_embeddings - np.mean(target_embeddings, axis=0)

        # SVD를 사용한 최적 변환 행렬 계산
        H = source_centered.T @ target_centered
        U, s, Vt = np.linalg.svd(H)

        self.transformation_matrix = U @ Vt

        return source_centered @ self.transformation_matrix

    def transform(self, embeddings):
        """새로운 임베딩에 변환 적용"""
        if self.transformation_matrix is None:
            raise ValueError("변환 행렬이 계산되지 않았습니다.")

        centered = embeddings - np.mean(embeddings, axis=0)
        return centered @ self.transformation_matrix

# 사용 예시
# 영어-한국어 병렬 단어 쌍
english_words = ["king", "queen", "man", "woman", "cat", "dog"]
korean_words = ["왕", "여왕", "남자", "여자", "고양이", "개"]

# 각 언어의 임베딩 얻기
english_embeddings = get_english_embeddings(english_words)
korean_embeddings = get_korean_embeddings(korean_words)

# 임베딩 공간 정렬
aligner = CrossLingualEmbeddingAligner()
aligned_english = aligner.align_embeddings(english_embeddings, korean_embeddings)

# 정렬된 공간에서 유사도 계산
similarity = cosine_similarity(aligned_english, korean_embeddings)
print("정렬 후 대응 단어 유사도:", np.diag(similarity))
```

## 5. 실제 프로덕션 사용 사례

### 5.1 대규모 임베딩 파이프라인

```python
import asyncio
import aiofiles
from concurrent.futures import ThreadPoolExecutor
import numpy as np

class ProductionEmbeddingPipeline:
    def __init__(self, embedding_model, batch_size=1000, max_workers=4):
        self.embedding_model = embedding_model
        self.batch_size = batch_size
        self.max_workers = max_workers
        self.executor = ThreadPoolExecutor(max_workers=max_workers)

    async def process_large_dataset(self, texts, output_file):
        """대용량 텍스트 데이터셋의 임베딩 생성"""
        total_batches = (len(texts) + self.batch_size - 1) // self.batch_size

        async with aiofiles.open(output_file, 'wb') as f:
            for i in range(0, len(texts), self.batch_size):
                batch_texts = texts[i:i + self.batch_size]

                # 배치 임베딩 생성
                loop = asyncio.get_event_loop()
                embeddings = await loop.run_in_executor(
                    self.executor,
                    self.embedding_model.encode,
                    batch_texts
                )

                # 넘파이 배열로 저장
                await f.write(embeddings.tobytes())

                # 진행률 출력
                batch_num = i // self.batch_size + 1
                print(f"배치 {batch_num}/{total_batches} 완료")

        print(f"임베딩 저장 완료: {output_file}")

    async def load_embeddings_chunk(self, file_path, start_idx, chunk_size, embedding_dim):
        """임베딩 파일에서 청크 단위로 로드"""
        async with aiofiles.open(file_path, 'rb') as f:
            # 시작 위치로 이동
            await f.seek(start_idx * embedding_dim * 4)  # float32 = 4바이트

            # 청크 읽기
            chunk_bytes = chunk_size * embedding_dim * 4
            data = await f.read(chunk_bytes)

            # 넘파이 배열로 변환
            embeddings = np.frombuffer(data, dtype=np.float32)
            return embeddings.reshape(-1, embedding_dim)

    def create_embedding_index(self, embeddings_file, index_file, embedding_dim):
        """임베딩 인덱스 생성"""
        import faiss

        # FAISS 인덱스 초기화
        index = faiss.IndexFlatIP(embedding_dim)  # Inner Product

        # 배치 단위로 인덱스에 추가
        with open(embeddings_file, 'rb') as f:
            while True:
                # 배치 읽기
                batch_data = f.read(self.batch_size * embedding_dim * 4)
                if not batch_data:
                    break

                batch_embeddings = np.frombuffer(batch_data, dtype=np.float32)
                batch_embeddings = batch_embeddings.reshape(-1, embedding_dim)

                # 인덱스에 추가
                index.add(batch_embeddings)

        # 인덱스 저장
        faiss.write_index(index, index_file)
        print(f"인덱스 저장 완료: {index_file}")

        return index

# 사용 예시
async def main():
    # 대용량 텍스트 데이터 준비
    texts = load_large_text_dataset()  # 100만개 문서

    # 임베딩 파이프라인 생성
    pipeline = ProductionEmbeddingPipeline(
        embedding_model=sentence_transformer_model,
        batch_size=1000,
        max_workers=4
    )

    # 임베딩 생성 및 저장
    await pipeline.process_large_dataset(texts, "embeddings.npy")

    # 검색 인덱스 생성
    index = pipeline.create_embedding_index(
        "embeddings.npy",
        "embeddings.index",
        embedding_dim=384
    )

# 실행
asyncio.run(main())
```

### 5.2 실시간 임베딩 서비스

```python
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import redis
import pickle
import hashlib

app = FastAPI(title="Embedding Service")

class EmbeddingRequest(BaseModel):
    texts: list[str]
    cache_ttl: int = 3600

class EmbeddingResponse(BaseModel):
    embeddings: list[list[float]]
    cached: list[bool]

class EmbeddingService:
    def __init__(self, model_name="sentence-transformers/all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)
        self.redis_client = redis.Redis(host='localhost', port=6379, db=0)

    def _get_cache_key(self, text):
        """텍스트의 캐시 키 생성"""
        return f"emb:{hashlib.md5(text.encode()).hexdigest()}"

    def get_cached_embedding(self, text):
        """캐시에서 임베딩 조회"""
        cache_key = self._get_cache_key(text)
        cached_data = self.redis_client.get(cache_key)

        if cached_data:
            return pickle.loads(cached_data)
        return None

    def cache_embedding(self, text, embedding, ttl=3600):
        """임베딩을 캐시에 저장"""
        cache_key = self._get_cache_key(text)
        self.redis_client.setex(
            cache_key,
            ttl,
            pickle.dumps(embedding.tolist())
        )

    def encode_batch(self, texts, cache_ttl=3600):
        """배치 임베딩 생성 (캐싱 포함)"""
        embeddings = []
        cached_flags = []
        uncached_texts = []
        uncached_indices = []

        # 캐시 확인
        for i, text in enumerate(texts):
            cached_emb = self.get_cached_embedding(text)
            if cached_emb:
                embeddings.append(cached_emb)
                cached_flags.append(True)
            else:
                embeddings.append(None)
                cached_flags.append(False)
                uncached_texts.append(text)
                uncached_indices.append(i)

        # 캐시되지 않은 텍스트들 임베딩 생성
        if uncached_texts:
            new_embeddings = self.model.encode(uncached_texts)

            # 결과 배치 및 캐싱
            for idx, emb in zip(uncached_indices, new_embeddings):
                embeddings[idx] = emb.tolist()
                self.cache_embedding(texts[idx], emb, cache_ttl)

        return embeddings, cached_flags

# 전역 서비스 인스턴스
embedding_service = EmbeddingService()

@app.post("/embeddings", response_model=EmbeddingResponse)
async def create_embeddings(request: EmbeddingRequest):
    """임베딩 생성 엔드포인트"""
    try:
        embeddings, cached_flags = embedding_service.encode_batch(
            request.texts,
            request.cache_ttl
        )

        return EmbeddingResponse(
            embeddings=embeddings,
            cached=cached_flags
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    """헬스체크"""
    return {"status": "healthy"}

@app.get("/stats")
async def get_stats():
    """캐시 통계"""
    info = embedding_service.redis_client.info()
    return {
        "redis_memory_used": info.get("used_memory_human"),
        "redis_keyspace": info.get("db0", {}),
    }

# 서비스 실행
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
```

임베딩은 현대 AI 시스템의 핵심 구성요소로, RAG부터 추천 시스템까지 다양한 애플리케이션에서 활용됩니다. 적절한 임베딩 기법 선택과 최적화를 통해 AI 서비스의 성능을 크게 향상시킬 수 있습니다.

---

## 6. 벡터 유사도 측정의 심층 분석

### 6.1 유사도 측정의 수학적 기초

벡터 유사도 측정은 **두 벡터 간의 관계를 수치화하는 방법**입니다. 임베딩 공간에서 유사도는 **의미적 유사성**을 반영합니다.

#### 코사인 유사도 (Cosine Similarity)

**코사인 유사도**는 두 벡터 간의 **각도**를 측정합니다. 벡터의 크기(노름)는 무시하고 방향만 고려합니다.

```python
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

def cosine_similarity_detailed(vec1, vec2):
    """
    코사인 유사도의 상세 계산 과정
    """
    # 1. 벡터의 내적 계산
    dot_product = np.dot(vec1, vec2)

    # 2. 각 벡터의 노름(크기) 계산
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)

    # 3. 코사인 유사도 계산
    cosine_sim = dot_product / (norm1 * norm2)

    return cosine_sim, dot_product, norm1, norm2

# 예시 벡터들
vec_a = np.array([1, 2, 3])      # "고양이"
vec_b = np.array([1, 2, 4])      # "강아지" (유사한 의미)
vec_c = np.array([-1, -2, -3])   # "고양이"와 반대 방향
vec_d = np.array([0, 0, 0])      # 영벡터

print("=== 코사인 유사도 상세 분석 ===")
print(f"vec_a vs vec_b: {cosine_similarity_detailed(vec_a, vec_b)}")
print(f"vec_a vs vec_c: {cosine_similarity_detailed(vec_a, vec_c)}")
print(f"vec_a vs vec_d: {cosine_similarity_detailed(vec_a, vec_d)}")
```

**코사인 유사도의 특징:**

- **범위**: -1 (완전 반대) ~ 1 (완전 동일)
- **장점**: 벡터 크기에 영향을 받지 않음
- **단점**: 각도만 고려하므로 크기 정보 손실

#### 유클리드 거리 (Euclidean Distance)

**유클리드 거리**는 두 점 사이의 **직선 거리**를 측정합니다. 피타고라스 정리를 일반화한 개념입니다.

```python
def euclidean_distance(vec1, vec2):
    """
    유클리드 거리의 상세 계산
    """
    # 1. 차이 벡터 계산
    diff = vec1 - vec2

    # 2. 각 차원의 제곱합
    squared_sum = np.sum(diff ** 2)

    # 3. 제곱근 취하기
    distance = np.sqrt(squared_sum)

    return distance, diff, squared_sum

# 예시 계산
vec1 = np.array([1, 2, 3])
vec2 = np.array([4, 6, 8])

distance, diff, squared_sum = euclidean_distance(vec1, vec2)
print(f"유클리드 거리: {distance}")
print(f"차이 벡터: {diff}")
print(f"제곱합: {squared_sum}")
```

**유클리드 거리의 특징:**

- **범위**: 0 (완전 동일) ~ ∞ (완전 다름)
- **장점**: 직관적이고 계산이 간단함
- **단점**: 차원 수가 증가할수록 거리가 커지는 현상 (차원의 저주)

#### 맨하탄 거리 (Manhattan Distance)

**맨하탄 거리**는 **격자 도시의 거리 측정**처럼 각 차원의 절대값 합을 계산합니다.

```python
def manhattan_distance(vec1, vec2):
    """
    맨하탄 거리 계산 (L1 노름)
    """
    # 각 차원의 절대값 합
    distance = np.sum(np.abs(vec1 - vec2))

    return distance

# 예시: 도시 블록 거리
vec1 = np.array([0, 0])  # 출발점
vec2 = np.array([3, 4])  # 도착점

manhattan_dist = manhattan_distance(vec1, vec2)
euclidean_dist = euclidean_distance(vec1, vec2)[0]

print(f"맨하탄 거리: {manhattan_dist}")    # 7 (3+4)
print(f"유클리드 거리: {euclidean_dist:.2f}")  # 5.0 (직선 거리)
```

#### 내적 유사도 (Dot Product Similarity)

**내적 유사도**는 두 벡터의 **스칼라 곱**입니다. 코사인 유사도와 밀접한 관계가 있습니다.

```python
def dot_product_similarity(vec1, vec2):
    """
    내적 유사도 계산
    """
    # 정규화되지 않은 내적
    raw_dot = np.dot(vec1, vec2)

    # 정규화된 내적 (코사인 유사도와 동일)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    normalized_dot = raw_dot / (norm1 * norm2)

    return raw_dot, normalized_dot

# 예시
vec1 = np.array([1, 2, 3])
vec2 = np.array([2, 4, 6])  # vec1의 2배

raw_dot, norm_dot = dot_product_similarity(vec1, vec2)
print(f"원시 내적: {raw_dot}")        # 1*2 + 2*4 + 3*6 = 28
print(f"정규화 내적: {norm_dot}")    # 코사인 유사도 = 1.0
```

### 6.2 현재 Agent의 유사도 측정 방식

우리 RAG 시스템의 **Agent**에서는 **코사인 유사도**를 기본 유사도 측정 방식으로 사용합니다.

#### 선택 이유

```python
# 실제 agent 코드에서 사용하는 방식 (vector_store.py)
from qdrant_client.models import Distance, VectorParams

# 컬렉션 생성 시 코사인 유사도 지정
vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE)
```

**코사인 유사도를 선택한 이유:**

1. **의미적 유사성 포착**: 텍스트 임베딩에서 **방향성**이 의미를 더 잘 표현

   ```python
   # 예: 비슷한 주제의 문서들은 비슷한 방향을 가짐
   science_doc = "머신러닝 알고리즘"      # 방향: [0.2, 0.8, 0.1]
   tech_doc = "딥러닝 모델"             # 방향: [0.3, 0.7, 0.2] (유사)
   sports_doc = "축구 경기"             # 방향: [0.9, 0.1, 0.8] (다름)
   ```

2. **벡터 크기 정규화**: SentenceTransformers 모델이 이미 L2 정규화를 수행하므로 최적

   ```python
   # SentenceTransformers의 기본 동작
   from sentence_transformers import SentenceTransformer
   model = SentenceTransformer('all-MiniLM-L6-v2')
   embeddings = model.encode(texts, normalize_embeddings=True)  # L2 정규화
   ```

3. **Qdrant 최적화**: 벡터 데이터베이스가 코사인 유사도를 가장 효율적으로 처리

   ```python
   # Qdrant의 코사인 유사도 계산 최적화
   # 1. 정규화된 벡터이므로 내적만 계산하면 됨
   # 2. SIMD 명령어로 고속 계산 가능
   # 3. 메모리 사용량 감소
   ```

4. **다국어 지원**: 한국어-영어 간 비교 시 크기 차이 무시하고 방향만 고려

#### 실제 사용 예시

```python
# agent의 실제 검색 로직 (vector_store.py)
async def search_similar(self, collection_name: str, query_vector: List[float]):
    search_result = await asyncio.to_thread(
        self.client.search,
        collection_name=collection_name,
        query_vector=query_vector,
        limit=5,
        score_threshold=0.7,  # 코사인 유사도 임계값
        query_filter=query_filter
    )

    # 결과: 코사인 유사도 점수 (0.7 ~ 1.0 범위)
    for hit in search_result:
        score = hit.score  # 코사인 유사도 값
        content = hit.payload["content"]
```

#### 다른 유사도 방식과의 비교

| 방식         | 장점                          | 단점                   | 우리 시스템 적합성 |
| ------------ | ----------------------------- | ---------------------- | ------------------ |
| **코사인**   | 의미적 유사성 우수, 크기 독립 | 각도만 고려            | ⭐⭐⭐⭐⭐         |
| **유클리드** | 직관적, 계산 간단             | 차원의 저주, 크기 영향 | ⭐⭐               |
| **맨하탄**   | 이상치에 강함                 | 의미적 유사성 부족     | ⭐                 |
| **내적**     | 계산 효율성                   | 크기 편향              | ⭐⭐⭐             |

---

## 7. 임베딩 모델의 종류와 심층 분석

### 7.1 텍스트 임베딩 모델

#### 7.1.1 Sentence-BERT 계열

**개념**: BERT를 문장 임베딩용으로 미세 조정한 모델들

```python
from sentence_transformers import SentenceTransformer

# 대표적인 Sentence-BERT 모델들
models = {
    "all-MiniLM-L6-v2": "범용, 빠름, 384차원",
    "all-MiniLM-L12-v2": "더 깊음, 느림, 384차원",
    "all-mpnet-base-v2": "높은 품질, 768차원",
    "paraphrase-multilingual-MiniLM-L12-v2": "다국어 지원"
}

# 모델 로딩 및 사용
model = SentenceTransformer('all-MiniLM-L6-v2')
embeddings = model.encode(["Hello world", "Bonjour le monde"])
```

**장점:**

- 문장 수준 이해
- 의미적 유사성 우수
- 미세 조정 가능

**단점:**

- 계산 비용较高
- 모델 크기 큼

#### 7.1.2 E5 계열 (EmbEddings from bidirEctional Encoder)

**개념**: Microsoft에서 개발한 범용 임베딩 모델

```python
# E5 모델들
e5_models = {
    "intfloat/e5-small": "작고 빠름",
    "intfloat/e5-base": "균형 잡힘",
    "intfloat/e5-large": "높은 품질"
}

# 사용법 (비대칭 검색용 프롬프트)
query = "query: 머신러닝 알고리즘"
document = "passage: 머신러닝은 데이터로부터 패턴을 학습하는 기술"

model = SentenceTransformer('intfloat/e5-base')
query_emb = model.encode(query)
doc_emb = model.encode(document)
```

**특징:**

- 비대칭 검색에 강함 (질문 ↔ 문서)
- 프롬프트 튜닝으로 성능 향상
- 오픈소스

#### 7.1.3 BGE (BAAI General Embedding)

**개념**: 중국 BAAI에서 개발한 고성능 임베딩 모델

```python
# BGE 모델들
bge_models = {
    "BAAI/bge-small-en": "영어 특화",
    "BAAI/bge-base-en": "영어 범용",
    "BAAI/bge-large-en": "영어 고품질",
    "BAAI/bge-m3": "멀티모달 지원"
}
```

**장점:**

- 검색 성능 우수
- 긴 텍스트 처리 능력
- 다국어 지원

### 7.2 멀티모달 임베딩 모델

#### 7.2.1 CLIP (Contrastive Language-Image Pretraining)

**개념**: OpenAI에서 개발한 텍스트-이미지 통합 임베딩

```python
from transformers import CLIPProcessor, CLIPModel

model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# 텍스트와 이미지 임베딩
text = "a photo of a cat"
image = Image.open("cat.jpg")

inputs = processor(text=text, images=image, return_tensors="pt")
outputs = model(**inputs)

text_emb = outputs.text_embeds
image_emb = outputs.image_embeds
```

**특징:**

- 텍스트-이미지 의미 연결
- 제로샷 분류 가능
- 강력한 일반화 능력

#### 7.2.2 BLIP (Bootstrapping Language-Image Pretraining)

**개념**: Salesforce에서 개발한 이미지 캡셔닝용 모델

```python
from transformers import BlipProcessor, BlipModel

processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipModel.from_pretrained("Salesforce/blip-image-captioning-base")

# 이미지로부터 텍스트 생성 + 임베딩
inputs = processor(images=image, return_tensors="pt")
outputs = model.generate(**inputs)  # 캡션 생성

# 임베딩 추출
vision_outputs = model.vision_model(pixel_values=inputs["pixel_values"])
image_embeds = vision_outputs.last_hidden_state[:, 0, :]  # CLS 토큰
```

### 7.3 현재 Agent에서 사용하는 임베딩 모델

우리 RAG 시스템의 **Agent**에서는 **세 가지 임베딩 모델**을 상황에 맞게 사용합니다.

#### 7.3.1 기본 한국어 모델: `jhgan/ko-sroberta-multitask`

```python
# config.py에서 설정
DEFAULT_EMBEDDING_MODEL = "jhgan/ko-sroberta-multitask"
```

**선택 이유:**

1. **한국어 특화**: 한국어 텍스트에 최적화된 성능

   ```python
   # 한국어 이해 능력 테스트
   texts = [
       "머신러닝 모델 학습",     # 과학/기술
       "축구 경기 결과",         # 스포츠
       "요리 레시피 공유"        # 일상/요리
   ]

   model = SentenceTransformer("jhgan/ko-sroberta-multitask")
   embeddings = model.encode(texts)

   # 각 주제별로 다른 방향의 벡터 생성
   ```

2. **멀티태스크 학습**: 문장 분류, 유사도 측정, 자연어 추론 등 다양한 작업에 특화

   ```python
   # 멀티태스크 학습의 장점
   # 1. 단일 모델로 여러 작업 수행 가능
   # 2. 작업 간 지식 공유로 성능 향상
   # 3. 메모리 효율성 증가
   ```

3. **적절한 크기**: 768차원으로 정확도와 속도의 균형
   ```python
   # 크기 비교
   model_info = {
       "jhgan/ko-sroberta-multitask": "768차원, 한국어 최적화",
       "all-MiniLM-L6-v2": "384차원, 범용",
       "text-embedding-ada-002": "1536차원, 고품질"
   }
   ```

#### 7.3.2 멀티모달 모델: `clip-ViT-B-32`

```python
# config.py에서 설정
MULTIMODAL_EMBEDDING_MODEL = "clip-ViT-B-32"
```

**선택 이유:**

1. **텍스트-이미지 통합**: PDF 내 이미지와 텍스트를 함께 처리

   ```python
   # 멀티모달 검색 예시
   query = "보일러실 배치도"
   pdf_images = ["boiler_room_layout.jpg", "pipe_diagram.png"]

   # 텍스트 쿼리와 이미지들을 같은 벡터 공간에 매핑
   # 의미적으로 유사한 이미지 찾기 가능
   ```

2. **제로샷 능력**: 학습하지 않은 카테고리도 분류 가능

   ```python
   # 제로샷 이미지 분류
   categories = ["기술 도면", "배관 설계", "전기 배선도"]
   # 별도 학습 없이도 분류 가능
   ```

3. **범용성**: 다양한 이미지 형식 지원 (JPG, PNG, PDF 내 이미지)

#### 7.3.3 범용 텍스트 모델: `all-MiniLM-L6-v2`

```python
# config.py에서 설정
TEXT_EMBEDDING_MODEL = "all-MiniLM-L6-v2"
```

**선택 이유:**

1. **속도와 품질의 균형**: 빠른 처리 속도 + 준수한 성능

   ```python
   # 성능 벤치마크 (대략적)
   benchmarks = {
       "all-MiniLM-L6-v2": {"speed": "빠름", "quality": "중간", "size": "23MB"},
       "all-mpnet-base-v2": {"speed": "중간", "quality": "높음", "size": "109MB"},
       "text-embedding-ada-002": {"speed": "중간", "quality": "매우 높음", "size": "API"}
   }
   ```

2. **다국어 지원**: 영어 문서와 한국어 문서의 통합 검색

   ```python
   # 다국어 문서 처리
   mixed_docs = [
       "Machine learning algorithms",  # 영어
       "머신러닝 알고리즘",            # 한국어
       "Algorithmes d'apprentissage"   # 프랑스어
   ]
   # 같은 의미의 문서들이 가까운 벡터 위치
   ```

3. **생태계 호환성**: SentenceTransformers 라이브러리와 완벽 호환

### 7.4 모델 선택 전략

```python
# embedding_manager.py의 모델 선택 로직
class EmbeddingManager:
    def select_model(self, content_type: str, language: str) -> str:
        """
        콘텐츠 타입과 언어에 따른 최적 모델 선택
        """
        if content_type == "multimodal":
            return "clip-ViT-B-32"
        elif language == "korean":
            return "jhgan/ko-sroberta-multitask"
        elif language == "english":
            return "all-MiniLM-L6-v2"
        else:
            return "paraphrase-multilingual-MiniLM-L12-v2"  # 다국어
```

**선택 기준:**

1. **콘텐츠 타입**: 텍스트만 vs 텍스트+이미지
2. **언어**: 한국어 vs 영어 vs 다국어
3. **성능 요구사항**: 정확도 vs 속도 vs 메모리
4. **비용**: 오픈소스 vs API

---

## 8. 추가 최적화 기법

### 8.1 하이브리드 검색 (Hybrid Search)

**개념**: 유사도 검색 + 키워드 검색 결합

```python
class HybridSearch:
    def __init__(self, vector_search, keyword_search):
        self.vector_search = vector_search
        self.keyword_search = keyword_search

    def search(self, query: str, alpha: float = 0.5):
        """
        하이브리드 검색 수행
        alpha: 벡터 검색 가중치 (1-alpha: 키워드 검색 가중치)
        """
        # 벡터 유사도 검색
        vector_results = self.vector_search.search(query)

        # 키워드 검색
        keyword_results = self.keyword_search.search(query)

        # 결과 결합 (가중치 적용)
        combined_results = self.combine_results(
            vector_results, keyword_results, alpha
        )

        return combined_results
```

### 8.2 쿼리 확장 (Query Expansion)

**개념**: 원본 쿼리를 여러 관련 쿼리로 확장

```python
class QueryExpander:
    def expand_query(self, query: str) -> List[str]:
        """
        쿼리 확장 기법들
        """
        expansions = [query]  # 원본 쿼리 포함

        # 1. 동의어 확장
        synonyms = self.get_synonyms(query)
        expansions.extend(synonyms)

        # 2. 하이퍼님 확장
        hyponyms = self.get_hyponyms(query)
        expansions.extend(hyponyms)

        # 3. 관련 용어 확장
        related = self.get_related_terms(query)
        expansions.extend(related)

        return list(set(expansions))  # 중복 제거
```

### 8.3 리랭킹 (Re-ranking)

**개념**: 초기 검색 결과를 더 정교한 모델로 재정렬

```python
class ReRanker:
    def rerank(self, query: str, candidates: List[Document]) -> List[Document]:
        """
        검색 결과를 재정렬
        """
        reranked = []

        for doc in candidates:
            # 더 정교한 유사도 계산
            score = self.cross_encoder_score(query, doc.content)
            doc.score = score
            reranked.append(doc)

        # 점수 기준 정렬
        reranked.sort(key=lambda x: x.score, reverse=True)

        return reranked
```

이러한 고급 기법들을 통해 검색 정확도를 더욱 향상시킬 수 있습니다.
