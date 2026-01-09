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
