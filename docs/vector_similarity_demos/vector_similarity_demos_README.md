# 벡터 유사도 측정 데모 파일들

이 디렉토리에는 RAG 시스템의 벡터 유사도 측정 개념을 실제 코드로 학습할 수 있는 데모 파일들이 있습니다.

## 📁 파일 구조

```
vector_similarity_demos/
├── vector_visualization.py          # 3D 벡터 좌표 시각화
├── cosine_similarity_demo.py        # 코사인 유사도 계산
├── euclidean_distance_demo.py       # 유클리드 거리 계산
├── dot_product_demo.py              # 내적 유사도 계산
├── similarity_comparison_demo.py    # 모든 방식 종합 비교
├── run_vector_visualization.sh      # 3D 벡터 시각화 실행 스크립트
├── run_cosine_similarity.sh         # 코사인 유사도 실행 스크립트
├── run_euclidean_distance.sh        # 유클리드 거리 실행 스크립트
├── run_dot_product.sh               # 내적 유사도 실행 스크립트
├── run_similarity_comparison.sh     # 유사도 비교 실행 스크립트
└── vector_similarity_demos_README.md # 이 파일
```

## 🚀 실행 방법

### 방법 1: 실행 스크립트 사용 (권장)

각 데모를 실행 스크립트로 쉽게 실행할 수 있습니다:

```bash
cd vector_similarity_demos

./run_vector_visualization.sh      # 3D 벡터 시각화
./run_cosine_similarity.sh         # 코사인 유사도
./run_euclidean_distance.sh        # 유클리드 거리
./run_dot_product.sh               # 내적 유사도
./run_similarity_comparison.sh     # 종합 비교
```

### 방법 2: 직접 Python 실행

Python 파일을 직접 실행할 수도 있습니다:

```bash
cd vector_similarity_demos

python3 vector_visualization.py      # 또는 python vector_visualization.py
python3 cosine_similarity_demo.py
python3 euclidean_distance_demo.py
python3 dot_product_demo.py
python3 similarity_comparison_demo.py
```

## 🎯 일괄 실행

모든 데모를 순서대로 실행하려면:

```bash
cd vector_similarity_demos

echo "=== 1. 벡터 시각화 ===" && ./run_vector_visualization.sh && echo
echo "=== 2. 코사인 유사도 ===" && ./run_cosine_similarity.sh && echo
echo "=== 3. 유클리드 거리 ===" && ./run_euclidean_distance.sh && echo
echo "=== 4. 내적 유사도 ===" && ./run_dot_product.sh && echo
echo "=== 5. 유사도 비교 ===" && ./run_similarity_comparison.sh
```

## 📚 학습 순서 추천

1. **3D 벡터 시각화** → 벡터 개념 입체적 이해 (`./run_vector_visualization.sh`)
2. **코사인 유사도** → 가장 중요한 유사도 측정 방식 (`./run_cosine_similarity.sh`)
3. **유클리드 거리** → 직관적인 거리 개념 (`./run_euclidean_distance.sh`)
4. **내적 유사도** → 기본적인 벡터 연산 (`./run_dot_product.sh`)
5. **유사도 비교** → 종합 비교 및 이해 (`./run_similarity_comparison.sh`)

## 🎯 RAG 시스템 적용

우리 RAG 시스템에서는 **코사인 유사도**를 기본 유사도 측정 방식으로 사용합니다:

- 의미적 유사성 포착에 최적
- 벡터 크기 정규화로 다국어 지원
- Qdrant 벡터 데이터베이스 최적화

## 📚 관련 문서

- [01-RAG-Concepts.md](../01-RAG-Concepts.md) - 벡터 유사도 심층 분석
- [07-Vector-Database-Guide.md](../07-Vector-Database-Guide.md) - 벡터 데이터베이스 가이드
- [08-Embedding-Guide.md](../08-Embedding-Guide.md) - 임베딩 가이드
