#!/usr/bin/env python3
"""
벡터 유사도 비교 데모 - 모든 방식 비교
실행 방법: python similarity_comparison_demo.py
"""

import numpy as np

def cosine_similarity(vec1, vec2):
    """코사인 유사도 계산"""
    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    return dot_product / (norm1 * norm2)

def euclidean_distance(vec1, vec2):
    """유클리드 거리 계산"""
    diff = vec1 - vec2
    return np.sqrt(np.sum(diff ** 2))

def dot_product_similarity(vec1, vec2):
    """내적 유사도 계산"""
    return np.dot(vec1, vec2)

def compare_similarities(vectors_dict):
    """모든 벡터 쌍 간 유사도 비교"""
    names = list(vectors_dict.keys())
    results = []

    for i, name1 in enumerate(names):
        for j, name2 in enumerate(names):
            if i < j:  # 중복 계산 방지
                vec1, vec2 = vectors_dict[name1], vectors_dict[name2]
                cos_sim = cosine_similarity(vec1, vec2)
                euc_dist = euclidean_distance(vec1, vec2)
                dot_sim = dot_product_similarity(vec1, vec2)
                results.append((name1, name2, cos_sim, euc_dist, dot_sim))

    # 코사인 유사도 기준 정렬 (높은 유사도 순)
    results.sort(key=lambda x: x[2], reverse=True)

    print("=== 유사도 비교 결과 (코사인 유사도 높은 순) ===")
    print("형식: 개념1 ↔ 개념2: 코사인=값, 거리=값, 내적=값")
    print("-" * 60)
    for name1, name2, cos_sim, euc_dist, dot_sim in results:
        print(f"{name1} ↔ {name2}: 코사인={cos_sim:.3f}, 거리={euc_dist:.2f}, 내적={dot_sim:.2f}")

    return results

# 벡터 데이터 (3차원 확장)
vectors = {
    "고양이": np.array([2.1, 3.4, 1.8]),    # [x, y, z] - 생물, 개인적, 작음
    "강아지": np.array([2.3, 3.1, 1.9]),    # [x, y, z] - 생물, 개인적, 작음
    "사자": np.array([1.8, 3.8, 4.2]),      # [x, y, z] - 생물, 개인적, 큼
    "자동차": np.array([7.2, 1.5, 2.8]),    # [x, y, z] - 기계, 공공적, 중간
    "트럭": np.array([7.5, 1.2, 5.1]),      # [x, y, z] - 기계, 공공적, 큼
}

# 실행
results = compare_similarities(vectors)

print("\n=== 유사도 방식 비교 표 ===")
print("| 방식 | 고양이↔강아지 | 고양이↔자동차 | 계산 복잡도 | 의미 해석 |")
print("|------|---------------|----------------|-------------|-----------|")

# 고양이-강아지 비교
cat_dog_cos = cosine_similarity(vectors["고양이"], vectors["강아지"])
cat_dog_euc = euclidean_distance(vectors["고양이"], vectors["강아지"])
cat_dog_dot = dot_product_similarity(vectors["고양이"], vectors["강아지"])

# 고양이-자동차 비교
cat_car_cos = cosine_similarity(vectors["고양이"], vectors["자동차"])
cat_car_euc = euclidean_distance(vectors["고양이"], vectors["자동차"])
cat_car_dot = dot_product_similarity(vectors["고양이"], vectors["자동차"])

print(f"| 코사인 유사도 | {cat_dog_cos:.3f} (매우 유사) | {cat_car_cos:.3f} (다름) | 중간 | 방향 기반 의미 |")
print(f"| 유클리드 거리 | {cat_dog_euc:.2f} (가까움) | {cat_car_euc:.2f} (멀리 떨어짐) | 낮음 | 절대 거리 |")
print(f"| 내적 유사도 | {cat_dog_dot:.2f} (강한 상관) | {cat_car_dot:.2f} (강한 상관) | 낮음 | 크기 영향 받음 |")