#!/usr/bin/env python3
"""
코사인 유사도 계산 데모
실행 방법: python cosine_similarity_demo.py
"""

import numpy as np

def cosine_similarity(vec1, vec2):
    """코사인 유사도 계산"""
    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    return dot_product / (norm1 * norm2)

# 실제 계산 예시
cat_vec = np.array([2.1, 3.4, 1.8])      # 고양이 [x, y, z] - 생물, 개인적, 작음
dog_vec = np.array([2.3, 3.1, 1.9])      # 강아지 [x, y, z] - 생물, 개인적, 작음
car_vec = np.array([7.2, 1.5, 2.8])      # 자동차 [x, y, z] - 기계, 공공적, 중간

print("=== 코사인 유사도 계산 ===")
print(f"고양이 ↔ 강아지: {cosine_similarity(cat_vec, dog_vec):.3f}")  # 유사: 높은 값
print(f"고양이 ↔ 자동차: {cosine_similarity(cat_vec, car_vec):.3f}")  # 다름: 낮은 값
print(f"고양이 ↔ 고양이: {cosine_similarity(cat_vec, cat_vec):.3f}")  # 동일: 1.0

print("\n=== 코사인 유사도의 특징 ===")
print("- 범위: -1 (완전 반대) ~ 1 (완전 동일)")
print("- 장점: 벡터 크기에 영향을 받지 않음")
print("- 단점: 각도만 고려하므로 크기 정보 손실")