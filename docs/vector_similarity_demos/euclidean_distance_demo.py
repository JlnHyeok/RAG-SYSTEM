#!/usr/bin/env python3
"""
유클리드 거리 계산 데모
실행 방법: python euclidean_distance_demo.py
"""

import numpy as np

def euclidean_distance(vec1, vec2):
    """유클리드 거리 계산"""
    diff = vec1 - vec2
    return np.sqrt(np.sum(diff ** 2))

# 실제 계산 예시
cat_vec = np.array([2.1, 3.4, 1.8])      # 고양이 [x, y, z] - 생물, 개인적, 작음
dog_vec = np.array([2.3, 3.1, 1.9])      # 강아지 [x, y, z] - 생물, 개인적, 작음
car_vec = np.array([7.2, 1.5, 2.8])      # 자동차 [x, y, z] - 기계, 공공적, 중간

print("=== 유클리드 거리 계산 ===")
print(f"고양이 ↔ 강아지: {euclidean_distance(cat_vec, dog_vec):.2f}")  # 가까움: 작은 값
print(f"고양이 ↔ 자동차: {euclidean_distance(cat_vec, car_vec):.2f}")  # 멀리 떨어짐: 큰 값
print(f"고양이 ↔ 고양이: {euclidean_distance(cat_vec, cat_vec):.2f}")  # 동일: 0.0

print("\n=== 유클리드 거리의 특징 ===")
print("- 범위: 0 (완전 동일) ~ ∞ (완전 다름)")
print("- 장점: 직관적이고 계산이 간단함")
print("- 단점: 차원 수가 증가할수록 거리가 커지는 현상 (차원의 저주)")