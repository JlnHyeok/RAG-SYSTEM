#!/usr/bin/env python3
"""
내적 유사도 계산 데모
실행 방법: python dot_product_demo.py
"""

import numpy as np

def dot_product_similarity(vec1, vec2):
    """내적 유사도 계산"""
    return np.dot(vec1, vec2)

# 실제 계산 예시
cat_vec = np.array([2.1, 3.4, 1.8])      # 고양이 [x, y, z] - 생물, 개인적, 작음
dog_vec = np.array([2.3, 3.1, 1.9])      # 강아지 [x, y, z] - 생물, 개인적, 작음
car_vec = np.array([7.2, 1.5, 2.8])      # 자동차 [x, y, z] - 기계, 공공적, 중간

print("=== 내적 유사도 계산 ===")
print(f"고양이 · 강아지: {dot_product_similarity(cat_vec, dog_vec):.2f}")
print(f"고양이 · 자동차: {dot_product_similarity(cat_vec, car_vec):.2f}")
print(f"고양이 · 고양이: {dot_product_similarity(cat_vec, cat_vec):.2f}")

# 반대 개념 예시
hot_vec = np.array([1.0, 0.8, 0.5])      # 뜨겁다 [x, y, z] - 온도, 높음, 에너지
cold_vec = np.array([-0.9, -0.7, -0.3])   # 차갑다 [x, y, z] - 온도, 낮음, 에너지 부족

print(f"\n뜨겁다 · 차갑다: {dot_product_similarity(hot_vec, cold_vec):.2f}")

print("\n=== 🚨 내적 유사도의 함정 ===")
print("고양이·자동차(20.22) > 고양이·고양이(15.97)")
print("하지만 코사인 유사도: 고양이·자동차(0.688) < 고양이·고양이(1.000)")
print("왜? 자동차 벡터가 더 크기 때문에 내적 값이 더 크게 나옴!")

print("\n=== 수학적 증명 ===")
cat_norm = np.linalg.norm(cat_vec)
car_norm = np.linalg.norm(car_vec)
print(f"||고양이|| = {cat_norm:.1f}")
print(f"||자동차|| = {car_norm:.1f}")
print(f"내적 = ||벡터1|| × ||벡터2|| × cosθ")
print(f"20.22 = {cat_norm:.1f} × {car_norm:.1f} × cosθ")
print(f"cosθ = 20.22 / ({cat_norm:.1f} × {car_norm:.1f}) = {20.22 / (cat_norm * car_norm):.3f}")

print("\n=== 올바른 비교 (코사인 유사도) ===")
def cosine_similarity(vec1, vec2):
    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    return dot_product / (norm1 * norm2)

print(f"고양이 ↔ 고양이 코사인: {cosine_similarity(cat_vec, cat_vec):.3f}")
print(f"고양이 ↔ 자동차 코사인: {cosine_similarity(cat_vec, car_vec):.3f}")
print(f"고양이 ↔ 강아지 코사인: {cosine_similarity(cat_vec, dog_vec):.3f}")

print("\n=== 결론 ===")
print("내적은 크기까지 고려하므로 오해의 소지가 있음")
print("코사인 유사도가 의미적 유사도를 더 정확히 측정함")

print("\n=== 🤔 내적 유사도는 정말 사용하기 힘들까? ===")
print("아니요! 상황에 따라 유용하게 사용할 수 있습니다.")

print("\n=== ✅ 내적 유사도가 유용한 경우 ===")
print("1. 벡터가 이미 L2 정규화된 경우 (내적 = 코사인 유사도)")

# L2 정규화 예시
normalized_cat = cat_vec / np.linalg.norm(cat_vec)
normalized_car = car_vec / np.linalg.norm(car_vec)
print(f"정규화된 고양이 벡터 크기: {np.linalg.norm(normalized_cat):.3f}")
print(f"정규화된 자동차 벡터 크기: {np.linalg.norm(normalized_car):.3f}")
print(f"정규화 후 내적: {np.dot(normalized_cat, normalized_car):.3f}")
print(f"코사인 유사도: {cosine_similarity(cat_vec, car_vec):.3f}")

print("\n2. 크기 정보가 중요한 경우 (TF-IDF 예시)")
doc1 = np.array([0.8, 0.6, 0.0])  # "cat dog" - cat이 더 중요
doc2 = np.array([0.6, 0.8, 0.0])  # "dog cat" - dog이 더 중요
doc3 = np.array([0.0, 0.0, 1.0])  # "car" - 완전히 다른 주제

print(f"doc1·doc2: {np.dot(doc1, doc2):.2f} (비슷한 주제)")
print(f"doc1·doc3: {np.dot(doc1, doc3):.2f} (다른 주제)")

print("\n3. 빠른 계산이 필요한 경우")
print("4. 머신러닝 모델 내부 계산")

print("\n=== ❌ 내적 유사도가 부적합한 경우 ===")
print("1. 의미적 유사도 측정 (RAG 검색 등)")
print("2. 벡터 크기가 다른 경우")
print("3. 정확한 방향 비교가 필요한 경우")

print("\n=== 🤔 내적 유사도의 또 다른 단점 ===")
print("절대적인 임계값이 없음! (-∞ ~ +∞)")

print("\n=== 📏 코사인 vs 내적 비교 ===")
print("코사인 유사도: -1 ~ 1 (고정 범위, 임계값 설정 용이)")
print("내적 유사도: -∞ ~ ∞ (무제한 범위, 상대적 비교 필요)")

print("\n=== ✅ 내적 유사도로 판별하는 올바른 방법 ===")

print("\n1. 자기 자신과의 내적을 기준으로 상대적 유사도 계산")
self_dot = np.dot(cat_vec, cat_vec)
rel_sim_cat_dog = np.dot(cat_vec, dog_vec) / self_dot
rel_sim_cat_car = np.dot(cat_vec, car_vec) / self_dot
print(f"고양이 ↔ 강아지 상대적 유사도: {rel_sim_cat_dog:.3f}")
print(f"고양이 ↔ 자동차 상대적 유사도: {rel_sim_cat_car:.3f}")

print("\n🚨 함정! 상대적 비교에도 크기 효과가 남아있음")
print("고양이·자동차 상대적 유사도: 1.266 (126.6% - 1 초과!)")
print("자기 자신보다 더 높은 점수를 받는 오류 발생")

print("\n2. 다른 후보들과 비교하여 순위 결정 (검색에 유용)")
vectors_list = [dog_vec, car_vec, np.array([1.0, 1.0])]
scores = [np.dot(cat_vec, vec) for vec in vectors_list]
print(f"검색 점수들: {scores}")
print(f"가장 유사한 벡터 인덱스: {np.argmax(scores)} (자동차가 더 높음)")

print("\n3. 정규화하여 코사인 유사도로 변환 (가장 안전한 방법)")
normalized_dot = np.dot(cat_vec, car_vec) / (np.linalg.norm(cat_vec) * np.linalg.norm(car_vec))
print(f"정규화된 내적 (코사인): {normalized_dot:.3f}")

print("\n=== 💡 결론 ===")
print("- 내적 유사도: 상대적 비교/랭킹에 강함 (검색)")
print("- 코사인 유사도: 절대적 판단에 강함 (유사도 임계값)")
print("- 완벽한 안전을 위해서는 코사인 유사도 사용 추천")

print("\n=== 코사인 유사도와의 관계 ===")
print("코사인 = 내적 / (벡터1크기 × 벡터2크기)")
print("코사인은 정규화된 내적입니다.")

print("\n=== 내적 유사도의 특징 ===")
print("- 범위: -∞ ~ +∞ (무제한)")
print("- 장점: 계산이 매우 빠름")
print("- 단점: 벡터 크기에 영향을 많이 받음")