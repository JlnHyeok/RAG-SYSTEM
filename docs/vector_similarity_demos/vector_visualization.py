#!/usr/bin/env python3
"""
벡터 유사도 측정 데모 - 3D 벡터 좌표 시각화
실행 방법: python vector_visualization.py
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.font_manager as fm

# UTF-8 인코딩 설정
import sys
import locale
sys.stdout.reconfigure(encoding='utf-8')

# 한글 폰트 설정
def set_korean_font():
    """matplotlib에서 한글 폰트를 설정"""
    try:
        # 시스템에 설치된 한글 폰트 찾기
        font_list = [f.name for f in fm.fontManager.ttflist]
        korean_fonts = ['AppleGothic', 'Malgun Gothic', 'NanumGothic', 'Noto Sans CJK']

        for font_name in korean_fonts:
            if font_name in font_list:
                plt.rcParams['font.family'] = font_name
                print(f"✅ 한글 폰트 설정됨: {font_name}")
                return True

        # 기본 폰트로 설정 (한글 폰트가 없으면)
        plt.rcParams['font.family'] = 'DejaVu Sans'
        print("⚠️  한글 폰트가 없어 기본 폰트로 설정됨")
        return False

    except Exception as e:
        print(f"❌ 폰트 설정 중 오류: {e}")
        return False

# 한글 폰트 설정 실행
set_korean_font()
import matplotlib.font_manager as fm

# 한글 폰트 설정
def set_korean_font():
    """matplotlib에서 한글 폰트를 설정"""
    try:
        # 시스템에 설치된 한글 폰트 찾기
        font_list = [f.name for f in fm.fontManager.ttflist]
        korean_fonts = ['AppleGothic', 'Malgun Gothic', 'NanumGothic', 'Noto Sans CJK']

        for font_name in korean_fonts:
            if font_name in font_list:
                plt.rcParams['font.family'] = font_name
                print(f"✅ 한글 폰트 설정됨: {font_name}")
                return True

        # 기본 폰트로 설정 (한글 폰트가 없으면)
        plt.rcParams['font.family'] = 'DejaVu Sans'
        print("⚠️  한글 폰트가 없어 기본 폰트로 설정됨")
        return False

    except Exception as e:
        print(f"❌ 폰트 설정 중 오류: {e}")
        return False

# 한글 폰트 설정 실행
set_korean_font()

# 실제 벡터 좌표 예시 (3차원으로 확장)
vectors = {
    "고양이": np.array([2.1, 3.4, 1.8]),    # 🐱 [x, y, z] - 생물, 개인적, 작음
    "강아지": np.array([2.3, 3.1, 1.9]),    # 🐶 [x, y, z] - 생물, 개인적, 작음
    "사자": np.array([1.8, 3.8, 4.2]),      # 🦁 [x, y, z] - 생물, 개인적, 큼
    "자동차": np.array([7.2, 1.5, 2.8]),    # 🚗 [x, y, z] - 기계, 공공적, 중간
    "트럭": np.array([7.5, 1.2, 5.1]),      # 🚛 [x, y, z] - 기계, 공공적, 큼
}

print("=== 3D 벡터 좌표 시각화 ===")
for name, vec in vectors.items():
    print(f"{name}: ({vec[0]:.1f}, {vec[1]:.1f}, {vec[2]:.1f})")

def plot_3d_vectors_interactive(vectors_dict):
    """3D 벡터들을 인터랙티브하게 시각화"""
    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection='3d')

    # 색상과 마커 설정
    colors = ['red', 'orange', 'blue', 'green', 'purple']
    markers = ['o', 's', '^', 'D', '*']

    # 벡터들을 그리기
    for i, (name, vec) in enumerate(vectors_dict.items()):
        color = colors[i % len(colors)]
        marker = markers[i % len(markers)]

        # 3D 점 그리기 (크게)
        ax.scatter(vec[0], vec[1], vec[2],
                  c=color, marker=marker, s=200,
                  label=name, alpha=0.8, edgecolors='black', linewidth=2)

        # 벡터 화살표 그리기 (원점에서 시작)
        ax.quiver(0, 0, 0, vec[0], vec[1], vec[2],
                 color=color, alpha=0.4, arrow_length_ratio=0.08,
                 linewidth=2)

        # 텍스트 레이블 (벡터 끝에 간단히)
        ax.text(vec[0]+0.15, vec[1]+0.15, vec[2]+0.15,
               name, fontsize=10, fontweight='bold',
               bbox=dict(boxstyle="round,pad=0.2", facecolor='white', alpha=0.9),
               ha='left', va='bottom')

    # 축 설정
    ax.set_xlabel('X축: 기계 ↔ 생물', fontsize=12, fontweight='bold')
    ax.set_ylabel('Y축: 개인적 ↔ 공공적', fontsize=12, fontweight='bold')
    ax.set_zlabel('Z축: 큼 ↔ 작음', fontsize=12, fontweight='bold')

    ax.set_title('3D 벡터 공간: 의미적 유사도 시각화', fontsize=14, fontweight='bold', pad=20)

    # 범례
    ax.legend(loc='upper left', bbox_to_anchor=(0.02, 0.98), fontsize=10)

    # 그리드와 축 범위
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 8)
    ax.set_ylim(0, 5)
    ax.set_zlim(0, 6)

    # 축 눈금 설정
    ax.set_xticks([0, 2, 4, 6, 8])
    ax.set_yticks([0, 1, 2, 3, 4, 5])
    ax.set_zticks([0, 1, 2, 3, 4, 5, 6])

    # 시점 설정 (더 입체적으로 보이게)
    ax.view_init(elev=25, azim=45)

    plt.tight_layout()
    plt.show()

def analyze_spatial_relationships(vectors_dict):
    """공간적 관계 분석"""
    print("\n=== 3D 공간 거리 분석 ===")
    names = list(vectors_dict.keys())

    for i, name1 in enumerate(names):
        for j, name2 in enumerate(names):
            if i < j:
                vec1, vec2 = vectors_dict[name1], vectors_dict[name2]
                distance = np.linalg.norm(vec1 - vec2)
                print(f"{name1} ↔ {name2}: 3D 거리 = {distance:.2f}")

    print("\n=== 의미적 그룹 분석 ===")
    cat_vec = vectors_dict["고양이"]
    dog_vec = vectors_dict["강아지"]
    lion_vec = vectors_dict["사자"]
    car_vec = vectors_dict["자동차"]
    truck_vec = vectors_dict["트럭"]

    # 동물 그룹
    animal_group = [cat_vec, dog_vec, lion_vec]
    animal_center = np.mean(animal_group, axis=0)
    print(f"동물 그룹 중심: ({animal_center[0]:.1f}, {animal_center[1]:.1f}, {animal_center[2]:.1f})")

    # 탈것 그룹
    vehicle_group = [car_vec, truck_vec]
    vehicle_center = np.mean(vehicle_group, axis=0)
    print(f"탈것 그룹 중심: ({vehicle_center[0]:.1f}, {vehicle_center[1]:.1f}, {vehicle_center[2]:.1f})")

    # 그룹 간 거리
    group_distance = np.linalg.norm(animal_center - vehicle_center)
    print(f"동물 ↔ 탈것 그룹 거리: {group_distance:.2f}")

if __name__ == "__main__":
    print("\n3D 벡터 공간 설명:")
    print("• X축: 생물(0) ↔ 기계(8)")
    print("• Y축: 개인적(0) ↔ 공공적(5)")
    print("• Z축: 작음(0) ↔ 큼(6)")

    analyze_spatial_relationships(vectors)

    print("\n=== 3D 시각화 생성 중... ===")
    print("💡 matplotlib이 설치되어 있다면 3D 플롯이 표시됩니다.")
    print("💡 설치되지 않았다면 pip install matplotlib로 설치하세요.")

    try:
        plot_3d_vectors_interactive(vectors)
    except ImportError:
        print("❌ matplotlib이 설치되지 않았습니다.")
        print("   pip install matplotlib로 설치하세요.")
    except Exception as e:
        print(f"❌ 시각화 중 오류 발생: {e}")