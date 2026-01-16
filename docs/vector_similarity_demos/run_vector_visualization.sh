#!/bin/bash
# 3D 벡터 시각화 데모 실행 스크립트

echo "=== 3D 벡터 시각화 데모 실행 ==="
echo "실행 파일: vector_visualization.py"
echo "설명: 실제 벡터 좌표를 3D 공간에서 입체적으로 시각화"
echo "주의: matplotlib이 설치되어 있어야 3D 플롯이 표시됩니다."
echo ""

cd "$(dirname "$0")"

if command -v python3 &> /dev/null; then
    python3 vector_visualization.py
elif command -v python &> /dev/null; then
    python vector_visualization.py
else
    echo "Python이 설치되어 있지 않습니다."
    exit 1
fi