#!/bin/bash
# 유클리드 거리 데모 실행 스크립트

echo "=== 유클리드 거리 데모 실행 ==="
echo "실행 파일: euclidean_distance_demo.py"
echo "설명: 유클리드 거리의 원리와 계산 방법"
echo ""

cd "$(dirname "$0")"

if command -v python3 &> /dev/null; then
    python3 euclidean_distance_demo.py
elif command -v python &> /dev/null; then
    python euclidean_distance_demo.py
else
    echo "Python이 설치되어 있지 않습니다."
    exit 1
fi