#!/bin/bash
# 유사도 비교 데모 실행 스크립트

echo "=== 유사도 비교 데모 실행 ==="
echo "실행 파일: similarity_comparison_demo.py"
echo "설명: 코사인 유사도, 유클리드 거리, 내적 유사도를 모두 비교"
echo ""

cd "$(dirname "$0")"

if command -v python3 &> /dev/null; then
    python3 similarity_comparison_demo.py
elif command -v python &> /dev/null; then
    python similarity_comparison_demo.py
else
    echo "Python이 설치되어 있지 않습니다."
    exit 1
fi