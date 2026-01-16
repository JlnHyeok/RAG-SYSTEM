#!/bin/bash
# 내적 유사도 데모 실행 스크립트

echo "=== 내적 유사도 데모 실행 ==="
echo "실행 파일: dot_product_demo.py"
echo "설명: 벡터 내적의 개념과 계산"
echo ""

cd "$(dirname "$0")"

if command -v python3 &> /dev/null; then
    python3 dot_product_demo.py
elif command -v python &> /dev/null; then
    python dot_product_demo.py
else
    echo "Python이 설치되어 있지 않습니다."
    exit 1
fi