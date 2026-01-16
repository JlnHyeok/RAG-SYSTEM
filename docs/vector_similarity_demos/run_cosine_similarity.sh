#!/bin/bash
# 코사인 유사도 데모 실행 스크립트

echo "=== 코사인 유사도 데모 실행 ==="
echo "실행 파일: cosine_similarity_demo.py"
echo "설명: 코사인 유사도의 원리와 계산 방법"
echo ""

cd "$(dirname "$0")"

if command -v python3 &> /dev/null; then
    python3 cosine_similarity_demo.py
elif command -v python &> /dev/null; then
    python cosine_similarity_demo.py
else
    echo "Python이 설치되어 있지 않습니다."
    exit 1
fi