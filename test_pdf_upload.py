#!/usr/bin/env python3
"""
PDF 업로드 및 이미지 처리 테스트 스크립트
"""

import requests
import os
from pathlib import Path

# 서버 URL
BASE_URL = "http://localhost:8000"

def create_test_pdf():
    """테스트용 PDF 파일 생성"""
    from reportlab.pdfgen import canvas
    from reportlab.lib.pagesizes import letter
    import os
    
    # 테스트 PDF 파일 경로
    test_dir = Path("test_files")
    test_dir.mkdir(exist_ok=True)
    pdf_path = test_dir / "test_document.pdf"
    
    # PDF 생성
    c = canvas.Canvas(str(pdf_path), pagesize=letter)
    
    # 첫 번째 페이지 - 텍스트만
    c.drawString(100, 750, "테스트 문서 - 첫 번째 페이지")
    c.drawString(100, 700, "이 문서는 RAG 시스템의 PDF 처리 기능을 테스트합니다.")
    c.drawString(100, 650, "한글과 영어가 모두 포함되어 있습니다.")
    c.drawString(100, 600, "This document contains both Korean and English text.")
    c.drawString(100, 550, "PDF 파일에서 텍스트를 올바르게 추출할 수 있는지 확인합니다.")
    
    # 두 번째 페이지
    c.showPage()
    c.drawString(100, 750, "테스트 문서 - 두 번째 페이지")
    c.drawString(100, 700, "이 페이지도 텍스트 추출 테스트용입니다.")
    c.drawString(100, 650, "여러 페이지로 구성된 PDF도 처리할 수 있어야 합니다.")
    
    c.save()
    print(f"테스트 PDF 생성 완료: {pdf_path}")
    return pdf_path

def upload_pdf(pdf_path):
    """PDF 파일 업로드"""
    url = f"{BASE_URL}/api/v1/documents/upload"
    
    with open(pdf_path, 'rb') as f:
        files = {'file': (pdf_path.name, f, 'application/pdf')}
        data = {'user_id': 'test_user'}
        
        print(f"PDF 업로드 중: {pdf_path.name}")
        response = requests.post(url, files=files, data=data, timeout=60)
        
        if response.status_code == 200:
            result = response.json()
            print("✓ PDF 업로드 성공!")
            print(f"  - Document ID: {result.get('document_id')}")
            print(f"  - 텍스트 청크: {result.get('text_chunks', 0)}")
            print(f"  - 이미지 청크: {result.get('image_chunks', 0)}")
            print(f"  - 총 임베딩: {result.get('total_embeddings', 0)}")
            print(f"  - 처리 시간: {result.get('processing_time', 0):.2f}초")
            return result
        else:
            print(f"✗ PDF 업로드 실패: {response.status_code}")
            print(f"  오류: {response.text}")
            return None

def test_query(query_text):
    """업로드된 문서로 질의 테스트"""
    url = f"{BASE_URL}/api/v1/query"
    
    payload = {
        "query": query_text,
        "user_id": "test_user",
        "include_sources": True
    }
    
    print(f"\n질의 테스트: '{query_text}'")
    response = requests.post(url, json=payload)
    
    if response.status_code == 200:
        result = response.json()
        print("✓ 질의 성공!")
        print(f"  답변: {result.get('answer', '')}")
        
        sources = result.get('sources', [])
        if sources:
            print(f"  관련 문서 {len(sources)}개 발견:")
            for i, source in enumerate(sources[:3], 1):
                print(f"    {i}. {source.get('content', '')[:100]}...")
        return result
    else:
        print(f"✗ 질의 실패: {response.status_code}")
        print(f"  오류: {response.text}")
        return None

def main():
    """메인 테스트 함수"""
    print("=== PDF 업로드 및 처리 테스트 ===\n")
    
    try:
        # 1. 테스트 PDF 생성
        pdf_path = create_test_pdf()
        
        # 2. PDF 업로드
        upload_result = upload_pdf(pdf_path)
        
        if upload_result:
            # 3. 질의 테스트
            test_queries = [
                "이 문서는 무엇에 대한 내용인가요?",
                "한글과 영어가 모두 포함되어 있나요?",
                "테스트 문서의 목적은 무엇인가요?"
            ]
            
            for query in test_queries:
                test_query(query)
                print()
        
        print("=== 테스트 완료 ===")
        
    except Exception as e:
        print(f"테스트 중 오류 발생: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()