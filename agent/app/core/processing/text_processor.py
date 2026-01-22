"""
텍스트 처리 유틸리티 모듈
RAG 시스템에서 사용되는 텍스트 처리, 컨텍스트 구성, 중복 제거 등의 기능을 담당합니다.
"""
import re
import logging
from typing import List, Dict, Any

from app.models.schemas import SearchResult

logger = logging.getLogger(__name__)


class TextProcessor:
    """텍스트 처리 및 컨텍스트 구성을 담당하는 클래스"""
    
    # 한국어 불용어 목록
    KOREAN_STOPWORDS = {
        '은', '는', '이', '가', '을', '를', '에', '의', '로', '와', '과',
        '도', '만', '까지', '부터', '에서', '으로', '라고', '하고'
    }
    
    # 영어 불용어 목록
    ENGLISH_STOPWORDS = {
        'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 
        'to', 'for', 'of', 'with', 'by', 'from', 'as', 'is', 'was'
    }
    
    def build_context(self, search_results: List[SearchResult]) -> str:
        """검색 결과를 바탕으로 컨텍스트 구성"""
        if not search_results:
            return ""
        
        context_parts = []
        for i, result in enumerate(search_results, 1):
            metadata = result.metadata
            
            # 컨텍스트 헤더 구성
            context_header = f"[문서 {i}"
            if metadata.get("page"):
                context_header += f", 페이지 {metadata['page']}"
            if metadata.get("type"):
                context_header += f", {metadata['type']}"
            if metadata.get("file_path"):
                file_name = metadata["file_path"].split("/")[-1]
                context_header += f", 출처: {file_name}"
            context_header += "]"
            
            # 내용 추가
            content = result.content.strip()
            if len(content) > 1000:  # 긴 내용은 요약
                content = content[:1000] + "..."
            
            context_part = f"{context_header}\n{content}\n"
            context_parts.append(context_part)
        
        return "\n".join(context_parts)
    
    def remove_duplicate_content(self, answer: str) -> str:
        """답변에서 중복된 내용을 제거하고 표를 재포맷팅"""
        # 먼저 표 재포맷팅
        answer = self._reformat_markdown_table(answer)
        
        # 섹션별로 나누기 (예: **간단한 답변:**, **자세한 설명:** 등)
        sections = re.split(r'(\*\*.*?\*\*:)', answer)
        
        # 중복 제거를 위한 집합
        seen_lines = set()
        cleaned_sections = []
        
        for section in sections:
            if section.startswith('**') and section.endswith('**:'):
                # 섹션 헤더는 그대로 유지
                cleaned_sections.append(section)
                seen_lines.clear()  # 섹션 바뀔 때마다 초기화
            else:
                # 섹션 내용에서 중복 라인 제거
                lines = section.split('\n')
                unique_lines = []
                for line in lines:
                    line_stripped = line.strip()
                    if line_stripped and line_stripped not in seen_lines:
                        unique_lines.append(line)
                        seen_lines.add(line_stripped)
                    elif not line_stripped:
                        unique_lines.append(line)  # 빈 줄은 유지
                cleaned_sections.append('\n'.join(unique_lines))
        
        return ''.join(cleaned_sections).strip()

    def _reformat_markdown_table(self, text: str) -> str:
        """Markdown 표를 찾아서 정렬된 표로 재포맷팅"""
        lines = text.split('\n')
        table_start = -1
        table_end = -1
        
        for i, line in enumerate(lines):
            if line.strip().startswith('|') and '|' in line:
                if table_start == -1:
                    table_start = i
                table_end = i
            elif table_start != -1 and not line.strip().startswith('|'):
                break
        
        if table_start == -1 or table_end - table_start < 2:
            return text  # 표가 없거나 너무 작으면 그대로 반환
        
        # 표 라인들 추출
        table_lines = lines[table_start:table_end + 1]
        
        # 각 열의 내용 추출
        columns = []
        for line in table_lines:
            cells = [cell.strip() for cell in line.split('|')[1:-1]]
            columns.append(cells)
        
        if not columns:
            return text
        
        # 각 열의 최대 너비 계산
        max_widths = []
        for col_idx in range(len(columns[0])):
            max_width = 0
            for row in columns:
                if col_idx < len(row):
                    max_width = max(max_width, len(row[col_idx]))
            max_widths.append(max_width)
        
        # 표 재구성
        formatted_lines = []
        for row_idx, row in enumerate(columns):
            formatted_cells = []
            for col_idx, cell in enumerate(row):
                if col_idx < len(max_widths):
                    formatted_cells.append(cell.center(max_widths[col_idx]))
                else:
                    formatted_cells.append(cell)
            
            formatted_line = '| ' + ' | '.join(formatted_cells) + ' |'
            formatted_lines.append(formatted_line)
            
            # 헤더 다음에 구분선 추가
            if row_idx == 0:
                separator_cells = ['-' * max_widths[col_idx] for col_idx in range(len(max_widths))]
                separator_line = '| ' + ' | '.join(separator_cells) + ' |'
                formatted_lines.append(separator_line)
        
        # 원본 텍스트에 재삽입
        new_lines = lines[:table_start] + formatted_lines + lines[table_end + 1:]
        return '\n'.join(new_lines)

    def parse_document_sections(self, context: str) -> List[Dict[str, str]]:
        """문서 섹션 파싱"""
        sections = []
        lines = context.strip().split('\n')
        
        current_header = ""
        current_content = []
        
        for line in lines:
            if line.startswith('[문서'):
                if current_header and current_content:
                    sections.append({
                        'header': current_header,
                        'content': '\n'.join(current_content).strip()
                    })
                current_header = line.strip('[]')
                current_content = []
            elif line.strip():
                current_content.append(line)
        
        if current_header and current_content:
            sections.append({
                'header': current_header,
                'content': '\n'.join(current_content).strip()
            })
        
        return sections

    def extract_keywords(self, text: str) -> List[str]:
        """간단한 키워드 추출 (한국어 지원)"""
        # 한국어, 영어, 숫자 조합 추출
        words = re.findall(r'[가-힣a-zA-Z0-9]+', text.lower())
        
        # 불용어 제거
        all_stopwords = self.KOREAN_STOPWORDS | self.ENGLISH_STOPWORDS
        keywords = [word for word in words if len(word) > 1 and word not in all_stopwords]
        
        return list(set(keywords))  # 중복 제거
    
    def calculate_text_relevance(self, text: str, keywords: List[str]) -> float:
        """텍스트와 키워드 간 관련성 점수 계산"""
        if not keywords:
            return 0.0
        
        text_lower = text.lower()
        matches = 0
        
        for keyword in keywords:
            if keyword in text_lower:
                matches += text_lower.count(keyword)
        
        # 매치 수를 텍스트 길이로 정규화
        return matches / max(len(text.split()), 1)
    
    def create_direct_document_answer(self, question: str, context: str) -> str:
        """문서 내용을 직접 구조화하여 답변 생성"""
        if not context.strip():
            return "관련 문서를 찾을 수 없습니다. 다른 키워드로 검색해보세요."
        
        # 문서 섹션 파싱
        document_sections = self.parse_document_sections(context)
        
        # 질문 키워드 기반 관련성 높은 내용 우선 배치
        question_keywords = self.extract_keywords(question)
        scored_sections = []
        
        for section in document_sections:
            relevance_score = self.calculate_text_relevance(
                section['content'], 
                question_keywords
            )
            scored_sections.append((relevance_score, section))
        
        # 관련성 순으로 정렬
        scored_sections.sort(key=lambda x: x[0], reverse=True)
        
        # 답변 구성
        answer_parts = [
            f"**'{question}'** 질문과 관련하여 다음과 같은 정보를 찾았습니다:\n"
        ]
        
        for i, (score, section) in enumerate(scored_sections[:5]):
            answer_parts.append(f"**{section['header']}**")
            content = section['content']
            if len(content) > 1200:
                sentences = content.split('.')
                if len(sentences) > 3:
                    content = '. '.join(sentences[:int(len(sentences)*0.7)]) + "...\n\n(추가 내용 있음)"
                else:
                    content = content[:1200] + "..."
            answer_parts.append(content)
            answer_parts.append("")
        
        if len(scored_sections) > 5:
            answer_parts.append(f"📋 추가로 {len(scored_sections) - 5}개의 관련 문서가 더 있습니다.")
        
        answer_parts.append("💡 **AI 분석이 일시적으로 제한되어 원본 문서 내용을 직접 제공했습니다.**")
        
        return "\n".join(answer_parts)
    
    def create_llm_free_summary(self, question: str, context: str) -> str:
        """LLM 없이 질문에 맞는 문서 요약 (최종 fallback)"""
        if not context.strip():
            return "관련 문서를 찾을 수 없습니다."
        
        sections = self.parse_document_sections(context)
        
        answer_parts = [
            f"# 📄 '{question}' 관련 문서 내용\n",
            "⚠️ **AI 답변 생성에 실패하여 원본 문서 내용을 직접 제공합니다.**\n"
        ]
        
        for i, section in enumerate(sections[:5], 1):
            answer_parts.append(f"## {i}. {section['header']}")
            
            content = section['content']
            if len(content) > 1000:
                content = content[:1000] + "\n\n... (내용이 길어 일부만 표시됨)"
            
            answer_parts.append(content)
            answer_parts.append("")
        
        if len(sections) > 5:
            answer_parts.append(f"📋 **추가로 {len(sections) - 5}개의 문서 섹션이 더 있습니다.**")
        
        answer_parts.append("---")
        answer_parts.append("💡 **더 정확한 답변을 원하시면 구체적인 키워드로 다시 질문해주세요.**")
        
        return self.remove_duplicate_content("\n".join(answer_parts))


# 전역 인스턴스
text_processor = TextProcessor()
