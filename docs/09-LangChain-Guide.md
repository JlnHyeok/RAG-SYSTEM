# LangChain 완전 가이드

## 1. LangChain이란 무엇인가?

**LangChain**은 **대형 언어 모델(LLM)을 활용한 애플리케이션 개발을 위한 프레임워크**입니다. 복잡한 AI 워크플로우를 체인 형태로 연결하여 강력하고 유연한 AI 애플리케이션을 구축할 수 있게 해줍니다.

### LLM(Large Language Model)이란?

**대형 언어 모델(LLM)**은 방대한 텍스트 데이터로 훈련된 신경망 모델로, 인간과 유사한 텍스트 이해 및 생성 능력을 가집니다.

**LLM의 핵심 특징:**

1. **규모**: 수십억~수조 개의 파라미터를 가진 거대한 모델

   - GPT-4: 약 1.7조 개 파라미터 (추정)
   - Claude: 수천억 개 파라미터
   - LLaMA 2: 7B~70B 파라미터

2. **학습 방식**:

   - **Pre-training**: 인터넷의 방대한 텍스트로 언어 패턴 학습
   - **Fine-tuning**: 특정 작업에 맞게 추가 훈련
   - **RLHF**: 인간 피드백을 통한 강화학습으로 정렬

3. **능력**:
   - **언어 이해**: 문맥, 의도, 감정 파악
   - **텍스트 생성**: 자연스럽고 일관된 텍스트 생성
   - **추론**: 논리적 사고와 문제 해결
   - **Few-shot Learning**: 적은 예시로 새로운 작업 수행

**LLM의 한계와 LangChain의 해결책:**

- **한계**: 단일 입력-출력, 외부 데이터 접근 불가, 도구 사용 불가
- **LangChain 해결**: 체인을 통한 복잡한 워크플로우, 외부 도구 연동, 메모리 관리

### LangChain의 철학: "체이닝(Chaining)"

전통적인 AI 개발에서는 하나의 모델이 모든 것을 처리해야 했습니다. LangChain은 **"작은 구성요소들을 체인으로 연결하여 복잡한 작업 해결"**이라는 철학을 제시합니다.

**체이닝의 장점:**

- **모듈성**: 각 단계를 독립적으로 개발/테스트
- **재사용성**: 동일한 체인을 다양한 용도로 활용
- **디버깅**: 문제 발생 지점을 쉽게 추적
- **최적화**: 각 단계별로 성능 튜닝 가능

### 1.1 LangChain의 핵심 개념

```python
# 기본적인 LangChain 사용 예시
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

# 1. LLM 초기화
llm = OpenAI(temperature=0.7)

# 2. 프롬프트 템플릿 정의
prompt = PromptTemplate(
    input_variables=["topic"],
    template="다음 주제에 대해 간단히 설명해주세요: {topic}"
)

# 3. 체인 생성
chain = LLMChain(llm=llm, prompt=prompt)

# 4. 실행
result = chain.run(topic="머신러닝")
print(result)
```

### 1.2 LangChain의 장점

- **모듈성**: 각 컴포넌트를 독립적으로 사용 가능
- **확장성**: 다양한 LLM과 도구 지원
- **재사용성**: 체인을 조합하여 복잡한 워크플로우 구성
- **표준화**: 일관된 인터페이스 제공

## 2. LangChain 핵심 컴포넌트

### 2.1 LLMs (Large Language Models)

**LangChain에서의 LLM 추상화**

LangChain은 다양한 LLM 제공업체들을 통일된 인터페이스로 사용할 수 있게 해줍니다. 이를 통해 모델 간 전환이 용이하고, 벤더 락인을 방지할 수 있습니다.

**주요 LLM 카테고리:**

1. **텍스트 완성 모델 (Completion Models)**

   - 주어진 텍스트를 이어서 작성
   - 예: GPT-3.5-turbo-instruct, text-davinci-003
   - 용도: 창작, 코드 생성, 긴 텍스트 작성

2. **채팅 모델 (Chat Models)**

   - 대화 형태의 입출력
   - 예: GPT-4, Claude, ChatGPT
   - 용도: Q&A, 대화형 애플리케이션

3. **임베딩 모델 (Embedding Models)**
   - 텍스트를 벡터로 변환
   - 예: text-embedding-ada-002, sentence-transformers
   - 용도: 검색, 유사도 계산, 분류

**모델 선택 기준:**

- **성능 vs 비용**: GPT-4 (고성능/고비용) vs GPT-3.5 (적당한 성능/저비용)
- **속도 vs 품질**: 빠른 응답이 필요한 실시간 앱 vs 고품질이 중요한 배치 작업
- **데이터 보안**: 클라우드 API vs 온프레미스 모델
- **다국어 지원**: 특정 언어에 특화된 모델 고려

**LLM 파라미터 이해:**

- **Temperature (0.0-2.0)**: 창의성 조절

  - 0.0: 일관되고 예측 가능한 출력
  - 1.0: 균형잡힌 창의성
  - 2.0: 매우 창의적이지만 불안정할 수 있음

- **Max Tokens**: 응답 길이 제한
- **Top-p**: 다음 토큰 선택 시 고려할 확률 범위

```python
from langchain.llms import OpenAI, HuggingFacePipeline
from langchain.chat_models import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage

# OpenAI LLM
openai_llm = OpenAI(
    model_name="gpt-3.5-turbo-instruct",
    temperature=0.7,
    max_tokens=1000
)

# ChatGPT 모델
chat_model = ChatOpenAI(
    model_name="gpt-4",
    temperature=0.1
)

# HuggingFace 모델
hf_llm = HuggingFacePipeline.from_model_id(
    model_id="microsoft/DialoGPT-medium",
    task="text-generation",
    model_kwargs={"temperature": 0.7, "max_length": 1000}
)

# 채팅 모델 사용
messages = [
    SystemMessage(content="당신은 도움이 되는 AI 어시스턴트입니다."),
    HumanMessage(content="Python에 대해 설명해주세요.")
]
response = chat_model(messages)
print(response.content)
```

### 2.2 Prompts (프롬프트 템플릿)

**프롬프트 엔지니어링의 이론**

프롬프트는 LLM과 사용자 간의 **커뮤니케이션 인터페이스**입니다. 좋은 프롬프트는 LLM의 능력을 최대한 끌어낼 수 있는 핸들역할을 합니다.

**효과적인 프롬프트의 원칙:**

1. **명확성 (Clarity)**: 모호하지 않은 지시사항
2. **구체성 (Specificity)**: 구체적인 예시와 가이드라인
3. **맥락성 (Context)**: 최적의 배경 정보 제공
4. **제약성 (Constraints)**: 답변의 형식이나 범위 제한
5. **예시성 (Examples)**: Few-shot learning을 통한 성능 향상

**프롬프트 태입:**

1. **Zero-shot**: 예시 없이 지시만 제공

   ```
   "다음 문서를 3줄로 요약해주세요: {document}"
   ```

2. **One-shot**: 하나의 예시 제공

   ```
   "예시: 입력 - '기계학습은...' 출력 - '기계학습: 데이터를 통해...'
   이제 다음을 요약해주세요: {document}"
   ```

3. **Few-shot**: 여러 예시를 통한 학습
   - 일반적으로 3-5개 예시가 최적
   - 다양한 케이스를 커버하는 예시 선택

**비즈니스 도메인별 프롬프트 모범 사례:**

- **기술 문서 Q&A**: 전문 용어 정의, 정확한 참조 요구
- **법률 문서**: 신중한 해석, 근거 조항 명시 요구
- **의료 정보**: 전문의르타스리 방지, 매우 신중한 표현
- **금융 데이터**: 위험 경고, 단순 해석 금지

```python
from langchain.prompts import (
    PromptTemplate,
    ChatPromptTemplate,
    FewShotPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate
)

# 기본 프롬프트 템플릿
simple_prompt = PromptTemplate(
    input_variables=["product", "audience"],
    template="{product}를 {audience}에게 마케팅하기 위한 슬로건을 만들어주세요."
)

# 채팅 프롬프트 템플릿
chat_prompt = ChatPromptTemplate.from_messages([
    SystemMessagePromptTemplate.from_template(
        "당신은 {role} 전문가입니다."
    ),
    HumanMessagePromptTemplate.from_template(
        "{question}에 대해 전문적인 답변을 해주세요."
    )
])

# Few-shot 프롬프트 (예시 기반)
examples = [
    {"word": "happy", "antonym": "sad"},
    {"word": "tall", "antonym": "short"},
]

example_formatter_template = """
단어: {word}
반대말: {antonym}
"""

example_prompt = PromptTemplate(
    input_variables=["word", "antonym"],
    template=example_formatter_template,
)

few_shot_prompt = FewShotPromptTemplate(
    examples=examples,
    example_prompt=example_prompt,
    prefix="다음은 단어와 그 반대말의 예시입니다:",
    suffix="단어: {input}\n반대말:",
    input_variables=["input"],
    example_separator="\n\n",
)

# 사용 예시
formatted_prompt = few_shot_prompt.format(input="good")
print(formatted_prompt)
```

### 2.3 Chains (체인)

**체인의 개념과 설계 철학**

체인은 LangChain의 핵심 개념으로, **여러 컴포넌트를 순차적 또는 병렬로 연결하여 복잡한 워크플로우를 구성**하는 방법입니다. 전통적인 함수형 프로그래밍의 파이프라인 개념을 AI 워크플로우에 적용한 것입니다.

**체인의 장점:**

1. **모듈화**: 각 단계를 독립적으로 개발/테스트
2. **재사용성**: 동일한 체인을 다양한 용도로 활용
3. **디버깅**: 문제 발생 지점을 쉽게 추적
4. **최적화**: 각 단계별로 성능 튜닝 가능
5. **확장성**: 새로운 단계를 쉽게 추가

**체인 유형별 사용 사례:**

- **SimpleSequentialChain**: 단순한 순차 처리 (A→B→C)
- **SequentialChain**: 복잡한 입출력을 가진 순차 처리
- **MapReduceChain**: 대용량 데이터 병렬 처리
- **MapRerankChain**: 여러 결과 중 최적 선택
- **ConversationalChain**: 대화형 애플리케이션

#### Simple Chain 예시

```python
from langchain.chains import LLMChain, SimpleSequentialChain, SequentialChain

# 첫 번째 체인: 주제 생성
topic_chain = LLMChain(
    llm=llm,
    prompt=PromptTemplate(
        input_variables=["subject"],
        template="{subject}에 관한 블로그 글 제목을 생성해주세요."
    ),
    output_key="title"
)

# 두 번째 체인: 내용 작성
content_chain = LLMChain(
    llm=llm,
    prompt=PromptTemplate(
        input_variables=["title"],
        template="{title}라는 제목으로 블로그 글을 작성해주세요."
    ),
    output_key="content"
)

# 순차적 체인 연결
blog_chain = SimpleSequentialChain(
    chains=[topic_chain, content_chain],
    verbose=True
)

# 실행
result = blog_chain.run("인공지능")
```

#### Complex Sequential Chain

```python
# 복잡한 순차 체인 (여러 입력/출력)
from langchain.chains import SequentialChain

# 체인 1: 제목 생성
title_chain = LLMChain(
    llm=llm,
    prompt=PromptTemplate(
        input_variables=["topic", "audience"],
        template="{audience}을 위한 {topic} 관련 글 제목을 만들어주세요."
    ),
    output_key="title"
)

# 체인 2: 개요 생성
outline_chain = LLMChain(
    llm=llm,
    prompt=PromptTemplate(
        input_variables=["title", "audience"],
        template="{title}에 대한 {audience} 대상 글의 개요를 작성해주세요."
    ),
    output_key="outline"
)

# 체인 3: 본문 작성
content_chain = LLMChain(
    llm=llm,
    prompt=PromptTemplate(
        input_variables=["title", "outline"],
        template="제목: {title}\n개요: {outline}\n\n위 내용을 바탕으로 전체 글을 작성해주세요."
    ),
    output_key="content"
)

# 전체 체인 구성
full_chain = SequentialChain(
    chains=[title_chain, outline_chain, content_chain],
    input_variables=["topic", "audience"],
    output_variables=["title", "outline", "content"],
    verbose=True
)

# 실행
result = full_chain({
    "topic": "Python 프로그래밍",
    "audience": "초보자"
})
```

### 2.4 Memory (메모리)

**메모리의 필요성과 동작 원리**

LLM은 기본적으로 **상태가 없는(stateless)** 모델입니다. 각 요청을 독립적으로 처리하므로 이전 대화 맥락을 기억하지 못합니다. LangChain의 메모리 컴포넌트는 이러한 한계를 극복하여 **대화형 애플리케이션에서 연속성**을 제공합니다.

**메모리 관리의 도전 과제:**

1. **토큰 제한**: LLM의 컨텍스트 창 크기 제약
2. **비용 최적화**: 긴 대화 기록으로 인한 API 비용 증가
3. **성능**: 대화 기록 검색 및 관리 속도
4. **관련성**: 현재 질문과 관련된 과거 정보만 선별

**메모리 전략별 특징:**

- **ConversationBufferMemory**: 모든 대화 저장 (단순하지만 토큰 급증)
- **ConversationSummaryMemory**: 요약을 통한 압축 (정보 손실 가능)
- **ConversationBufferWindowMemory**: 최근 N개만 보관 (맥락 손실)
- **ConversationSummaryBufferMemory**: 하이브리드 접근 (복잡하지만 효율적)

**프로덕션 환경 고려사항:**

- 사용자별 세션 관리
- 메모리 데이터 영속성 (Redis, Database)
- 민감 정보 자동 삭제
- 대화 품질 모니터링

```python
from langchain.memory import (
    ConversationBufferMemory,
    ConversationSummaryMemory,
    ConversationBufferWindowMemory,
    ConversationSummaryBufferMemory
)
from langchain.chains import ConversationChain

# 1. 기본 버퍼 메모리 (모든 대화 저장)
buffer_memory = ConversationBufferMemory()

conversation = ConversationChain(
    llm=llm,
    memory=buffer_memory,
    verbose=True
)

# 대화 진행
conversation.predict(input="안녕하세요!")
conversation.predict(input="제 이름은 김철수입니다.")
conversation.predict(input="제 이름이 무엇이었나요?")

# 2. 윈도우 메모리 (최근 K개 대화만 저장)
window_memory = ConversationBufferWindowMemory(k=2)

# 3. 요약 메모리 (오래된 대화는 요약)
summary_memory = ConversationSummaryMemory(
    llm=llm,
    max_token_limit=1000
)

# 4. 요약 버퍼 메모리 (혼합형)
summary_buffer_memory = ConversationSummaryBufferMemory(
    llm=llm,
    max_token_limit=1000
)

# 커스텀 메모리 키 사용
custom_memory = ConversationBufferMemory(
    memory_key="chat_history",
    input_key="user_input",
    output_key="assistant_output"
)
```

### 2.5 Agents (에이전트)

```python
from langchain.agents import initialize_agent, AgentType, Tool
from langchain.tools import DuckDuckGoSearchRun, ShellTool
from langchain.utilities import WikipediaAPIWrapper

# 도구 정의
search = DuckDuckGoSearchRun()
wikipedia = WikipediaAPIWrapper()
shell = ShellTool()

tools = [
    Tool(
        name="Search",
        func=search.run,
        description="인터넷에서 최신 정보를 검색할 때 유용합니다."
    ),
    Tool(
        name="Wikipedia",
        func=wikipedia.run,
        description="위키피디아에서 정보를 찾을 때 유용합니다."
    ),
    Tool(
        name="Shell",
        func=shell.run,
        description="쉘 명령어를 실행할 때 사용합니다. 주의해서 사용하세요."
    )
]

# 에이전트 초기화
agent = initialize_agent(
    tools=tools,
    llm=llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True,
    max_iterations=3,
    early_stopping_method="generate"
)

# 에이전트 실행
result = agent.run("2024년 올림픽은 어디에서 개최되나요?")
```

#### 커스텀 도구 생성

```python
from langchain.tools import BaseTool
from typing import Optional, Type
from pydantic import BaseModel, Field

class CalculatorInput(BaseModel):
    expression: str = Field(description="수학 표현식 (예: 2+2*3)")

class CalculatorTool(BaseTool):
    name = "calculator"
    description = "수학 계산을 수행합니다. 입력으로 수학 표현식을 받습니다."
    args_schema: Type[BaseModel] = CalculatorInput

    def _run(self, expression: str) -> str:
        """계산 실행"""
        try:
            result = eval(expression)  # 주의: 실제 환경에서는 안전한 계산기 사용
            return f"{expression} = {result}"
        except Exception as e:
            return f"계산 오류: {str(e)}"

    async def _arun(self, expression: str) -> str:
        """비동기 실행"""
        return self._run(expression)

# 커스텀 도구 사용
calculator = CalculatorTool()
tools_with_calc = tools + [calculator]

math_agent = initialize_agent(
    tools=tools_with_calc,
    llm=llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True
)

result = math_agent.run("15의 제곱근을 구하고, 그 결과에 10을 곱해주세요.")
```

## 3. RAG 구현 with LangChain

### 3.1 기본 RAG 파이프라인

```python
from langchain.document_loaders import TextLoader, PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma, FAISS
from langchain.chains import RetrievalQA
from langchain.llms import OpenAI

class RAGPipeline:
    def __init__(self, openai_api_key):
        self.llm = OpenAI(openai_api_key=openai_api_key)
        self.embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
        self.vectorstore = None
        self.qa_chain = None

    def load_documents(self, file_paths):
        """문서 로드"""
        documents = []

        for file_path in file_paths:
            if file_path.endswith('.pdf'):
                loader = PyPDFLoader(file_path)
            else:
                loader = TextLoader(file_path)

            docs = loader.load()
            documents.extend(docs)

        return documents

    def process_documents(self, documents):
        """문서 전처리 및 청킹"""
        # 텍스트 분할
        texts = self.text_splitter.split_documents(documents)

        # 벡터 저장소 생성
        self.vectorstore = Chroma.from_documents(
            documents=texts,
            embedding=self.embeddings,
            persist_directory="./chroma_db"
        )

        return texts

    def create_qa_chain(self, chain_type="stuff"):
        """QA 체인 생성"""
        if self.vectorstore is None:
            raise ValueError("먼저 문서를 처리해주세요.")

        self.qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type=chain_type,
            retriever=self.vectorstore.as_retriever(
                search_kwargs={"k": 3}  # 상위 3개 문서 검색
            ),
            return_source_documents=True,
            verbose=True
        )

        return self.qa_chain

    def query(self, question):
        """질의응답"""
        if self.qa_chain is None:
            raise ValueError("먼저 QA 체인을 생성해주세요.")

        result = self.qa_chain({"query": question})

        return {
            "answer": result["result"],
            "source_documents": result["source_documents"]
        }

# 사용 예시
rag = RAGPipeline(openai_api_key="your-api-key")

# 문서 로드 및 처리
documents = rag.load_documents(["document1.pdf", "document2.txt"])
texts = rag.process_documents(documents)

# QA 체인 생성
qa_chain = rag.create_qa_chain()

# 질의응답
result = rag.query("문서에서 언급된 주요 개념은 무엇인가요?")
print("답변:", result["answer"])
print("출처 문서 수:", len(result["source_documents"]))
```

### 3.2 고급 RAG 기법

#### Conversational RAG

```python
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory

class ConversationalRAG:
    def __init__(self, vectorstore, llm):
        self.vectorstore = vectorstore
        self.llm = llm

        # 대화 메모리
        self.memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True
        )

        # 대화형 검색 체인
        self.chain = ConversationalRetrievalChain.from_llm(
            llm=self.llm,
            retriever=self.vectorstore.as_retriever(),
            memory=self.memory,
            return_source_documents=True
        )

    def chat(self, question):
        """대화형 질의응답"""
        result = self.chain({"question": question})
        return result

# 사용 예시
conv_rag = ConversationalRAG(vectorstore, llm)

# 연속적인 대화
response1 = conv_rag.chat("문서의 주요 내용을 요약해주세요.")
response2 = conv_rag.chat("방금 언급한 첫 번째 항목에 대해 더 자세히 설명해주세요.")
```

#### Multi-Query RAG

```python
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate

# 다중 쿼리 생성 프롬프트
QUERY_PROMPT = PromptTemplate(
    input_variables=["question"],
    template="""You are an AI language model assistant. Your task is to generate five
    different versions of the given user question to retrieve relevant documents from a vector
    database. By generating multiple perspectives on the user question, your goal is to help
    the user overcome some of the limitations of the distance-based similarity search.
    Provide these alternative questions separated by newlines.
    Original question: {question}"""
)

class MultiQueryRAG:
    def __init__(self, vectorstore, llm):
        self.vectorstore = vectorstore
        self.llm = llm

        # 다중 쿼리 검색기
        self.retriever = MultiQueryRetriever.from_llm(
            retriever=vectorstore.as_retriever(),
            llm=llm,
            prompt=QUERY_PROMPT
        )

        # QA 체인
        self.qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=self.retriever,
            return_source_documents=True
        )

    def query(self, question):
        """다중 쿼리를 사용한 검색"""
        # 생성된 쿼리들 확인
        generated_queries = self.retriever.generate_queries(question)
        print("생성된 쿼리들:")
        for i, query in enumerate(generated_queries, 1):
            print(f"{i}. {query}")

        # 검색 및 답변 생성
        result = self.qa_chain({"query": question})
        return result

# 사용 예시
multi_query_rag = MultiQueryRAG(vectorstore, llm)
result = multi_query_rag.query("머신러닝의 장점은 무엇인가요?")
```

#### Self-Query RAG

```python
from langchain.retrievers.self_query.base import SelfQueryRetriever
from langchain.chains.query_constructor.base import AttributeInfo

# 메타데이터 정보 정의
metadata_field_info = [
    AttributeInfo(
        name="source",
        description="문서의 출처",
        type="string",
    ),
    AttributeInfo(
        name="page",
        description="페이지 번호",
        type="integer",
    ),
    AttributeInfo(
        name="category",
        description="문서 카테고리",
        type="string",
    ),
]

document_content_description = "머신러닝과 AI에 관한 문서"

class SelfQueryRAG:
    def __init__(self, vectorstore, llm):
        self.vectorstore = vectorstore
        self.llm = llm

        # 자체 쿼리 검색기
        self.retriever = SelfQueryRetriever.from_llm(
            llm=llm,
            vectorstore=vectorstore,
            document_contents=document_content_description,
            metadata_field_info=metadata_field_info,
            verbose=True
        )

        # QA 체인
        self.qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=self.retriever,
            return_source_documents=True
        )

    def query(self, question):
        """자체 쿼리를 사용한 검색"""
        result = self.qa_chain({"query": question})
        return result

# 사용 예시 - 필터링이 포함된 쿼리
self_query_rag = SelfQueryRAG(vectorstore, llm)
result = self_query_rag.query("5페이지 이후의 딥러닝에 관한 내용을 찾아주세요.")
```

## 4. 문서 로더와 전처리

### 4.1 다양한 문서 로더

```python
from langchain.document_loaders import (
    TextLoader, PyPDFLoader, UnstructuredPDFLoader,
    Docx2txtLoader, UnstructuredWordDocumentLoader,
    CSVLoader, JSONLoader, UnstructuredHTMLLoader,
    WebBaseLoader, YoutubeLoader
)

class DocumentLoaderFactory:
    @staticmethod
    def get_loader(file_path):
        """파일 확장자에 따른 적절한 로더 반환"""
        if file_path.endswith('.txt'):
            return TextLoader(file_path)
        elif file_path.endswith('.pdf'):
            return PyPDFLoader(file_path)
        elif file_path.endswith('.docx'):
            return Docx2txtLoader(file_path)
        elif file_path.endswith('.csv'):
            return CSVLoader(file_path)
        elif file_path.endswith('.json'):
            return JSONLoader(file_path)
        elif file_path.endswith('.html'):
            return UnstructuredHTMLLoader(file_path)
        else:
            raise ValueError(f"지원하지 않는 파일 형식: {file_path}")

    @staticmethod
    def load_web_content(urls):
        """웹 페이지 로드"""
        loader = WebBaseLoader(urls)
        return loader.load()

    @staticmethod
    def load_youtube_transcript(video_url):
        """유튜브 자막 로드"""
        loader = YoutubeLoader.from_youtube_url(video_url)
        return loader.load()

# 사용 예시
loader_factory = DocumentLoaderFactory()

# 다양한 파일 로드
pdf_docs = loader_factory.get_loader("document.pdf").load()
web_docs = loader_factory.load_web_content(["https://example.com"])
youtube_docs = loader_factory.load_youtube_transcript("https://youtube.com/watch?v=...")
```

### 4.2 고급 텍스트 분할

```python
from langchain.text_splitter import (
    RecursiveCharacterTextSplitter,
    CharacterTextSplitter,
    TokenTextSplitter,
    SpacyTextSplitter,
    NLTKTextSplitter
)

class AdvancedTextSplitter:
    def __init__(self):
        self.splitters = {
            'recursive': RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200,
                separators=["\n\n", "\n", " ", ""]
            ),
            'character': CharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200,
                separator="\n"
            ),
            'token': TokenTextSplitter(
                chunk_size=1000,
                chunk_overlap=200
            ),
            'spacy': SpacyTextSplitter(
                chunk_size=1000,
                chunk_overlap=200
            ),
            'nltk': NLTKTextSplitter(
                chunk_size=1000,
                chunk_overlap=200
            )
        }

    def split_by_content_type(self, documents):
        """문서 유형별 최적화된 분할"""
        results = {}

        for doc in documents:
            content_type = self._detect_content_type(doc.page_content)

            if content_type == 'code':
                # 코드는 토큰 기반 분할
                splitter = self.splitters['token']
            elif content_type == 'academic':
                # 학술 논문은 SpaCy 분할
                splitter = self.splitters['spacy']
            else:
                # 일반 텍스트는 재귀적 분할
                splitter = self.splitters['recursive']

            chunks = splitter.split_documents([doc])
            results[content_type] = results.get(content_type, []) + chunks

        return results

    def _detect_content_type(self, text):
        """텍스트 유형 감지"""
        code_indicators = ['def ', 'class ', 'import ', 'function', 'var ', 'const ']
        academic_indicators = ['abstract', 'introduction', 'methodology', 'references']

        if any(indicator in text.lower() for indicator in code_indicators):
            return 'code'
        elif any(indicator in text.lower() for indicator in academic_indicators):
            return 'academic'
        else:
            return 'general'

# 시맨틱 청킹
class SemanticTextSplitter:
    def __init__(self, embedding_model, similarity_threshold=0.7):
        self.embedding_model = embedding_model
        self.similarity_threshold = similarity_threshold

    def split_text_semantically(self, text, base_chunk_size=500):
        """의미적 유사성 기반 텍스트 분할"""
        from sklearn.metrics.pairwise import cosine_similarity

        # 문장 단위로 분할
        sentences = text.split('. ')

        # 각 문장의 임베딩 계산
        sentence_embeddings = self.embedding_model.encode(sentences)

        chunks = []
        current_chunk = []
        current_chunk_embedding = None

        for i, sentence in enumerate(sentences):
            sentence_embedding = sentence_embeddings[i:i+1]

            if current_chunk_embedding is None:
                # 첫 번째 문장
                current_chunk = [sentence]
                current_chunk_embedding = sentence_embedding
            else:
                # 현재 청크와 유사도 계산
                similarity = cosine_similarity(
                    current_chunk_embedding.reshape(1, -1),
                    sentence_embedding.reshape(1, -1)
                )[0][0]

                if similarity >= self.similarity_threshold and len(' '.join(current_chunk)) < base_chunk_size * 2:
                    # 유사도가 높고 청크 크기가 적절하면 추가
                    current_chunk.append(sentence)
                    # 청크 임베딩 업데이트 (평균)
                    chunk_text = ' '.join(current_chunk)
                    current_chunk_embedding = self.embedding_model.encode([chunk_text])
                else:
                    # 새로운 청크 시작
                    chunks.append('. '.join(current_chunk))
                    current_chunk = [sentence]
                    current_chunk_embedding = sentence_embedding

        # 마지막 청크 추가
        if current_chunk:
            chunks.append('. '.join(current_chunk))

        return chunks
```

## 5. 고급 체인 패턴

### 5.1 MapReduce 체인

```python
from langchain.chains.mapreduce import MapReduceChain
from langchain.chains.combine_documents.map_reduce import MapReduceDocumentsChain
from langchain.chains.combine_documents.stuff import StuffDocumentsChain

class MapReduceRAG:
    def __init__(self, llm):
        self.llm = llm
        self._setup_chains()

    def _setup_chains(self):
        """Map-Reduce 체인 설정"""
        # Map 단계: 각 문서를 개별 처리
        map_template = """다음 문서를 요약해주세요:
        {docs}

        간결한 요약:"""

        map_prompt = PromptTemplate.from_template(map_template)
        map_chain = LLMChain(llm=self.llm, prompt=map_prompt)

        # Reduce 단계: 요약들을 종합
        reduce_template = """다음은 여러 문서의 요약들입니다:
        {doc_summaries}

        이들을 종합하여 전체적인 요약을 작성해주세요:"""

        reduce_prompt = PromptTemplate.from_template(reduce_template)
        reduce_chain = LLMChain(llm=self.llm, prompt=reduce_prompt)

        # 문서 결합 체인
        combine_documents_chain = StuffDocumentsChain(
            llm_chain=reduce_chain,
            document_variable_name="doc_summaries"
        )

        # 전체 MapReduce 체인
        self.map_reduce_chain = MapReduceDocumentsChain(
            llm_chain=map_chain,
            combine_document_chain=combine_documents_chain,
            document_variable_name="docs",
            return_intermediate_steps=True,
        )

    def summarize_documents(self, documents):
        """문서들을 MapReduce 방식으로 요약"""
        result = self.map_reduce_chain({"input_documents": documents})

        return {
            "final_summary": result["output_text"],
            "intermediate_summaries": result["intermediate_steps"]
        }

# 사용 예시
map_reduce_rag = MapReduceRAG(llm)
result = map_reduce_rag.summarize_documents(documents)
```

### 5.2 조건부 체인

```python
from langchain.chains.router import MultiPromptChain
from langchain.chains.router.llm_router import LLMRouterChain, RouterOutputParser
from langchain.chains.router.multi_prompt_prompt import MULTI_PROMPT_ROUTER_TEMPLATE

class ConditionalRAG:
    def __init__(self, llm):
        self.llm = llm
        self._setup_routing_chains()

    def _setup_routing_chains(self):
        """라우팅 체인 설정"""

        # 기술 문서용 체인
        tech_template = """당신은 기술 전문가입니다. 다음 기술 관련 질문에 정확하고 상세하게 답변해주세요:

        {input}

        답변:"""

        # 일반 문서용 체인
        general_template = """당신은 도움이 되는 어시스턴트입니다. 다음 질문에 친근하고 이해하기 쉽게 답변해주세요:

        {input}

        답변:"""

        # 학술 문서용 체인
        academic_template = """당신은 학술 연구자입니다. 다음 학술적 질문에 엄밀하고 체계적으로 답변해주세요:

        {input}

        답변:"""

        # 프롬프트 정보 정의
        prompt_infos = [
            {
                "name": "tech",
                "description": "기술, 프로그래밍, 소프트웨어 개발에 관한 질문",
                "prompt_template": tech_template
            },
            {
                "name": "general",
                "description": "일반적인 생활, 상식, 기본적인 질문",
                "prompt_template": general_template
            },
            {
                "name": "academic",
                "description": "학술적, 연구 관련, 이론적인 질문",
                "prompt_template": academic_template
            }
        ]

        # 체인들 생성
        destination_chains = {}
        for p_info in prompt_infos:
            name = p_info["name"]
            prompt_template = p_info["prompt_template"]
            prompt = PromptTemplate(template=prompt_template, input_variables=["input"])
            chain = LLMChain(llm=self.llm, prompt=prompt)
            destination_chains[name] = chain

        # 기본 체인
        default_prompt = PromptTemplate(template=general_template, input_variables=["input"])
        default_chain = LLMChain(llm=self.llm, prompt=default_prompt)

        # 라우터 체인
        destinations = [f"{p['name']}: {p['description']}" for p in prompt_infos]
        destinations_str = "\n".join(destinations)
        router_template = MULTI_PROMPT_ROUTER_TEMPLATE.format(destinations=destinations_str)
        router_prompt = PromptTemplate(
            template=router_template,
            input_variables=["input"],
            output_parser=RouterOutputParser(),
        )
        router_chain = LLMRouterChain.from_llm(self.llm, router_prompt)

        # 멀티 프롬프트 체인
        self.chain = MultiPromptChain(
            router_chain=router_chain,
            destination_chains=destination_chains,
            default_chain=default_chain,
            verbose=True
        )

    def query(self, question):
        """조건부 라우팅을 통한 질의응답"""
        return self.chain.run(question)

# 사용 예시
conditional_rag = ConditionalRAG(llm)

# 다양한 유형의 질문
tech_question = "Python에서 비동기 프로그래밍을 어떻게 구현하나요?"
general_question = "좋은 아침 식사 메뉴를 추천해주세요."
academic_question = "양자역학의 불확정성 원리에 대해 설명해주세요."

print("기술 질문:", conditional_rag.query(tech_question))
print("일반 질문:", conditional_rag.query(general_question))
print("학술 질문:", conditional_rag.query(academic_question))
```

## 6. 성능 최적화 및 모니터링

### 6.1 캐싱 전략

```python
from langchain.cache import InMemoryCache, SQLiteCache
from langchain.globals import set_llm_cache
import sqlite3

class LangChainCacheManager:
    def __init__(self, cache_type="memory"):
        self.cache_type = cache_type
        self._setup_cache()

    def _setup_cache(self):
        """캐시 설정"""
        if self.cache_type == "memory":
            set_llm_cache(InMemoryCache())
        elif self.cache_type == "sqlite":
            set_llm_cache(SQLiteCache(database_path=".langchain.db"))

    def clear_cache(self):
        """캐시 초기화"""
        if self.cache_type == "sqlite":
            import os
            if os.path.exists(".langchain.db"):
                os.remove(".langchain.db")
                self._setup_cache()

# 커스텀 캐시
class RedisCache:
    def __init__(self, redis_client):
        self.redis_client = redis_client

    def lookup(self, prompt, llm_string):
        """캐시에서 결과 조회"""
        key = self._generate_key(prompt, llm_string)
        result = self.redis_client.get(key)
        return result.decode() if result else None

    def update(self, prompt, llm_string, return_val):
        """캐시에 결과 저장"""
        key = self._generate_key(prompt, llm_string)
        self.redis_client.setex(key, 3600, return_val)  # 1시간 TTL

    def _generate_key(self, prompt, llm_string):
        """캐시 키 생성"""
        import hashlib
        content = f"{prompt}:{llm_string}"
        return f"langchain:{hashlib.md5(content.encode()).hexdigest()}"
```

### 6.2 비동기 처리

```python
import asyncio
from langchain.llms import OpenAI
from langchain.chains import LLMChain
from langchain.callbacks import AsyncCallbackHandler

class AsyncRAGPipeline:
    def __init__(self, api_key):
        self.llm = OpenAI(openai_api_key=api_key)
        self.chains = {}

    async def process_multiple_queries(self, queries):
        """여러 쿼리를 비동기로 처리"""
        tasks = []
        for query in queries:
            task = self._async_query(query)
            tasks.append(task)

        results = await asyncio.gather(*tasks)
        return results

    async def _async_query(self, query):
        """단일 쿼리 비동기 처리"""
        try:
            chain = LLMChain(llm=self.llm, prompt=self._get_prompt())
            result = await chain.arun(query=query)
            return {"query": query, "result": result, "status": "success"}
        except Exception as e:
            return {"query": query, "error": str(e), "status": "error"}

    def _get_prompt(self):
        """프롬프트 템플릿 반환"""
        return PromptTemplate(
            input_variables=["query"],
            template="다음 질문에 답변해주세요: {query}"
        )

# 콜백을 통한 모니터링
class RAGMonitoringCallback(AsyncCallbackHandler):
    def __init__(self):
        self.start_time = None
        self.metrics = []

    async def on_llm_start(self, serialized, prompts, **kwargs):
        """LLM 시작 시"""
        import time
        self.start_time = time.time()
        print(f"LLM 호출 시작: {prompts[0][:50]}...")

    async def on_llm_end(self, response, **kwargs):
        """LLM 완료 시"""
        import time
        duration = time.time() - self.start_time
        self.metrics.append({
            "duration": duration,
            "tokens": len(response.generations[0][0].text.split())
        })
        print(f"LLM 호출 완료 ({duration:.2f}초)")

    async def on_llm_error(self, error, **kwargs):
        """LLM 오류 시"""
        print(f"LLM 오류: {error}")

# 사용 예시
async def main():
    pipeline = AsyncRAGPipeline("your-api-key")
    callback = RAGMonitoringCallback()

    queries = [
        "Python이란 무엇인가요?",
        "머신러닝의 장점은?",
        "딥러닝과 머신러닝의 차이점은?"
    ]

    results = await pipeline.process_multiple_queries(queries)

    for result in results:
        if result["status"] == "success":
            print(f"Q: {result['query']}")
            print(f"A: {result['result']}")
        else:
            print(f"오류: {result['query']} - {result['error']}")

# 실행
asyncio.run(main())
```

### 6.3 성능 메트릭

```python
import time
import psutil
import threading
from functools import wraps

class RAGPerformanceMonitor:
    def __init__(self):
        self.metrics = {
            "query_count": 0,
            "total_response_time": 0,
            "average_response_time": 0,
            "error_count": 0,
            "memory_usage": [],
            "cpu_usage": []
        }
        self.monitoring = False
        self.monitor_thread = None

    def start_monitoring(self):
        """시스템 리소스 모니터링 시작"""
        self.monitoring = True
        self.monitor_thread = threading.Thread(target=self._monitor_resources)
        self.monitor_thread.start()

    def stop_monitoring(self):
        """모니터링 중지"""
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join()

    def _monitor_resources(self):
        """리소스 사용량 모니터링"""
        while self.monitoring:
            memory_percent = psutil.virtual_memory().percent
            cpu_percent = psutil.cpu_percent()

            self.metrics["memory_usage"].append(memory_percent)
            self.metrics["cpu_usage"].append(cpu_percent)

            time.sleep(1)  # 1초마다 측정

    def track_query(self, func):
        """쿼리 성능 추적 데코레이터"""
        @wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.time()

            try:
                result = func(*args, **kwargs)

                # 성공 메트릭 업데이트
                response_time = time.time() - start_time
                self.metrics["query_count"] += 1
                self.metrics["total_response_time"] += response_time
                self.metrics["average_response_time"] = (
                    self.metrics["total_response_time"] / self.metrics["query_count"]
                )

                return result

            except Exception as e:
                # 오류 메트릭 업데이트
                self.metrics["error_count"] += 1
                raise e

        return wrapper

    def get_performance_report(self):
        """성능 리포트 생성"""
        import numpy as np

        report = {
            "쿼리 처리 통계": {
                "총 쿼리 수": self.metrics["query_count"],
                "평균 응답 시간": f"{self.metrics['average_response_time']:.2f}초",
                "오류 수": self.metrics["error_count"],
                "성공률": f"{(1 - self.metrics['error_count'] / max(1, self.metrics['query_count'])) * 100:.1f}%"
            },
            "시스템 리소스": {
                "평균 메모리 사용량": f"{np.mean(self.metrics['memory_usage']):.1f}%",
                "최대 메모리 사용량": f"{np.max(self.metrics['memory_usage']):.1f}%",
                "평균 CPU 사용량": f"{np.mean(self.metrics['cpu_usage']):.1f}%",
                "최대 CPU 사용량": f"{np.max(self.metrics['cpu_usage']):.1f}%"
            }
        }

        return report

# 사용 예시
monitor = RAGPerformanceMonitor()
monitor.start_monitoring()

@monitor.track_query
def rag_query(question):
    # RAG 쿼리 실행
    time.sleep(2)  # 시뮬레이션
    return f"{question}에 대한 답변"

# 여러 쿼리 실행
for i in range(10):
    result = rag_query(f"질문 {i}")

monitor.stop_monitoring()
report = monitor.get_performance_report()
print(report)
```

LangChain은 RAG 시스템을 포함한 다양한 LLM 기반 애플리케이션 개발을 위한 강력한 프레임워크입니다. 모듈형 설계와 풍부한 기능을 통해 복잡한 AI 워크플로우를 효율적으로 구현할 수 있습니다.
