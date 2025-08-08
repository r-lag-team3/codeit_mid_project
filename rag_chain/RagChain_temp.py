from langchain.embeddings import OpenAIEmbeddings
from langchain.schema import Document
from langchain.vectorstores import FAISS
from openai import OpenAI
from typing import List, Optional, Callable
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from transformers import AutoTokenizer, AutoModelForCausalLM
from sentence_transformers import CrossEncoder
from collections import deque
from keybert import KeyBERT
import torch
import re
import os


class KeywordMemory:
    """
    이전 대화 키워드 메모리 관리 클래스
    """
    def __init__(self, max_history: int = 10):
        self.keywords = deque(maxlen=max_history)  # 각 턴 키워드 리스트 저장

    def add_keywords(self, new_keywords: List[str]):
        self.keywords.append(new_keywords)

    def get_all_keywords(self) -> List[str]:
        # 저장된 모든 키워드 플랫 리스트, 중복 제거
        flat = [kw for kws in self.keywords for kw in kws]
        return list(set(flat))

    def __repr__(self):
        return f"[Keyword Memory] 최근 키워드: {self.get_all_keywords()}"


class RAGChain:
    def __init__(self, retriever: Callable, model_name: str = "gpt-4.1-mini", top_k: int = 10):
        """
        업그레이드된 RAG Chain 클래스
        
        Args:
            retriever: 검색을 수행하는 retriever 함수
            model_name: LLM 모델 이름
            openai_api_key: OpenAI API 키
            top_k: rerank에서 사용할 상위 k개 문서 개수
        """
        self.retriever = retriever
        if model_name.startswith("gpt"):
            self.llm = ChatOpenAI(model_name=model_name, temperature=0.3)
            self.openai = True
        else:
            self.model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype= torch.bfloat16).to("cuda")
            self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
            self.openai = False
        self.top_k = top_k

        # 이전 대화 키워드 기억용
        self.keyword_memory = KeywordMemory()
        # 한글 키워드 추출기 (KeyBERT)
        self.keyword_extractor = KeyBERT(model="distiluse-base-multilingual-cased-v2")
        self.cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
        
        # 기본 프롬프트 템플릿
        template = """
        당신은 문서를 기반으로 답변하는 AI입니다.
        이전 대화 키워드: {past_keywords}
        
        문서에 있는 내용만을 사용하여 답변하세요.
        문서가 충분하지 않으면 "문서에는 내용이 부족합니다."라고 답변하세요.
        답변에는 반드시 관련 문서의 근거(문장 또는 항목 번호 등)를 명시하세요.
        문서의 목적, 요구사항, 평가기준, 제출조건 등 RFP의 주요 항목을 우선적으로 참고하세요.
        질문이 모호하거나 여러 해석이 가능하면, 가능한 해석을 모두 제시하고 각각에 대해 답변하세요.
        단 문서에 명시되지 않은 법령명, 조항 번호, 기관명, 다국어 문장 등은 절대 포함하지 마십시오.

        다음 문서를 참고하여 질문에 답하세요.

        문서:
        {context}

        질문:
        {query}

        답변:
        """
        self.prompt = PromptTemplate(
            template=template,
            input_variables=["past_keywords", "context", "query"]
        )
    def has_information(self, query: str, docs: List[Document], model: str = "gpt-4.1-mini") -> bool:
        """
        OpenAI LLM을 사용하여 문서에 질문에 대한 정보가 있는지 판단
        """

        # 문서 내용 결합 (최대 10개)
        context = "\n\n".join(doc.page_content for doc in docs[:10])

        # 시스템 메시지와 유저 메시지를 분리하여 명시
        messages = [
            {
                "role": "system",
                "content": "문서에 질문에 대한 답이 있는지 판단하는 AI입니다. '있음' 또는 '없음'으로만 응답하세요.",
            },
            {
                "role": "user",
                "content": f"""
                다음 문서에 질문에 대한 직접적인 정보나 답변이 포함되어 있습니까?
                반드시 '있음' 또는 '없음'으로만 답하십시오.

                문서:
                {context}

                질문:
                {query}

                답:
                """,
            },
        ]

        try:
            client = OpenAI()  # 환경변수에서 OPENAI_API_KEY 로드
            res = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=0.0,
                max_tokens=5,
            )
            answer = res.choices[0].message.content.strip()
            return "있음" in answer
        except Exception as e:
            print(f"[정보 유무 판단 실패] {e}")
            return True  # 오류 발생 시 기본적으로 True 처리
        
    def extract_keywords(self, text: str, top_n: int = 5) -> List[str]:
        """
        입력 텍스트에서 주요 키워드 추출
        
        Args:
            text: 텍스트
            top_n: 추출할 키워드 개수
            
        Returns:
            키워드 리스트
        """
        try:
            keywords = self.keyword_extractor.extract_keywords(text, top_n=top_n, stop_words=None)
            return [kw for kw, _ in keywords]
        except Exception as e:
            print(f"키워드 추출 오류: {e}")
            return []

    def summarize_chunks(self, docs: List[Document], question: str, past_keywords_text: str,
                         model: str = "gpt-4.1-mini") -> str:
        """
        LLM 기반 3줄 요약(Map-Reduce) - 이전 키워드 문맥 포함
        
        Args:
            docs: 문서 리스트
            question: 질문
            past_keywords_text: 이전 대화 키워드를 문장 형태로 변환한 텍스트
            model: LLM 모델 이름
            
        Returns:
            요약된 텍스트
        """
        try:
            client = OpenAI()
            summaries = []

            for doc in docs:
                messages = [
                    {
                        "role": "system",
                        "content": "당신은 주어진 문서와 질문, 그리고 과거 대화 키워드를 기반으로 3줄 요약을 생성하는 요약 AI입니다.",
                    },
                    {
                        "role": "user",
                        "content": f"이전 대화 키워드: {past_keywords_text}\n\n다음 내용을 질문 관점에서 3줄로 요약해 주세요.\n\n질문: {question}\n\n문서:\n{doc.page_content}",
                    }
                ]

                response = client.chat.completions.create(
                    model=model,
                    messages=messages,
                    temperature=0.5,
                    max_tokens=512,
                )
                summaries.append(response.choices[0].message.content.strip())

            # Reduce 요약
            reduce_messages = [
                {
                    "role": "system",
                    "content": "당신은 여러 개의 요약을 종합하여 최종 요약을 만드는 요약 AI입니다.",
                },
                {
                    "role": "user",
                    "content": f"이전 대화 키워드: {past_keywords_text}\n\n아래 요약들을 종합해서 3줄로 다시 요약해 주세요:\n\n{chr(10).join(summaries)}",
                }
            ]

            final_response = client.chat.completions.create(
                model=model,
                messages=reduce_messages,
                temperature=0.5,
                max_tokens=512,
            )
            return final_response.choices[0].message.content.strip()

        except Exception as e:
            print(f"[OpenAI 요약 실패] {e}")
            return self._basic_summarize_with_keywords(docs, question, past_keywords_text)

    def _basic_summarize_with_keywords(self, docs: List[Document], question: str, past_keywords_text: str) -> str:
        """
        기본 요약 방식 (OpenAI API 없이) - 이전 키워드 문맥 포함
        
        Args:
            docs: 문서 리스트
            question: 질문
            past_keywords_text: 이전 대화 키워드 텍스트
            
        Returns:
            요약된 텍스트
        """
        context = "\n\n".join([doc.page_content for doc in docs])
        filled_prompt = self.prompt.format(past_keywords=past_keywords_text, context=context, query=question)

        if not self.has_information(question, docs):
            return "탐색된 문서에 해당 정보가 없습니다."
        
        if self.openai:
            response = self.llm.invoke(filled_prompt)
            return response.content if hasattr(response, "content") else response
        else:
            inputs = self.tokenizer(filled_prompt, return_tensors="pt").to("cuda")
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=256,
                temperature=0.2,
                do_sample=True,
                top_p=0.8,
                top_k=30,
                repetition_penalty=1.2,
                no_repeat_ngram_size=3,
                pad_token_id=self.tokenizer.eos_token_id,
            )
            decoded = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            # '답변:' 뒤부터만 잘라내는 후처리 추천
            return decoded.split("답변:")[-1].strip()

    def extract_doc_keyword(self, query: str) -> str:
        """
        질문에서 문서명 키워드(명사구)를 정규표현식으로 추출
        예: '한국해양조사협회의', '한국해양조사협회에서', '한국해양조사협회 프로젝트' 등
        
        Args:
            query: 질문 문자열
            
        Returns:
            추출된 문서명 키워드
        """
        # 조사/구분자 패턴: '의', '에서', '에 대한', '에 관한', '프로젝트', '요구사항', '정의', '관리', '보고', '에'
        pattern = r"([가-힣A-Za-z0-9\(\)]+)(?:의|에서|에 대한|에 관한|프로젝트|요구사항|정의|관리|보고|에)"
        m = re.search(pattern, query)
        if m:
            return m.group(1)
        # fallback: 첫 단어
        return query.split()[0]
    def rerank_by_distance(self, query: str, docs: List[Document], top_k: int = 5) -> list[Document]:
        """
        CrossEncoder를 활용하여 query와 문서 간의 relevance score 기반 rerank
        
        Args:
            query: 사용자 질문
            docs: 필터링된 Document 리스트
            top_k: 상위 몇 개 문서를 반환할지
            
        Returns:
            점수가 높은 상위 top_k Document 리스트
        """
        # 1. (query, doc) 쌍 만들기
        pairs = [(query, doc.page_content) for doc in docs]
        
        # 2. 스코어 계산
        scores = self.cross_encoder.predict(pairs)

        # 3. 문서와 스코어 결합 후 상위 top_k 선택
        ranked = sorted(zip(docs, scores), key=lambda x: x[1], reverse=True)
        
        return [doc for doc, _ in ranked[:top_k]]

    def rag_pipeline(self, query: str, keywords: Optional[List[str]] = None) -> dict:
        """
        전체 RAG 파이프라인
        
        Args:
            query: 질문
            chunks: 문서 청크들
            embedding_model: 임베딩 모델
            keywords: 사용자가 입력한 필터링 키워드 (현재는 필터링에 사용 안 함)
            
        Returns:
            최종 답변
        """

        # 1. retriever로 top_k개 후보를 먼저 뽑는다
        candidates = self.retriever(query)
        docs = [chunk.get('doc') for chunk in candidates]

        # 2. 질문에서 문서명 키워드 추출 (명사구 기반)
        doc_keyword = self.extract_doc_keyword(query)
        print("doc_keyword:", doc_keyword)

        # --- 전체 docs에서 파일명 기준으로 필터링 ---
        def get_filename(source):
            return os.path.splitext(os.path.basename(source))[0]
        
        all_docs = getattr(self, 'all_docs', docs)  # self.all_docs가 있으면 사용, 없으면 기존 docs 사용
        filtered_docs = [doc for doc in all_docs if doc_keyword in get_filename(doc.metadata.get('source', ''))]
        if not filtered_docs:
            filtered_docs = docs  # 없으면 전체 사용
        print("filtered_docs sources:", [doc.metadata.get('source', '') for doc in filtered_docs])

        # --- filtered_docs에서 임베딩 기반 top_k 재검색 ---
        # 기존: summary = self.summarize_chunks(filtered_docs[:self.top_k], query, past_keywords_text)
        # 개선: top_k개가 넘으면 임베딩 기반 재검색
        if len(filtered_docs) > self.top_k:
            filtered_docs = self.rerank_by_distance(query, filtered_docs, top_k=self.top_k)
        else:
            filtered_docs = filtered_docs[:self.top_k]

        # 현재 질문에서 키워드 추출하여 기억에 저장
        extracted_keywords = self.extract_keywords(query)
        self.keyword_memory.add_keywords(extracted_keywords)

        # 이전 대화 키워드를 텍스트 문맥으로 준비
        past_keywords_list = self.keyword_memory.get_all_keywords()
        past_keywords_text = ", ".join(past_keywords_list) if past_keywords_list else "없음"

        # LLM 요약 및 답변 생성 (이전 키워드 문맥 포함)
        if self.openai:
            summary = self.summarize_chunks(filtered_docs[:self.top_k], query, past_keywords_text)
        else:
            # OpenAI가 아닌 경우 기본 요약 방식 사용
            summary = self._basic_summarize_with_keywords(filtered_docs[:self.top_k], query, past_keywords_text)
            
        return {
            'response': summary,
            'retrieved_docs': filtered_docs[:self.top_k]
        }

    def query(self, query: str, keywords: Optional[List[str]] = None) -> dict:
        """
        질의 처리 메인 함수 (retriever 기반, 문서명 필터링 포함)
        
        Args:
            query: 질문
            keywords: 필터링 키워드 (현재는 사용 안 함)
        Returns:
            dict 형태 결과
                - query(str): 입력 질문
                - response(str): 답변
        """
        result = self.rag_pipeline(query, keywords)
        return {
            'query': query,
            'response': result['response'],
            'retrieved_docs': result['retrieved_docs'],
        }
