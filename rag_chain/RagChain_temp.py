import numpy as np
import faiss
from langchain.embeddings import OpenAIEmbeddings
from langchain.schema import Document
from langchain.vectorstores import FAISS
from sentence_transformers import CrossEncoder
import openai
from typing import List, Optional, Callable
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from collections import deque
from keybert import KeyBERT
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
    def __init__(self, retriever: Callable, model_name: str = "gpt-4.1-mini", openai_api_key: str = None, top_k: int = 10):
        """
        업그레이드된 RAG Chain 클래스
        
        Args:
            retriever: 검색을 수행하는 retriever 함수
            model_name: LLM 모델 이름
            openai_api_key: OpenAI API 키
            top_k: rerank에서 사용할 상위 k개 문서 개수
        """
        self.retriever = retriever
        self.llm = ChatOpenAI(model_name=model_name, temperature=0.3)
        self.openai_api_key = openai_api_key
        self.top_k = top_k

        # 이전 대화 키워드 기억용
        self.keyword_memory = KeywordMemory()
        # 한글 키워드 추출기 (KeyBERT)
        self.keyword_extractor = KeyBERT(model="distiluse-base-multilingual-cased-v2")
        
        # 기본 프롬프트 템플릿
        template = """
        당신은 문서를 기반으로 답변하는 AI입니다.
        이전 대화 키워드: {past_keywords}
        
        문서에 있는 내용만을 사용하여 답변하세요.
        문서가 충분하지 않으면 "문서에는 내용이 부족합니다."라고 답변하세요.
        답변에는 반드시 관련 문서의 근거(문장 또는 항목 번호 등)를 명시하세요.
        문서의 목적, 요구사항, 평가기준, 제출조건 등 RFP의 주요 항목을 우선적으로 참고하세요.
        질문이 모호하거나 여러 해석이 가능하면, 가능한 해석을 모두 제시하고 각각에 대해 답변하세요.


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

    def semantic_search(self, query: str, chunks: List[Document], embedding_model, k: int = None) -> List[Document]:
        """
        FAISS 기반 semantic search (re-rank 미사용)
        
        Args:
            query: 검색 쿼리
            chunks: 문서 청크들
            embedding_model: 임베딩 모델
            k: 검색할 문서 개수 (None이면 self.top_k 사용)
            
        Returns:
            검색된 문서 리스트
        """
        if k is None:
            k = self.top_k
        db = FAISS.from_documents(chunks, embedding_model)
        retriever = db.as_retriever(search_kwargs={"k": k})
        return retriever(query)

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
        if not self.openai_api_key:
            # OpenAI API 키가 없는 경우 기본 방식 사용
            return self._basic_summarize_with_keywords(docs, question, past_keywords_text)
        
        try:
            openai.api_key = self.openai_api_key
            summaries = []
            for doc in docs:
                prompt = f"이전 대화 키워드: {past_keywords_text}\n다음 내용을 3줄로 요약해 주세요. 질문: {question}\n\n{doc.page_content}"
                response = openai.ChatCompletion.create(
                    model=model,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=512,
                    temperature=0.5,
                )
                summary = response['choices'][0]['message']['content']
                summaries.append(summary)
            
            # Reduce: 전체 요약
            final_prompt = f"이전 대화 키워드: {past_keywords_text}\n아래 요약들을 종합해서 3줄로 다시 요약해 주세요.\n\n" + "\n\n".join(summaries)
            final_response = openai.ChatCompletion.create(
                model=model,
                messages=[{"role": "user", "content": final_prompt}],
                max_tokens=512,
                temperature=0.5,
            )
            return final_response['choices'][0]['message']['content']
        except Exception as e:
            print(f"OpenAI 요약 중 오류 발생: {e}")
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
        response = self.llm.invoke(filled_prompt)
        return response.content

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
            from langchain.vectorstores import FAISS
            from langchain.embeddings import OpenAIEmbeddings
            # 임베딩 모델: self.llm.embeddings가 있으면 사용, 없으면 OpenAIEmbeddings()
            embedding_model = getattr(self.llm, "embeddings", None) or OpenAIEmbeddings()
            db = FAISS.from_documents(filtered_docs, embedding_model)
            retriever = db.as_retriever(search_kwargs={"k": self.top_k})
            # retriever는 함수가 아니라 객체이므로 get_relevant_documents 사용
            filtered_docs = retriever.get_relevant_documents(query)
            # retriever 반환값이 Document 리스트가 아닐 경우 변환
            if filtered_docs and isinstance(filtered_docs[0], dict) and 'doc' in filtered_docs[0]:
                filtered_docs = [item['doc'] for item in filtered_docs]
        # else:
        #     filtered_docs = filtered_docs[:self.top_k]

        # 현재 질문에서 키워드 추출하여 기억에 저장
        extracted_keywords = self.extract_keywords(query)
        self.keyword_memory.add_keywords(extracted_keywords)

        # 이전 대화 키워드를 텍스트 문맥으로 준비
        past_keywords_list = self.keyword_memory.get_all_keywords()
        past_keywords_text = ", ".join(past_keywords_list) if past_keywords_list else "없음"

        # LLM 요약 및 답변 생성 (이전 키워드 문맥 포함)
        summary = self.summarize_chunks(filtered_docs[:self.top_k], query, past_keywords_text)
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
