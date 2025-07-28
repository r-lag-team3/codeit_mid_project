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


class UpgradedRAGChain:
    def __init__(self, retriever: Callable, model_name: str = "gpt-4.1-mini", openai_api_key: str = None):
        """
        업그레이드된 RAG Chain 클래스
        
        Args:
            retriever: 검색을 수행하는 retriever 함수
            model_name: LLM 모델 이름
            openai_api_key: OpenAI API 키
        """
        self.retriever = retriever
        self.llm = ChatOpenAI(model_name=model_name, temperature=0.3)
        self.openai_api_key = openai_api_key
        
        # 기본 프롬프트 템플릿
        template = """
        당신은 문서를 기반으로 답변하는 AI입니다.
        다음 문서를 참고하여 질문에 답하세요.

        문서:
        {context}

        질문:
        {query}

        답변:
        """
        self.prompt = PromptTemplate(
            template=template,
            input_variables=["context", "query"]
        )

    def semantic_search(self, query: str, chunks: List[Document], embedding_model, k: int = 50) -> List[Document]:
        """
        1차 Recall - FAISS 기반 semantic search
        
        Args:
            query: 검색 쿼리
            chunks: 문서 청크들
            embedding_model: 임베딩 모델
            k: 검색할 문서 개수
            
        Returns:
            검색된 문서 리스트
        """
        db = FAISS.from_documents(chunks, embedding_model)
        retriever = db.as_retriever(search_kwargs={"k": k})
        return retriever(query)

    def keyword_filter(self, docs: List[Document], keywords: Optional[List[str]]) -> List[Document]:
        """
        키워드 필터 함수
        
        Args:
            docs: 문서 리스트
            keywords: 필터링할 키워드 리스트
            
        Returns:
            필터링된 문서 리스트
        """
        if not keywords:
            return docs
        filtered = []
        for doc in docs:
            if any(kw.lower() in doc.page_content.lower() for kw in keywords):
                filtered.append(doc)
        return filtered

    def rerank(self, query: str, docs: List[Document], top_k: int = 10, 
               model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2") -> List[Document]:
        """
        Cross-Encoder 기반 Re-Ranking
        
        Args:
            query: 검색 쿼리
            docs: 문서 리스트
            top_k: 상위 k개 문서 선택
            model_name: Cross-Encoder 모델 이름
            
        Returns:
            재순위화된 문서 리스트
        """
        try:
            cross_encoder = CrossEncoder(model_name)
            pairs = [[query, doc.page_content] for doc in docs]
            scores = cross_encoder.predict(pairs)
            scored_docs = sorted(zip(docs, scores), key=lambda x: x[1], reverse=True)
            return [doc for doc, score in scored_docs[:top_k]]
        except Exception as e:
            print(f"Cross-Encoder 재순위화 중 오류 발생: {e}")
            print("원본 문서 순서를 유지합니다.")
            return docs[:top_k]

    def summarize_chunks(self, docs: List[Document], question: str, 
                        model: str = "gpt-4.1-mini") -> str:
        """
        LLM 기반 3줄 요약(Map-Reduce)
        
        Args:
            docs: 문서 리스트
            question: 질문
            model: LLM 모델 이름
            
        Returns:
            요약된 텍스트
        """
        if not self.openai_api_key:
            # OpenAI API 키가 없는 경우 기본 방식 사용
            return self._basic_summarize(docs, question)
        
        try:
            openai.api_key = self.openai_api_key
            summaries = []
            for doc in docs:
                prompt = f"다음 내용을 3줄로 요약해 주세요. 질문: {question}\n\n{doc.page_content}"
                response = openai.ChatCompletion.create(
                    model=model,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=256,
                    temperature=0.5,
                )
                summary = response['choices'][0]['message']['content']
                summaries.append(summary)
            
            # Reduce: 전체 요약
            final_prompt = f"아래 요약들을 종합해서 3줄로 다시 요약해 주세요.\n\n" + "\n\n".join(summaries)
            final_response = openai.ChatCompletion.create(
                model=model,
                messages=[{"role": "user", "content": final_prompt}],
                max_tokens=256,
                temperature=0.5,
            )
            return final_response['choices'][0]['message']['content']
        except Exception as e:
            print(f"OpenAI 요약 중 오류 발생: {e}")
            return self._basic_summarize(docs, question)

    def _basic_summarize(self, docs: List[Document], question: str) -> str:
        """
        기본 요약 방식 (OpenAI API 없이)
        
        Args:
            docs: 문서 리스트
            question: 질문
            
        Returns:
            요약된 텍스트
        """
        context = "\n\n".join([doc.page_content for doc in docs])
        filled_prompt = self.prompt.format(context=context, query=question)
        response = self.llm.invoke(filled_prompt)
        return response.content

    def rag_pipeline(self, query: str, chunks: List[Document], embedding_model, 
                    keywords: Optional[List[str]] = None) -> str:
        """
        전체 RAG 파이프라인
        
        Args:
            query: 질문
            chunks: 문서 청크들
            embedding_model: 임베딩 모델
            keywords: 필터링할 키워드 리스트
            
        Returns:
            최종 답변
        """
        # 1차 Recall
        candidates = self.semantic_search(query, chunks, embedding_model, k=50)
        # 키워드 필터
        filtered = self.keyword_filter(candidates, keywords)
        # 2차 Re-Ranking
        reranked = self.rerank(query, filtered, top_k=10)
        # LLM Map-Reduce 요약
        summary = self.summarize_chunks(reranked, query)
        return summary

    def query(self, query: str, chunks: List[Document] = None, embedding_model = None, 
              keywords: Optional[List[str]] = None) -> dict:
        """
        질의 처리 메인 함수
        
        Args:
            query: 질문
            chunks: 문서 청크들 (None인 경우 retriever 사용)
            embedding_model: 임베딩 모델 (None인 경우 retriever 사용)
            keywords: 필터링할 키워드 리스트
            
        Returns:
            질의 결과 딕셔너리
        """
        if chunks is not None and embedding_model is not None:
            # 업그레이드된 파이프라인 사용
            response = self.rag_pipeline(query, chunks, embedding_model, keywords)
        else:
            # 기본 retriever 사용
            docs = self.retriever(query)
            response = self._basic_summarize(docs, query)
        
        return {
            'query': query,
            'response': response
        }