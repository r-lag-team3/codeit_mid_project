# 라이브러리 이용 -------------------------------------------------------------------------------------------------

from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.schema import Document

# 동우님이 원래 사용하던 방식을 클래스화
# 용어 통일: chunks -> documents

class CustomFAISS2:
    def __init__(self, documents:list[Document], embedding_model_name:OpenAIEmbeddings="text-embedding-3-small", top_k:int=3):
        self.embedding_model = OpenAIEmbeddings(model=embedding_model_name)
        self.db = FAISS.from_documents(documents, self.embedding_model)
        self.retriever = self.db.as_retriever(search_kwargs={"k": top_k})

# 직접 구현 ---------------------------------------------------------------------------------------------------------

# 수정 
# embedding_model을 받는 방식에서 모델 이름을 받도록 설정
# OpenAIEmbeddings과 HuggingFaceEmbeddings 두 가지 방식 사용 가능

import faiss
import numpy as np
from langchain.embeddings import OpenAIEmbeddings, HuggingFaceEmbeddings
from typing import List, Callable
from langchain.schema import Document


class CustomFAISS:
    def __init__(self, index:faiss.IndexFlatL2, documents:list[Document], embedding_model:OpenAIEmbeddings | HuggingFaceEmbeddings):
        """
        # CustomFAISS 
        FAISS 인덱스를 사용하여 문서 검색을 수행
        ## Parameters
        - index: faiss.IndexFlatL2 인덱스 객체
        - documents: Document 객체의 리스트
        - embedding_model: OpenAIEmbeddings 또는 HuggingFaceEmbeddings 객체(OpenAIEmbeddings만 사용 가능, HuggingFaceEmbeddings는 추후 확장 가능)
        ## Methods
        - from_documents: 문서 리스트로부터 CustomFAISS 객체를 생성(index가 없는 경우)
        - as_retriever: 검색 기능을 제공하는 retriever 함수 반환
        ## Usage
        ```python
        custom_faiss = CustomFAISS.from_documents(documents, embedding_model)
        retriever = custom_faiss.as_retriever(top_k=3)
        """
        self.documents = documents
        self.index = index
        self.embedding_model = embedding_model


    @classmethod
    def from_documents(cls, documents: List[Document], embedding_model:OpenAIEmbeddings | HuggingFaceEmbeddings) -> 'CustomFAISS':
        texts = [doc.page_content for doc in documents]
        embeddings = embedding_model.embed_documents(texts)
        embeddings_np = np.array(embeddings).astype("float32")

        dim = embeddings_np.shape[1]
        index = faiss.IndexFlatL2(dim)  # L2 거리 기반 인덱스
        index.add(embeddings_np)

        return cls(index, documents, embedding_model)


    def as_retriever(self, top_k: int = 3)->Callable:
        """
        # as_retriever
         검색 기능을 제공하는 retriever 함수 반환
        ## Parameters
        - top_k: 검색할 문서의 개수
        ## Returns
        - retriever_fn: 검색 쿼리를 받아서 top_k 개의 문서를 반환하는 함수
        ## Usage
        ```python
        retriever = custom_faiss.as_retriever(top_k=3)
        results = retriever("검색 쿼리")
        ```
        """
        def retriever_fn(query: str) -> List[Document]:
            query_vec = self.embedding_model.embed_query(query)
            query_np = np.array(query_vec).astype("float32").reshape(1, -1)
            distances, indices = self.index.search(query_np, top_k)
            return [{'doc': self.documents[i], 'chunk_id': i} for i in indices[0] if i != -1]
        return retriever_fn
        