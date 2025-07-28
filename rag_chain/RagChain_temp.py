import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate

   

class RAGChain:
    def __init__(self, retriever, model_name="gpt-3.5-turbo"):
        self.retriever = retriever
        self.llm = ChatOpenAI(model_name=model_name, temperature=0.3)

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

    def query(self, query: str):
        docs = self.retriever(query)
        context = "\n\n".join([doc.page_content for doc in docs])
        filled_prompt = self.prompt.format(context=context, query=query)
        response = self.llm.invoke(filled_prompt)
        return {
            'query': query,
            'response': response.content
        }