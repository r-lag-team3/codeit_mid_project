# 스크립트
from data.processed import chunking
from embeddings import faiss as FAISS # faiss와 구분하기 위해 대문자 사용
from rag_chain import RagChain_temp

# PDF 로드
from langchain_community.document_loaders import PyPDFLoader

# 클래스 
from langchain_core.documents import Document

# 파일 관리
import os
import json
import pickle

# langchain 임베딩
from langchain.embeddings import OpenAIEmbeddings

# 기타
import faiss 

def pdf_load(path)->list[Document]: 
    loader = PyPDFLoader(path)
    pdf_page = loader.load()
    return pdf_page


def chunking_documents(chunk_size, chunk_overlap, separators=None, tokenizer=None):
    base_path = "./data/processed/split_documents"
    if not os.path.exists(base_path):
        os.makedirs(base_path)

    raw_data_file = "./data/files"
    all_file_list = os.listdir(raw_data_file)

    custom_splitter = chunking.CustomSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap, separators=None, tokenizer=None) 

    split_documents = []

    for file in all_file_list:
        if not file.endswith('.pdf'):  # .pdf 파일이 아닌 경우 건너뛰기
            continue  

        file_path = os.path.join(raw_data_file, file)
        raw_documents = pdf_load(file_path)

        for doc in raw_documents:
            text = doc.page_content
            meta_data = doc.metadata

            splited_doc = custom_splitter.split_text(text=text, metadata=meta_data)
            
            split_documents.extend(splited_doc)
    
    # 추후 separators와 tokenizer가 지정되어 사용되는 경우, 파일 이름 구분이 힘들 수 있음. -> 수정 필요
    path = f"{base_path}/split_documents_{chunk_size}_{chunk_overlap}.pkl"
    with open(path, "wb") as f:
        pickle.dump(split_documents, f)


def load_split_documents(chunk_size, chunk_overlap, separators=None, tokenizer=None):
    base_path = "./data/processed/split_documents"
    path = f"{base_path}/split_documents_{chunk_size}_{chunk_overlap}.pkl"

    if not os.path.exists(path):
        chunking_documents(chunk_size, chunk_overlap, separators, tokenizer)

    with open(path, "rb") as f:
        split_documents = pickle.load(f)

    return split_documents
    

def experiment(version, experiment_name):
    # ------------------------------------------------------------------------------------------------------------------
    # 구조 설정 

    experiment_path = f"./experiment/{experiment_name}"

    if not os.path.exists(experiment_path):  # {test_name} 폴더 없는 경우 생성
        os.makedirs(experiment_path)

    config_path = f"{experiment_path}/experiment_config.json"    
    
    if not os.path.exists(config_path):  # experiment_config.json 파일이 없는 경우 생성
        with open(config_path, "w", encoding="utf-8") as f:
            print(f"실험 설정 파일을 생성합니다.")
            embedding_model_name = input(f"Embedding 모델 이름을 입력하세요 (예: text-embedding-3-small): ")
            llm_model_name = input(f"LLM 모델 이름을 입력하세요 (예: gpt-4.1-mini): ")
            chunk_size = int(input("Chunk 크기를 입력하세요 (예: 100): "))
            chunk_overlap = int(input("Chunk 오버랩 크기를 입력하세요 (예: 10): "))
            top_k = int(input("Top K 값을 입력하세요 (예: 3): "))

            config = {
                "version": version,
                "test_name": experiment_name,
                "embedding_model": embedding_model_name,
                "llm_model": llm_model_name,
                "chunk_size": chunk_size,
                "chunk_overlap": chunk_overlap,
                "top_k": top_k
            }

            json.dump(config, f, indent=4, ensure_ascii=False)
            print("실험 설정 파일을 저장했습니다.")

    else:  # experiment_config.json 파일이 있는 경우 불러오기 
        with open(config_path, "r", encoding="utf-8") as f:
            config = json.load(f)
            print(f"실험 설정 파일을 불러옵니다: {config}")

            embedding_model_name = config.get("embedding_model", "text-embedding-3-small")
            llm_model_name = config.get("llm_model", "gpt-4.1-mini")
            chunk_size = config.get("chunk_size", 100)
            chunk_overlap = config.get("chunk_overlap", 10)
            top_k = config.get("top_k", 3)

    readme_path = f"{experiment_path}/README.md"

    if not os.path.exists(readme_path):  # README.md 파일이 없는 경우 생성
        with open(readme_path, "w", encoding="utf-8") as f:
            f.write(f"# {experiment_name} 실험 기록\n")
            f.write("## 실험 정보\n")
            f.write(f"- 버전: {version}\n")
            f.write(f"- 임베딩 모델: {embedding_model_name}\n")
            f.write(f"- LLM 모델: {llm_model_name}\n")
            f.write(f"- Chunk 크기: {chunk_size}\n")
            f.write(f"- Chunk 오버랩 크기: {chunk_overlap}\n")
            f.write(f"- Top K 값: {top_k}\n")
            f.write("\n")
            f.write("## 실험 내용\n")

        print("README.md 파일을 생성했습니다. 실험 내용을 기록해주세요.")
    # ------------------------------------------------------------------------------------------------------------------
    # 청킹된 문서 로드
    split_documents = load_split_documents(chunk_size, chunk_overlap, separators=None, tokenizer=None)
    print(f"청킹 문서를 로드 완료하였습니다.\n청킹된 문서의 개수: {len(split_documents)}")

    # ------------------------------------------------------------------------------------------------------------------
    # 임베딩 모델 설정 
    embedding_model = OpenAIEmbeddings(model=embedding_model_name)
    print(f"임베딩 모델을 설정하였습니다: {embedding_model_name}")

    # ------------------------------------------------------------------------------------------------------------------
    # retriever 생성
    faiss_index_path = f"{experiment_path}/faiss_index.idx"

    if not os.path.exists(faiss_index_path):  # faiss_index.idx 파일이 없는 경우 -> 청킹 부터 시작
        print("FAISS 인덱스를 생성합니다...")

        custom_faiss = FAISS.CustomFAISS.from_documents(documents=split_documents, embedding_model=embedding_model)
        faiss.write_index(custom_faiss.index, faiss_index_path)  # index 저장

        print("FAISS 인덱스를 저장했습니다")
    else:
        index = faiss.read_index(faiss_index_path)
        print("FAISS 인덱스를 불러왔습니다")
        custom_faiss = FAISS.CustomFAISS(index=index, documents=split_documents, embedding_model=embedding_model)
    # ------------------------------------------------------------------------------------------------------------------
    # embedding 저장
    
    retriever = custom_faiss.as_retriever(top_k=top_k)
    print(f"retriever를 생성하였습니다. Top K: {top_k}")

    # ------------------------------------------------------------------------------------------------------------------
    # RAG Chain 생성
    rag = RagChain_temp.RAGChain(retriever=retriever, model_name=llm_model_name)
    print(f"RAG Chain을 생성하였습니다. LLM 모델: {llm_model_name}")

    # ------------------------------------------------------------------------------------------------------------------
    # 질의 테스트
    print("질의 테스트를 시작합니다")
    while True:
        query = input("질의를 입력하세요 (종료하려면 'exit' 입력): ")
        if query.lower() == 'exit':
            print("질의 테스트를 종료합니다.")
            break

        result = rag.query(query)
        print(f"Q. {result.get('query')}\n\nA. {result.get('response')}\n")