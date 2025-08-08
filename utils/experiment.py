# 스크립트
from data.processed import chunking
from embeddings import faiss as FAISS  # faiss와 구분하기 위해 대문자 사용
from rag_chain import RagChain_temp
from utils import preprocess

# PDF 로드
from langchain_community.document_loaders import PyPDFLoader
import fitz

# 클래스 
from langchain_core.documents import Document

# 파일 관리
import os
import json
import pickle

# langchain 임베딩
from langchain.embeddings import OpenAIEmbeddings
from sentence_transformers import SentenceTransformer

# 기타
import faiss 

def pdf_load_PyPDFLoader(path) -> list[Document]:
    """
    # pdf_load_PyPDFLoader
    PyPDFLoader를 이용하여 개별 PDF파일을 로드
    ## 파라미터
    - path: pdf파일 경로
    ## return
    - documents: 각 페이지를 Document로 담은 리스트
    """
    loader = PyPDFLoader(path)
    pdf = loader.load()
    documents = []

    for page_index, doc in enumerate(pdf):
        metadata = {"source": path, "page": page_index + 1}
        text = doc.page_content
        documents.append(Document(page_content=text, metadata=metadata))

    return documents


def pdf_load_PyMuPDF(path)->list[Document]:
    """
    # pdf_load_PyMuPDF
    PyMuPDF 이용하여 개별 PDF파일을 로드
    ## 파라미터
    - path: pdf파일 경로
    ## return
    - pdf: 각 페이지를 Document로 담은 리스트
    """
    with fitz.open(path) as pdf:
        documents = []
        for page_index, page in enumerate(pdf):
            text = page.get_text()
            metadata = {"source": path, "page": page_index + 1}
            documents.append(Document(page_content=text, metadata=metadata))
    return documents


def chunking_documents_and_save(chunk_size, chunk_overlap, separators=None, tokenizer=None, version='2.0'):
    base_path = f"./data/processed/split_documents/version{version}"
    chunk_documents_path = f"{base_path}/chunksize{chunk_size}_overlap{chunk_overlap}.pkl"
    if not os.path.exists(base_path):
        os.makedirs(base_path)

    pdf_data_file = "./data/pdf_data"

    all_file_list = os.listdir(pdf_data_file)
    if tokenizer:
        hf_tokenizer = preprocess.HFTokenizerWrapper(tokenizer=tokenizer)
    else:
        hf_tokenizer = None
    custom_splitter = chunking.CustomSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap, separators=None, tokenizer= hf_tokenizer) 

    split_documents = []

    for file in all_file_list:
        if not file.endswith('.pdf'):  # .pdf 파일이 아닌 경우 건너뛰기
            continue  

        file_path = os.path.join(pdf_data_file, file).replace("\\", "/")
        if version == '1.0':
            raw_documents = pdf_load_PyPDFLoader(file_path)
        elif version == '2.0':
            raw_documents = preprocess.pdf_load(file_path)  # PyMuPDF를 이용한 로드 + 전처리

        for doc in raw_documents:
            text = doc.page_content
            meta_data = doc.metadata

            splited_doc = custom_splitter.split_text(text=text, metadata=meta_data)
            split_documents.extend(splited_doc)

    with open(chunk_documents_path, "wb") as f:
        pickle.dump(split_documents, f)


def load_split_documents(path):
    with open(path, "rb") as f:
        split_documents = pickle.load(f)

    return split_documents


def load_span(path):
    with open(path, 'r', encoding='utf-8') as f:
        span = json.load(f)
    return span


def get_sample_querys(span):
    sample_query = []
    for qa in span:
        sample_query.append(qa.get('query'))
    return sample_query


def question(ragchain, query, result_log_path):
    result = ragchain.query(query)
    response_text = result.get('response')
    retrieved_docx = result.get('retrieved_docs', [])

    # source 정보 추출 (문서명, 페이지)
    sources = {}
    for doc in retrieved_docx:
        source = doc.metadata.get('source', 'Unknown Source')
        page = doc.metadata.get('page', 'Unknown Page')
        if source not in sources:
            sources[source] = [page]
        else:
            if isinstance(page, list):
                sources[source].extend(page)
            else:
                sources[source].append(page)

    print("\n" + "="*30)
    print(f"Q. {query}\n")
    print(f"A. {response_text}\n")
    print("-"*30 + "\n")
    print("[참고]")

    for doc in retrieved_docx:
        print(f"Retrieved Document: {doc.metadata.get('source', 'Unknown Source')}")
        print(f"page: {doc.metadata.get('page', 'Unknown Page')}")
        print(f"Content: {doc.page_content}")
        print("-"*30)

    with open(result_log_path, "a", encoding="utf-8") as f:
        json.dump({
            "query": query,
            "response": response_text,
            "source": sources,
        }, f, ensure_ascii=False)
        f.write("\n")
    

def save_experiment_structure(version, experiment_name):
    config_path = f"./experiment/{experiment_name}/experiment_config.json"
    with open(config_path, "w", encoding="utf-8") as f:
        print(f"실험 설정 파일을 생성합니다.")
        print('[모델명 참고]')
        print("사용 가능 임베딩 모델: text-embedding-3-small")
        print("사용 가능 LLM 모델: gpt-4.1-mini, gpt-4.1-nano, gpt-4.1, gpt-4o")
        embedding_model_name = input(f"Embedding 모델 이름을 입력하세요 (미입력 시: text-embedding-3-small): ") or "text-embedding-3-small"

        llm_model_name = input(f"LLM 모델 이름을 입력하세요 (미입력 시: gpt-4.1-mini): ") or "gpt-4.1-mini"

        chunk_size_input = input("Chunk 크기를 입력하세요 (미입력 시: 100): ")
        chunk_size = int(chunk_size_input) if chunk_size_input else 100

        chunk_overlap_input = input("Chunk 오버랩 크기를 입력하세요 (미입력 시: 10): ")
        chunk_overlap = int(chunk_overlap_input) if chunk_overlap_input else 10
        
        top_k_input = input("Top K 값을 입력하세요 (미입력 시: 10): ")
        top_k = int(top_k_input) if top_k_input else 10

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
        print("실험 설정 파일을 저장했습니다.\n")

def experiment(version, experiment_name, tokenizer=None):
    # ------------------------------------------------------------------------------------------------------------------
    step = 1
    print(f"[{step}] 실험 구조 설정")
    step += 1
    experiment_path = f"./experiment/{experiment_name}"
    config_path = f"{experiment_path}/experiment_config.json"

    if not os.path.exists(experiment_path):
        os.makedirs(experiment_path)

    if not os.path.exists(config_path):
        save_experiment_structure(version, experiment_name)

    print(f"실험 설정 파일을 불러옵니다.")
    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)
    embedding_model_name = config.get("embedding_model")
    llm_model_name = config.get("llm_model")
    chunk_size = config.get("chunk_size")
    chunk_overlap = config.get("chunk_overlap")
    top_k = config.get("top_k")
    print(f"실험 설정을 불러왔습니다.: {config}")          

    readme_path = f"{experiment_path}/README.md"
    if not os.path.exists(readme_path):
        print('README.md 파일을 생성합니다.')
        with open(readme_path, "w", encoding="utf-8") as f:
            f.write(f"# {experiment_name} 실험 기록\n")
            f.write("## 실험 정보\n")
            f.write(f"- 버전: {version}\n")
            f.write(f"- 임베딩 모델: {embedding_model_name}\n")
            f.write(f"- LLM 모델: {llm_model_name}\n")
            f.write(f"- Chunk 크기: {chunk_size}\n")
            f.write(f"- Chunk 오버랩 크기: {chunk_overlap}\n")
            f.write(f"- Top K 값: {top_k}\n")
            f.write("\n## 실험 내용\n")
        print(f"README.md 파일을 생성했습니다. 실험 내용을 기록해주세요.\n")
    else: 
        print('README.md 파일이 존재합니다. 실험 내용을 기록해주세요.\n')
    # ------------------------------------------------------------------------------------------------------------------
    print(f"[{step}] Document Chunking")
    step += 1
    chunk_document_path = f"./data/processed/split_documents/version{version}/chunksize{chunk_size}_overlap{chunk_overlap}.pkl"
    if not os.path.exists(chunk_document_path):
        print("청크 파일을 생성합니다.")
        chunking_documents_and_save(chunk_size, chunk_overlap, separators=None, tokenizer=tokenizer, version='2.0')
        print("청크 파일을 생성하였습니다.")
    else: 
        print("청크 파일이 존재합니다.")
    
    split_documents = load_split_documents(chunk_document_path)
    print(f"청크 문서를 로드 완료하였습니다.\n문서 개수: {len(split_documents)}\n")
    # ------------------------------------------------------------------------------------------------------------------
    print(f"[{step}] 임베딩 및 리트리버 생성")
    step += 1 
    if embedding_model_name.startswith("text"):
        embedding_model = OpenAIEmbeddings(model=embedding_model_name)
    else:
        embedding_model = SentenceTransformer(embedding_model_name, trust_remote_code=True)
    print(f"임베딩을 설정하였습니다. \n모델: {embedding_model_name}")

    faiss_index_path = f"{experiment_path}/faiss_index.idx"
    if not os.path.exists(faiss_index_path):
        print(f" FAISS 인덱스를 생성합니다.")
        custom_faiss = FAISS.CustomFAISS.from_documents(documents=split_documents, embedding_model=embedding_model)
        faiss.write_index(custom_faiss.index, faiss_index_path)
        print("FAISS 인덱스를 저장했습니다")
    else:
        index = faiss.read_index(faiss_index_path)
        print("FAISS 인덱스를 불러왔습니다")
        custom_faiss = FAISS.CustomFAISS(index=index, documents=split_documents, embedding_model=embedding_model)
    print(f"retriever를 생성하였습니다.\n")
    # ------------------------------------------------------------------------------------------------------------------
    print(f"[{step}] RAG Chain")
    step += 1
    # RAGChain의 top_k는 사용자 입력값을 사용 (최종 답변 청크 개수)
    # retriever에 CustomFAISS 객체를 직접 넘김
    rag = RagChain_temp.RAGChain(retriever=custom_faiss, model_name=llm_model_name, top_k=top_k)
    rag.all_docs = split_documents
    print(f'RAG Chain을 생성했습니다. llm 모델: {llm_model_name}\n')
    # ------------------------------------------------------------------------------------------------------------------
    print(f"[{step}] 질의 테스트를 시작합니다")
    step += 1
    result_log_path = f"{experiment_path}/result.jsonl"
    span_path = f"./data/processed/span/span_list.json"

    while True:
        query = input("질의를 입력하세요\n(종료: exit, 자동 질의 입력: auto)\n질의: ")
        if query.lower() == 'exit':
            print("질의 테스트를 종료합니다.")
            break

        if query.lower() == 'auto':
            print('샘플 질의를 이용하여 질의를 진행합니다.')
            span = load_span(span_path)
            sample_querys = get_sample_querys(span)
            for query in sample_querys:
                question(ragchain=rag, query=query, result_log_path=result_log_path)
            break

        else:
            question(ragchain=rag, query=query, result_log_path=result_log_path)

    
    print("응답이 저장되었습니다.")