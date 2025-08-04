import os
import re
import fitz
import pickle
from data.processed import chunking
import pdfplumber
from transformers import AutoTokenizer
from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFLoader
from mecab import MeCab
from langchain.schema import Document
from typing import List, Union
from soynlp.word import WordExtractor
from soynlp.tokenizer import LTokenizer

def preprocess_text(text: str) -> str:
    # 특수문자 제거
    mecab = MeCab()
    
    text = re.sub(r"[^\uAC00-\uD7A3a-zA-Z0-9\s]", " ", text)

    # 1. 특수문자 제거 + 형태소 분석
    tokens = mecab.morphs(text)

    # 2. 문자 반복 제거 (ㅋㅋㅋㅋ, 한한한한)
    tokens = [re.sub(r'(.)\1{2,}', r'\1', t) for t in tokens]

    # 3. 단어 반복 제거 (글자글자글자)
    tokens = [re.sub(r'(\w+)\1{2,}', r'\1', t) for t in tokens]

    # 4. N-gram 중복 단어 제거
    tokens = remove_ngram_repetitions(tokens)
    # 단음절 병합 기업 / 신용 / 병합 
    tokens = merge_short_korean(tokens)
    
    tokens = remove_ngram_repetitions(tokens)

    # restored_text = restore_spacing_from_morphs(tokens)
    return " ".join(tokens)

def restore_spacing_from_morphs(tokens: List[str]) -> str:
    # 조사, 어미 등 붙여쓰기 대상
    no_space_after = {'은', '는', '이', '가', '을', '를', '의', '에', '에서', '한', '도', '만', '로', '과', '와'}

    restored = ""
    for i, token in enumerate(tokens):
        if i == 0:
            restored += token
        elif token in no_space_after:
            restored += token
        else:
            restored += " " + token
    return restored.strip()

def train_soynlp_tokenizer(corpus: List[str]) -> LTokenizer:
    extractor = WordExtractor()
    extractor.train(corpus)
    word_scores = extractor.extract()
    return LTokenizer(scores=word_scores)

def merge_short_korean(tokens: List[str], min_unit: int = 2) -> List[str]:
    merged = []
    buffer = ""

    for token in tokens:
        if len(token) == 1 and re.match(r'[가-힣]', token):
            buffer += token
        else:
            if len(buffer) >= min_unit:
                merged.append(buffer)
            elif buffer:
                merged.extend(list(buffer))  # 한 글자씩 나누기
            buffer = ""
            merged.append(token)

    if buffer:
        if len(buffer) >= min_unit:
            merged.append(buffer)
        else:
            merged.extend(list(buffer))
    return merged

def table_to_text(table: List[List[str]]) -> str:
    if not table or len(table) < 2:
        return ""

    header = table[0]
    rows = table[1:]

    lines = []
    for row in rows:
        if len(row) != len(header):
            continue  # 불완전한 행은 무시
        line = ", ".join([f"{col_name}: {cell}" for col_name, cell in zip(header, row)])
        lines.append(line)

    return "\n".join(lines)

def remove_ngram_repetitions(tokens: List[str], max_ngram=3) -> List[str]: # n gram 중복 단어 제거
    result = []
    i = 0
    while i < len(tokens):
        repeated = False
        for n in range(max_ngram, 0, -1):  # 3-gram → 2-gram → unigram 순으로 체크
            if i + 2 * n <= len(tokens):
                unit = tokens[i:i+n]
                next_unit = tokens[i+n:i+2*n]
                if unit == next_unit:
                    # 반복 발생 → 한 번만 추가하고 skip
                    result.extend(unit)
                    i += n * 2
                    repeated = True
                    break
        if not repeated:
            result.append(tokens[i])
            i += 1
    return result
    
def clean_repetitions(text: str) -> str:
    # 1. 문자 반복 3번 이상 제거 (예: ㅋㅋㅋㅋ → ㅋ)
    text = re.sub(r'(.)\1{2,}', r'\1', text)

    # 2. 단어 반복 3번 이상 제거 (예: 글자글자글자 → 글자)
    text = re.sub(r'\b(\w+)\1{2,}\b', r'\1', text)

    # 3. 문장/구 단위 반복 (예: "이 문장은 반복된다 이 문장은 반복된다" → 1회만 남김)
    tokens = text.split()
    seen = set()
    result = []
    for token in tokens:
        if token not in seen:
            seen.add(token)
            result.append(token)
    return " ".join(result)

def pdf_load(path: str) -> List[Document]:
    documents = []

    with pdfplumber.open(path) as pdf_plumber, fitz.open(path) as pdf_fitz:
        for page_num, (page_p, page_f) in enumerate(zip(pdf_plumber.pages, pdf_fitz)):

            metadata = {"source": path, "page": page_num + 1}
            texts = []

            # 1. 표 추출
            tables = page_p.extract_tables()
            for table in tables:
                table_text = table_to_text(table)  # 기존 함수 활용
                if table_text:
                    texts.append(preprocess_text(table_text))  # 중복 제거 등 전처리

            # 2. 일반 텍스트 (fitz로 더 정확하게 추출)
            raw_text = page_f.get_text()
            cleaned_text = preprocess_text(raw_text)
            texts.append(cleaned_text)

            # 병합 후 Document 생성
            full_text = "\n".join(texts).strip()
            if full_text:
                documents.append(Document(page_content=full_text, metadata=metadata))

    return documents

def chunking_documents(chunk_size, chunk_overlap, separators=None, tokenizer=None):
    base_path = "/split_documents"
    if not os.path.exists(base_path):
        os.makedirs(base_path)

    # raw_data_file = "/content/drive/MyDrive/midle_project/files"
    raw_data_file = "./"
    all_file_list = os.listdir(raw_data_file)

    custom_splitter = chunking.CustomSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap, separators= separators, tokenizer= tokenizer) 

    split_documents = []

    for file in all_file_list:
        if not file.endswith('.pdf'):  # .pdf 파일이 아닌 경우 건너뛰기
            continue  

        file_path = os.path.join(raw_data_file, file).replace("\\", "/")
        raw_documents = pdf_load(file_path)

        for doc in raw_documents:
            text = doc.page_content
            meta_data = doc.metadata

            splited_doc = custom_splitter.split_text(text=text, metadata=meta_data)
            split_documents.extend(splited_doc)

    path = f"{base_path}/split_documents_{chunk_size}_{chunk_overlap}_token2.pkl"
    with open(path, "wb") as f:
        pickle.dump(split_documents, f)

def load_split_documents(chunk_size, chunk_overlap, separators=None, tokenizer=None):
    base_path = "/split_documents"
    path = f"{base_path}/split_documents_{chunk_size}_{chunk_overlap}_token2.pkl"

    if not os.path.exists(path):
        chunking_documents(chunk_size, chunk_overlap, separators, tokenizer)

    with open(path, "rb") as f:
        split_documents = pickle.load(f)

    return split_documents


class HFTokenizerWrapper:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def tokenize(self, text):
        return self.tokenizer.tokenize(text)

    def convert_tokens_to_string(self, tokens):
        return self.tokenizer.convert_tokens_to_string(tokens)

    def encode(self, text, **kwargs):
        return self.tokenizer.encode(text, **kwargs)

    def decode(self, token_ids, **kwargs):  
        return self.tokenizer.decode(token_ids, **kwargs)