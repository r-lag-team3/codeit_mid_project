import os
import re
import fitz
import pickle
import logging
from data.processed import chunking
import pdfplumber
from transformers import AutoTokenizer
from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFLoader
# from konlpy.tag import Mecab
from mecab import MeCab
from langchain.schema import Document
from typing import List, Union
from soynlp.word import WordExtractor
from soynlp.tokenizer import LTokenizer

def preprocess_text(text: str) -> str:
    # MeCab 초기화
    mecab = MeCab()
    # mecab = Mecab(dicpath='C:/mecab/mecab-ko-dic')

    # 1. 특수문자 제거
    text = re.sub(r"[^\uAC00-\uD7A3a-zA-Z0-9\s]", " ", text)

    # 2. 문자 반복 제거
    text = re.sub(r'(.)\1{2,}', r'\1', text)

    # 3. 단어 반복 제거 (글자글자글자)
    text = re.sub(r'(\w+)\1{2,}', r'\1', text)

    token_pos_pairs = mecab.pos(text) # 형태소 분석 후 (토큰, 품사) 쌍 생성

    # 5. 토큰만 추출해서 중복 제거 (ngram 기반 반복 제거)
    tokens = [token for token, _ in token_pos_pairs]
    tokens = remove_ngram_repetitions(tokens)

    # 6. 중복 단어 제거 (seen 방식)
    KOREAN_JKG = {'은', '는', '이', '가', '을', '를', '에', '의', '에서', '와', '과', '도', '로', '으로', '만'}
    seen = set()
    unique_tokens = []
    for tok in tokens:
        if tok not in seen or tok in KOREAN_JKG:
            seen.add(tok)
            unique_tokens.append(tok)

    # 7. 단음절 병합
    merged_tokens = merge_short_korean(unique_tokens)

    # 8. 복합어 병합 (여기 추가!)
    merged_tokens = merge_known_compounds(merged_tokens)

    # 9. 띄어쓰기 복원 (품사 기반)
    final_token_pos = mecab.pos(" ".join(merged_tokens))
    restored_text = restore_spacing_with_pos(final_token_pos)

    return restored_text

def merge_known_compounds(tokens: List[str]) -> List[str]:
    compound_dict = {
        ("사용", "자"): "사용자",
        ("대표", "자"): "대표자",
        ("사업", "자"): "사업자",
        ("신청", "인"): "신청인",
        ("업무", "인"): "업무인",
    }

    result = []
    i = 0
    while i < len(tokens):
        matched = False
        for key in sorted(compound_dict, key=lambda x: -len(x)):  # 긴 조합 우선
            if tuple(tokens[i:i+len(key)]) == key:
                result.append(compound_dict[key])
                i += len(key)
                matched = True
                break
        if not matched:
            result.append(tokens[i])
            i += 1
    return result

def restore_spacing_with_pos(tokens_with_pos: List[tuple[str, str]]) -> str:
    no_space_pos = {'JKS', 'JKB', 'JKC', 'JKG', 'JKO', 'JX', 'JC', 'EP', 'EF', 'EC', 'ETN', 'ETM'} # 조사, 어미로 분류되어 띄어쓰기된 단어 제외

    restored = ""
    for i, (token, pos) in enumerate(tokens_with_pos):
        if i == 0:
            restored += token
        elif pos in no_space_pos:
            restored += token
        else:
            restored += " " + token
    return restored.strip()

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

def pdf_load(path: str) -> List[Document]:
    documents = []

    with pdfplumber.open(path) as pdf_plumber, fitz.open(path) as pdf_fitz:
        for page_num, (page_p, page_f) in enumerate(zip(pdf_plumber.pages, pdf_fitz)):
            try:
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

            except Exception as e:
                logging.warning(f"[page {page_num+1}] PDF 파싱 중 오류 발생: {e}")
                continue  # 문제 있는 페이지는 건너뜀

    return documents

class HFTokenizerWrapper:
    def __init__(self, tokenizer, max_length=512):
        self.tokenizer = tokenizer
        self.max_length = max_length

    def tokenize(self, text):
        return self.tokenizer.tokenize(text)

    def convert_tokens_to_string(self, tokens):
        return self.tokenizer.convert_tokens_to_string(tokens)

    def encode(self, text, **kwargs):
        return self.tokenizer.encode(
            text,
            truncation=kwargs.get("truncation", True),
            max_length=kwargs.get("max_length", self.max_length),
            **kwargs
        )

    def decode(self, token_ids, **kwargs):  
        return self.tokenizer.decode(token_ids, **kwargs)