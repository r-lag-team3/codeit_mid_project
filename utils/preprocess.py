import os
import re
import fitz  # PyMuPDF
import pdfplumber
from langchain_core.documents import Document

from langchain.schema import Document
from typing import List, Union

def pdf_load(path: str) -> List[Document]:
    documents = []

    with fitz.open(path) as pdf, pdfplumber.open(path) as plumber_pdf:
        for page_index, (page_fitz, page_plumber) in enumerate(zip(pdf, plumber_pdf.pages)):
            # 표 추출 (pdfplumber)
            tables = page_plumber.extract_tables()
            table_texts = []
            for table in tables:
                table_text = "\n".join([
                    "\t".join([str(cell) if cell is not None else "" for cell in row])
                    for row in table if row
                ])
                if table_text.strip():
                    table_texts.append(table_text)
            # 일반 텍스트 추출 (fitz)
            text = page_fitz.get_text()
            # 표와 일반 텍스트 합치기
            full_text = "\n".join(table_texts + [text]).strip()
            full_text = full_text.replace('\t', ' ')
            full_text = re.sub(r"[^\uAC00-\uD7A3a-zA-Z0-9\s]", " ", full_text)
            full_text = re.sub(r"\s+", " ", full_text)
            full_text = re.sub(r'\b(\w+)( \1\b)+', r'\1', full_text)
            metadata = {"source": path, "page": page_index + 1}
            documents.append(Document(page_content=full_text, metadata=metadata))

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