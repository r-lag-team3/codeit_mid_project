from typing import Callable, Optional
from langchain.schema import Document

class CustomSplitter:
    def __init__(
        self,
        chunk_size:int = 70, 
        chunk_overlap:int = 10,
        separators: Optional[list[str]] = None,
        tokenizer: Optional[Callable[[str], list]] = None,  #  str 입력 list 반환 형태의 데이터를 받으며 list 형태지만 기본값을 None 설정
    ):
        """
        # CustomSplitter
        문서를 청크 단위로 나누는 분리기
        ## Parameters
        - chunk_size: 청크의 크기
        - chunk_overlap: 청크 간의 오버랩 크기
        - separators: 청크를 나누는 구분자 리스트 (기본값은 마침표, 줄바꿈 등)
        - tokenizer: 선택적 토크나이저 함수 (기본값은 None -> 재귀 방식으로 청크 분리)
        ## Methods
        - split_text: 주어진 텍스트를 청크 단위로 나누고 Document 객체로 반환
        - _recursive_split: 재귀적으로 텍스트를 청크 단위로 나누는 내부 메소드
        - _add_overlap: 청크 간의 오버랩을 추가하는 내부 메소드
        ## Usage
        ```python
        splitter = CustomSplitter(chunk_size=100, chunk_overlap=20)
        documents = splitter.split_text(text="문서 내용", metadata={"source": "source.pdf"})
        ```
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.separators = separators or ["\n\n", "\n", ".", " ", ""]  # 별도의 구분자가 없으면 기본 구분으로 마침표와 줄바꿈등
        self.tokenizer = tokenizer

    def split_text(self, text:str, metadata: Optional[dict] = None) ->list[Document]:  
        if self.tokenizer:
            tokens = self.tokenizer.tokenize(text)
            chunks = []

            for i in range(0, len(tokens), self.chunk_size - self.chunk_overlap):  # for문으로 청크사이즈 - 오버랩 크기만큼 진행 + 청크사이즈로 나눠서 토크나이저에 입력
                chunk_tokens = tokens[i : i + self.chunk_size]                     # 단어를 자른 토큰
                chunk_text = self.tokenizer.convert_tokens_to_string(chunk_tokens) # 문자열로 다시 합치기 # AutoTokenizer 기반의 방식
                chunks.append(chunk_text)                                          # 단어가 길이로 인해 잘릴시 나누는게 가능한 단어 토큰으로 문장복원
        else:
            chunks = self._recursive_split(text, self.separators)
        
        return [Document(page_content=chunk, metadata=metadata or {}) for chunk in chunks]  # 최종 반환 타입 [] 리스트형의 Document 데이터 문장, 페이지 구성

    def _recursive_split(self, text: str, separators: list[str])->list[str]: # 기본 방식의 청킹
        if len(text) <= self.chunk_size:
            return [text]

        for sep in separators:
            parts = text.split(sep) if sep else list(text) # 문장기호를 기준으로 나누기
            splits, current = [], ""

            for part in parts:
                if current:
                    part = sep + part
                if len(current + part) > self.chunk_size:
                    if current:
                        splits.append(current)
                    current = part.lstrip(sep)
                else:
                    current += part
            if current:
                splits.append(current)

            final_chunks = []
            for chunk in splits:
                if len(chunk) > self.chunk_size and len(separators) > 1:
                    final_chunks.extend(self._recursive_split(chunk, separators[1:]))
                else:
                    final_chunks.append(chunk)

            return self._add_overlap(final_chunks)

        return [text]

    def _add_overlap(self, chunks):
        result = []
        for i, chunk in enumerate(chunks):
            if i == 0:
                result.append(chunk[:self.chunk_size])
            else:
                prev = result[-1]
                overlap = prev[-self.chunk_overlap:] if len(prev) > self.chunk_overlap else ""  # 입력받은 overlab 길이만큼 복사하여 추가 
                result.append((overlap + chunk)[:self.chunk_size])
        return result
