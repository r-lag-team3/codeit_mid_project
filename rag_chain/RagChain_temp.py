
from langchain.schema import Document
from openai import OpenAI
from typing import List, Optional, Callable
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from transformers import AutoTokenizer, AutoModelForCausalLM
from sentence_transformers import CrossEncoder
from collections import deque
from keybert import KeyBERT
import torch
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
    def __init__(self, retriever: Callable, model_name: str = "gpt-4.1-mini", top_k: int = 10):
        """
        업그레이드된 RAG Chain 클래스
        
        Args:
            retriever: 검색을 수행하는 retriever 함수
            model_name: LLM 모델 이름
            openai_api_key: OpenAI API 키
            top_k: rerank에서 사용할 상위 k개 문서 개수
        """
        self.retriever = retriever
        if model_name.startswith("gpt"):
            self.llm = ChatOpenAI(model_name=model_name, temperature=0.3)
            self.openai = True
        else:
            self.model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype= torch.bfloat16).to("cuda")
            self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
            self.openai = False
        self.top_k = top_k

        # 이전 대화 키워드 기억용
        self.keyword_memory = KeywordMemory()
        # 한글 키워드 추출기 (KeyBERT)
        self.keyword_extractor = KeyBERT(model="distiluse-base-multilingual-cased-v2")
        self.cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
        
        # 기본 프롬프트 템플릿
        template = """
        당신은 문서를 기반으로 답변하는 AI입니다.
        이전 대화 키워드: {past_keywords}
        
        문서에 있는 내용만을 사용하여 답변하세요.
        문서가 충분하지 않으면 "문서에는 내용이 부족합니다."라고 답변하세요.
        답변에는 반드시 관련 문서의 근거(문장 또는 항목 번호 등)를 명시하세요.
        문서의 목적, 요구사항, 평가기준, 제출조건 등 RFP의 주요 항목을 우선적으로 참고하세요.
        질문이 모호하거나 여러 해석이 가능하면, 가능한 해석을 모두 제시하고 각각에 대해 답변하세요.
        단 문서에 명시되지 않은 법령명, 조항 번호, 기관명, 다국어 문장 등은 절대 포함하지 마십시오.

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
    def has_information(self, query: str, reranked_docs: List[Document], model: str = "gpt-4.1-mini") -> bool:
        context = "\n\n".join(doc.page_content for doc in reranked_docs[:10])  # 상위 N만
        messages = [
            {"role": "system", "content": "문서에 질문에 대한 답이 있는지 판단하는 AI입니다. '있음' 또는 '없음'으로만 응답하세요."},
            {"role": "user", "content":
                f"문서에 질문과 관련된 정보가 조금이라도 있으면 '있음', 전혀 없으면 '없음'이라고만 답하세요.\n\n문서:\n{context}\n\n질문:\n{query}\n\n답:"}
        ]
        try:
            client = OpenAI()
            res = client.chat.completions.create(model=model, messages=messages, temperature=0.0, max_tokens=5)
            ans = (res.choices[0].message.content or "").strip().replace(" ", "")
            # ✅ 더 안전한 판정
            return ans.startswith("있음")
        except Exception as e:
            print(f"[정보 유무 판단 실패] {e}")
            return True
        
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
        try:
            client = OpenAI()
            summaries = []

            for doc in docs:
                messages = [
                    {
                        "role": "system",
                        "content": "당신은 주어진 문서와 질문, 그리고 과거 대화 키워드를 기반으로 3줄 요약을 생성하는 요약 AI입니다.",
                    },
                    {
                        "role": "user",
                        "content": f"이전 대화 키워드: {past_keywords_text}\n\n다음 내용을 질문 관점에서 3줄로 요약해 주세요.\n\n질문: {question}\n\n문서:\n{doc.page_content}",
                    }
                ]

                response = client.chat.completions.create(
                    model=model,
                    messages=messages,
                    temperature=0.5,
                    max_tokens=512,
                )
                summaries.append(response.choices[0].message.content.strip())

            # Reduce 요약
            reduce_messages = [
                {
                    "role": "system",
                    "content": "당신은 여러 개의 요약을 종합하여 최종 요약을 만드는 요약 AI입니다.",
                },
                {
                    "role": "user",
                    "content": f"이전 대화 키워드: {past_keywords_text}\n\n아래 요약들을 종합해서 3줄로 다시 요약해 주세요:\n\n{chr(10).join(summaries)}",
                }
            ]

            final_response = client.chat.completions.create(
                model=model,
                messages=reduce_messages,
                temperature=0.5,
                max_tokens=512,
            )
            return final_response.choices[0].message.content.strip()

        except Exception as e:
            print(f"[OpenAI 요약 실패] {e}")
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
        
        if self.openai:
            response = self.llm.invoke(filled_prompt)
            return response.content if hasattr(response, "content") else response
        else:
            inputs = self.tokenizer(filled_prompt, return_tensors="pt").to("cuda")
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=128,
                temperature=0.2,
                do_sample=True,
                top_p=0.8,
                top_k=30,
                repetition_penalty=1.2,
                no_repeat_ngram_size=3,
                pad_token_id=self.tokenizer.eos_token_id,
            )
            decoded = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            # '답변:' 뒤부터만 잘라내는 후처리 추천
            return decoded.split("답변:")[-1].strip()

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
        stop_word = {"사업"}
        if m:
            cand =  m.group(1)
            if cand in stop_word:
                return ""
            else:
                return cand       

        # fallback: 첫 단어
        return query.split()[0]
    def rerank_by_distance(self, query: str, docs: List[Document], top_k: int = 5) -> list[Document]:
        """
        CrossEncoder를 활용하여 query와 문서 간의 relevance score 기반 rerank
        
        Args:
            query: 사용자 질문
            docs: 필터링된 Document 리스트
            top_k: 상위 몇 개 문서를 반환할지
            
        Returns:
            점수가 높은 상위 top_k Document 리스트
        """
        # 1. (query, doc) 쌍 만들기
        pairs = [(query, doc.page_content) for doc in docs]
        
        # 2. 스코어 계산
        scores = self.cross_encoder.predict(pairs)

        # 3. 문서와 스코어 결합 후 상위 top_k 선택
        ranked = sorted(zip(docs, scores), key=lambda x: x[1], reverse=True)
        
        return [doc for doc, _ in ranked[:top_k]]
    def _rrf_fuse(self, lists: list[list], k: int = 60) -> list:
        """
        Reciprocal Rank Fusion (RRF) by rank only.
        lists: 각 리스트는 Document들의 순서 목록
        k: RRF 상수 (보통 60 근처)
        """
        from collections import defaultdict
        score = defaultdict(float)

        def key(d):
            # 문서 중복 식별 키 (source, page 중심)
            src = d.metadata.get("source", "")
            page = d.metadata.get("page", d.metadata.get("page_number", -1))
            return (src, page)

        # 랭크 합산
        for lst in lists:
            for r, d in enumerate(lst):
                score[key(d)] += 1.0 / (k + r + 1)

        # 키→Document 대표 매핑
        rep = {}
        for lst in lists:
            for d in lst:
                rep[key(d)] = d

        fused = sorted(rep.values(), key=lambda d: score[key(d)], reverse=True)
        return fused

    def _retrieve_candidates(self, query: str) -> list[Document]:
        """
        멀티쿼리(원쿼리+확장쿼리)로 각 쿼리마다 임베딩 검색(=리트리버 호출),
        RRF로 융합 후 (source,page) 단위 중복 제거.
        """
        # 1) 멀티 쿼리 생성
        mq = getattr(self, "use_multi_query", True)   # ← 외부에서 끄고 켤 수 있음
        num_variants = getattr(self, "num_query_variants", 3)
        queries = self._expand_queries(query, n=num_variants) if mq else [query]

        # 2) 각 쿼리별 후보 검색 (여기서 질문 임베딩 수행됨)
        lists = []
        pool_per_query = max(self.top_k * 3, 30)     # 풀 크게 뽑아서 RRF 이득
        # retriever가 top_k를 내부에서 쓰는 경우가 많으니, 가급적 retriever를 생성할 때 top_k를 크게 주는 편 권장
        for q in queries:
            cands = self.retriever(q)
            docs = []
            for c in cands:
                if isinstance(c, dict) and "doc" in c:
                    docs.append(c["doc"])
                elif isinstance(c, Document):
                    docs.append(c)
            lists.append(docs[:pool_per_query])

        # 3) RRF 융합
        fused = self._rrf_fuse(lists, k=60)

        # 4) (source, page) 기준 중복 제거
        seen = set()
        dedup = []
        for d in fused:
            key = (d.metadata.get("source",""), d.metadata.get("page", d.metadata.get("page_number",-1)))
            if key in seen:
                continue
            seen.add(key)
            dedup.append(d)

        # 최종 후보 풀 반환 (후속 단계에서 파일명 필터 → CE 재랭크)
        return dedup
    
    def _expand_queries(self, query: str, n: int = 3) -> list[str]:
        """질문 변형 쿼리 생성: LLM 우선, 실패 시 키워드 기반 보조"""
        variants = []
        try:
            client = OpenAI()
            prompt = (
                "다음 질문을 의미를 보존한 채 한국어로 서로 다른 형태로 3가지로 바꿔줘. "
                "각 줄 하나씩, 불필요한 설명 없이."
                f"\n질문: {query}"
            )
            res = client.chat.completions.create(
                model="gpt-4.1-mini",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7,
                max_tokens=256,
            )
            text = (res.choices[0].message.content or "").strip()
            for line in text.splitlines():
                t = line.strip("-• ").strip()
                if t:
                    variants.append(t)
        except Exception:
            pass

        # LLM 실패/부족 시 키워드 기반 보조 변형
        try:
            kws = self.extract_keywords(query, top_n=5)
            if kws:
                variants.append(query + " " + " ".join(kws[:3]))
        except Exception:
            pass

        # 최소한 원 쿼리는 포함
        uniq = [query] + [v for v in variants if v and v != query]
        return uniq[: n + 1]
    
    def rag_pipeline(self, query: str, keywords: Optional[List[str]] = None) -> dict:
        # retriever의 top_k는 pool 용도로!
        docs = self._retrieve_candidates(query)
        doc_keyword = self.extract_doc_keyword(query)

        if doc_keyword:  # 빈 문자열이면 건너뜀
            def get_filename(src): return os.path.splitext(os.path.basename(src or ""))[0]
            narrowed = [d for d in docs if doc_keyword in get_filename(d.metadata.get("source", ""))]
            if narrowed:
                filtered_docs = narrowed
            else:
                filtered_docs = docs
        else:
            filtered_docs = docs

        # 4) 현재 질문에서 키워드 추출하여 기억에 저장
        extracted_keywords = self.extract_keywords(query)
        self.keyword_memory.add_keywords(extracted_keywords)

        # 5) 이전 대화 키워드를 텍스트 문맥으로 준비
        past_keywords_list = self.keyword_memory.get_all_keywords()
        past_keywords_text = ", ".join(past_keywords_list) if past_keywords_list else "없음"

        # 6) CE 재랭크 → 상위 top_k
        ce_pool = filtered_docs[: max(self.top_k * 3, 30)]
        reranked = self.rerank_by_distance(query, ce_pool, top_k=self.top_k)

        # 7) LLM 요약 및 답변 생성 (이전 키워드 문맥 포함)
        if self.openai:
            summary = self.summarize_chunks(reranked[: self.top_k], query, past_keywords_text)
        else:
            summary = self._basic_summarize_with_keywords(reranked[: self.top_k], query, past_keywords_text)

        return {
            "response": summary,
            "retrieved_docs": reranked[: self.top_k],
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
