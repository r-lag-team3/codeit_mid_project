import os
import pickle
import json
import random
from typing import List, Dict
from openai import OpenAI

def load_chunks(pkl_path: str):
    with open(pkl_path, "rb") as f:
        return pickle.load(f)

def select_chunk_groups(chunks, n=10, min_chunks=1, max_chunks=6):
    chunk_groups = []
    total_chunks = len(chunks)
    for _ in range(n):
        group_size = random.randint(min_chunks, max_chunks)
        start = random.randint(0, total_chunks - group_size)
        group = [start]
        for i in range(1, group_size):
            if i < 3:
                group.append(group[-1] + 1)
            else:
                offset = random.choice([-2, -1, 1, 2])
                next_idx = min(max(group[-1] + offset, 0), total_chunks - 1)
                group.append(next_idx)
        group = sorted(set(group))
        chunk_groups.append(group)
    return chunk_groups

def make_prompt(chunks: List, chunk_ids: List[int]):
    chunk_texts = []
    doc_titles = set()
    for idx in chunk_ids:
        chunk_texts.append(f"[청크 {idx}]\n{chunks[idx].page_content}")
        # 문서 제목(파일명 등) 추출
        meta = getattr(chunks[idx], 'metadata', {})
        title = meta.get('source', None)
        if title:
            doc_titles.add(title)
    source = "\n\n".join(chunk_texts)
    titles_str = ", ".join(sorted(doc_titles)) if doc_titles else "(제목 정보 없음)"
    prompt = (
        "아래의 청크 내용을 참고하여, 반드시 아래 내용만을 바탕으로 자연스러운 질문과 그에 대한 답변을 만들어주세요.\n"
        "질문의 앞부분에는 반드시 청크가 포함되어있는 문서를 기반으로 기관이나 기업 명을 포함하여 질문을 만들어주세요.(질문 예시: 국민연금공단이 발주한 이러닝시스템 관련 사업 요구사항은 무엇인가요?)\n"
        "질문과 답변은 모두 한국어로 작성해주세요.\n"
        "답변은 반드시 아래 내용에서만 발췌하거나 요약해야 하며, 새로운 정보를 추가하지 마세요.\n"
        "청크에 없는 정보나 추론을 답변에 포함하지 마세요.\n"
        "형식: \n질문: ...\n답변: ...\n\n"
        f"사용된 청크 내용:\n{source}\n"
        f"사용된 문서 제목: {titles_str}\n"
    )
    return prompt

def generate_qa_pairs(chunks, chunk_groups, openai_key):
    client = OpenAI(api_key=openai_key)
    qa_pairs = []
    for group in chunk_groups:
        prompt = make_prompt(chunks, group)
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            max_tokens=512,
        )
        text = response.choices[0].message.content.strip()
        if "질문:" in text and "답변:" in text:
            q = text.split("질문:")[1].split("답변:")[0].strip()
            a = text.split("답변:")[1].strip()
        else:
            q, a = "질문 생성 실패", "답변 생성 실패"
        qa_pairs.append({
            "query": q,
            "answer": a,
            "chunk_id": group,
            "source": [chunks[i].page_content for i in group]
        })
    return qa_pairs

def save_span_list(qa_pairs: List[Dict], save_path: str):
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(qa_pairs, f, ensure_ascii=False, indent=2)

def generate_span(pkl_path: str, openai_key: str, n: int, save_path: str):
    chunks = load_chunks(pkl_path)
    chunk_groups = select_chunk_groups(chunks, n=n)
    qa_pairs = generate_qa_pairs(chunks, chunk_groups, openai_key)
    save_span_list(qa_pairs, save_path)