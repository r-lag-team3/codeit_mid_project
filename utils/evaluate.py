import json
from openai import OpenAI
import os

# 키를 환경변수로 설정 (주의: 실제 운영 시에는 .env 방식이 안전)
os.environ["OPENAI_API_KEY"] = ''

client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

def evaluate_rag_response_no_answer(query, response, context):
    prompt = f"""너는 RAG 시스템의 응답을 평가하는 AI 평가자입니다.

사용자의 질의:
{query}

RAG 시스템의 응답:
{response}

RAG가 참고한 문서(context):
{context}

다음 기준에 따라 평가하세요:
1. 응답이 질의에 대해 얼마나 정확한지 (정확성)
2. 참고 문서에 기반한 응답인지 여부 (근거 기반성)
3. 응답이 얼마나 이해하기 쉬운지 (명확성)

위 세 기준 각각에 대해 1~5 점으로 평가하고, 간단한 이유도 포함해주세요. 아래 형식을 따라 주세요:

정확성: 4/5 - 응답은 대부분 정확하지만 일부 세부정보 부족
근거 기반성: 5/5 - 응답은 명확히 문서 기반임
명확성: 5/5 - 간결하고 이해하기 쉬움
"""

    completion = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
    )

    return completion.choices[0].message.content

def evaluate_from_jsonl(jsonl_path):
    with open(jsonl_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    for i, line in enumerate(lines):
        item = json.loads(line)
        query = item["query"]
        response = item["response"]
        context = item["context"]

        print(f"\n===== 질의 {i+1} 평가 중 =====")
        evaluation = evaluate_rag_response_no_answer(query, response, context)
        print(evaluation)

if __name__ == "__main__":
    jsonl_path = "./experiment/dw_ex_1/rag_results.jsonl"  # 경로 확인
    evaluate_from_jsonl(jsonl_path)
