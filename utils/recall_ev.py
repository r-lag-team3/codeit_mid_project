import json

def load_jsonl(path):
    results = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            item = json.loads(line)
            results [item["query"]] =item["indeices"]
    return results

def load_gold(path):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
        return {item["query"]: item["chunk_id"] for item in data}

def evaluate_recall_at_k(retrieved_dict, gold_dict, k=3):
    total_gold = 0
    total_hit = 0

    for query, retrieved_ids in retrieved_dict.items():
        gold_ids = gold_dict.get(query)
        if not gold_ids:
            continue

        retrieved_topk = set(retrieved_ids[:k])
        gold_set = set(gold_ids)

        hit_count = len(retrieved_topk & gold_set)  # 교집합
        total_hit += hit_count
        total_gold += len(gold_set)

    recall = total_hit / total_gold if total_gold else 0.0
    print(f"Strict Recall@{k}: {recall:.4f} ({total_hit}/{total_gold})")
    return recall