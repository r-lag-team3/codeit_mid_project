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

def f1_score_at_k(retrieved_dict, gold_dict, k=-1):
    total_true_positives = 0
    total_false_positives = 0
    total_false_negatives = 0
    total_gold = 0

    for query, retrieved_ids in retrieved_dict.items():
        gold_ids = gold_dict.get(query)
        if not gold_ids:
            continue
        
        retrieved_topk = set(retrieved_ids[:k]) if k > 0 else set(retrieved_ids)
        gold_set = set(gold_ids)

        true_positives = retrieved_topk.intersection(gold_set)
        false_positives = retrieved_topk - gold_set
        false_negatives = gold_set - retrieved_topk    

        total_true_positives += len(true_positives)
        total_false_positives += len(false_positives)
        total_false_negatives += len(false_negatives)

        total_gold += len(gold_set)

    total_precision = total_true_positives / total_gold if total_gold > 0 else 0
    total_recall = total_true_positives /total_gold if total_gold > 0 else 0
    total_f1_score = 2 * (total_precision * total_recall) / (total_precision + total_recall) if total_precision + total_recall > 0 else 0

    if k == -1:
        print(f"Strict precision: {total_precision:.4f}")
        print(f"Strict recall: {total_recall:.4f}")
        print(f"Strict f1_score: {total_f1_score:.4f}")
    else: 
        print(f"Strict precision@{k}: {total_precision:.4f}")
        print(f"Strict recall@{k}: {total_recall:.4f}")
        print(f"Strict f1_score@{k}: {total_f1_score:.4f}")

    return total_precision, total_recall, total_f1_score

def evaluate_ndcg_at_k(retrieved_dict, gold_dict, k=3):

    import math

    total_ndcg = 0.0
    query_count = 0

    for query, retrieved_ids in retrieved_dict.items():
        gold_ids = set(gold_dict.get(query, []))
        if not gold_ids:
            continue

        # DCG 계산
        dcg = 0.0
        for rank, chunk_id in enumerate(retrieved_ids[:k], start=1):
            if chunk_id in gold_ids:
                dcg += 1 / math.log2(rank + 1)

        # 이상적인 DCG(IDCG) 계산
        ideal_hits = min(len(gold_ids), k)
        idcg = sum(1 / math.log2(rank + 1) for rank in range(1, ideal_hits + 1))

        ndcg = dcg / idcg if idcg > 0 else 0.0
        total_ndcg += ndcg
        query_count += 1

    avg_ndcg = total_ndcg / query_count if query_count else 0.0
    print(f"nDCG@{k}: {avg_ndcg:.4f}")
    
    return avg_ndcg