from collections import defaultdict

def reciprocal_rank_fusion(result_lists, k: int = 60, top_n: int | None = None) -> list[dict]:
    """Fuse ranked lists using RRF. Each result must have a stable `id`."""
    if k < 1:
        raise ValueError("k must be positive")
    scores = defaultdict(float)
    records = {}
    for results in result_lists:
        seen = set()
        for rank, item in enumerate(results, start=1):
            item_id = item.get("id")
            if item_id is None:
                raise ValueError("Each result must contain an 'id'")
            if item_id in seen:
                continue
            seen.add(item_id)
            scores[item_id] += 1.0 / (k + rank)
            records[item_id] = dict(item)
    ranked = []
    for item_id, score in scores.items():
        item = records[item_id]
        item["rrf_score"] = score
        ranked.append(item)
    ranked.sort(key=lambda item: (-item["rrf_score"], str(item["id"])))
    return ranked if top_n is None else ranked[:max(0, top_n)]
