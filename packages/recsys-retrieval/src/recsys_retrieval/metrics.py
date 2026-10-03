import math


def hit_rate_at_k(relevant_idx: int, ranked_list: list[int]) -> float:
    return 1.0 if relevant_idx in ranked_list else 0.0


def mrr_at_k(relevant_idx: int, ranked_list: list[int]) -> float:
    for rank, idx in enumerate(ranked_list, 1):
        if idx == relevant_idx:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(relevant_idx: int, ranked_list: list[int]) -> float:
    for rank, idx in enumerate(ranked_list, 1):
        if idx == relevant_idx:
            return 1.0 / math.log2(rank + 1)
    return 0.0
