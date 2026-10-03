import math

from recsys_retrieval.metrics import hit_rate_at_k, mrr_at_k, ndcg_at_k


def test_hit_rate_present():
    assert hit_rate_at_k(3, [1, 2, 3, 4, 5]) == 1.0


def test_hit_rate_absent():
    assert hit_rate_at_k(9, [1, 2, 3, 4, 5]) == 0.0


def test_mrr_first_position():
    assert mrr_at_k(1, [1, 2, 3]) == 1.0


def test_mrr_third_position():
    assert mrr_at_k(3, [1, 2, 3]) == 1.0 / 3


def test_mrr_absent():
    assert mrr_at_k(9, [1, 2, 3]) == 0.0


def test_ndcg_first_position():
    assert ndcg_at_k(1, [1, 2, 3]) == 1.0 / math.log2(2)


def test_ndcg_third_position():
    assert ndcg_at_k(3, [1, 2, 3]) == 1.0 / math.log2(4)


def test_ndcg_absent():
    assert ndcg_at_k(9, [1, 2, 3]) == 0.0


def test_empty_ranked_list():
    assert hit_rate_at_k(1, []) == 0.0
    assert mrr_at_k(1, []) == 0.0
    assert ndcg_at_k(1, []) == 0.0
