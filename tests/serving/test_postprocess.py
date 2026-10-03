from recsys_serving.postprocess import PostprocessConfig, postprocess


def test_deduplication():
    candidates = [
        {"item_id": "i1", "score": 0.9, "category": "A"},
        {"item_id": "i1", "score": 0.8, "category": "A"},
        {"item_id": "i2", "score": 0.7, "category": "B"},
    ]
    result = postprocess(candidates, PostprocessConfig())
    item_ids = [r["item_id"] for r in result]
    assert item_ids.count("i1") == 1


def test_diversity_cap():
    candidates = [{"item_id": f"i{i}", "score": 1.0 - i * 0.1, "category": "A"} for i in range(10)]
    cfg = PostprocessConfig(max_per_category=3)
    result = postprocess(candidates, cfg)
    assert len(result) == 3


def test_top_n_limit():
    candidates = [
        {"item_id": f"i{i}", "score": 1.0 - i * 0.01, "category": f"cat_{i}"} for i in range(20)
    ]
    cfg = PostprocessConfig(top_n=5)
    result = postprocess(candidates, cfg)
    assert len(result) == 5


def test_business_multipliers():
    candidates = [
        {"item_id": "i1", "score": 0.5, "category": "promo"},
        {"item_id": "i2", "score": 0.8, "category": "regular"},
    ]
    cfg = PostprocessConfig(business_multipliers={"promo": 2.0})
    result = postprocess(candidates, cfg)
    assert result[0]["item_id"] == "i1"  # boosted to 1.0 > 0.8


def test_preserves_score_order():
    candidates = [
        {"item_id": "i1", "score": 0.3, "category": "A"},
        {"item_id": "i2", "score": 0.9, "category": "B"},
        {"item_id": "i3", "score": 0.6, "category": "C"},
    ]
    result = postprocess(candidates, PostprocessConfig())
    scores = [r["score"] for r in result]
    assert scores == sorted(scores, reverse=True)
