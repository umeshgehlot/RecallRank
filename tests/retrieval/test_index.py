import numpy as np
from recsys_retrieval.index import FaissIndex


def test_build_and_query():
    dim = 16
    n_items = 100
    rng = np.random.default_rng(42)
    vectors = rng.standard_normal((n_items, dim)).astype(np.float32)
    ids = [f"item_{i}" for i in range(n_items)]

    index = FaissIndex(dim)
    index.build(vectors, ids)
    results = index.query(vectors[0], top_k=5)

    assert len(results) == 5
    assert results[0][0] == "item_0"  # closest to itself
    assert all(isinstance(r[1], float) for r in results)


def test_query_returns_top_k():
    dim = 8
    vectors = np.eye(dim, dtype=np.float32)
    ids = [f"item_{i}" for i in range(dim)]

    index = FaissIndex(dim)
    index.build(vectors, ids)

    results = index.query(vectors[3], top_k=3)
    assert len(results) == 3
    assert results[0][0] == "item_3"


def test_save_and_load(tmp_path):
    dim = 8
    rng = np.random.default_rng(0)
    vectors = rng.standard_normal((20, dim)).astype(np.float32)
    ids = [f"item_{i}" for i in range(20)]

    index = FaissIndex(dim)
    index.build(vectors, ids)
    index.save(tmp_path / "index")

    loaded = FaissIndex.load(tmp_path / "index")
    results_orig = index.query(vectors[0], top_k=5)
    results_loaded = loaded.query(vectors[0], top_k=5)

    assert [r[0] for r in results_orig] == [r[0] for r in results_loaded]
