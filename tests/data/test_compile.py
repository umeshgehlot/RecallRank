import json
from pathlib import Path

import pyarrow.parquet as pq
from recsys_data.compile import compile_dataset
from recsys_data.synthetic import SyntheticConfig, generate_synthetic_data


def _generate_small_dataset(tmp_path: Path) -> Path:
    raw_dir = tmp_path / "raw"
    cfg = SyntheticConfig(num_users=50, num_items=100, num_interactions=300, num_categories=5)
    generate_synthetic_data(cfg, raw_dir)
    return raw_dir


def test_compile_creates_artifacts(tmp_path):
    raw_dir = _generate_small_dataset(tmp_path)
    artifacts_dir = tmp_path / "artifacts"
    compile_dataset(raw_dir, artifacts_dir)

    assert (artifacts_dir / "user_id_vocab.json").exists()
    assert (artifacts_dir / "item_id_vocab.json").exists()
    assert (artifacts_dir / "category_vocab.json").exists()
    assert (artifacts_dir / "device_type_vocab.json").exists()
    assert (artifacts_dir / "stats.json").exists()
    assert (artifacts_dir / "popularity.json").exists()


def test_compile_creates_train_val_test_splits(tmp_path):
    raw_dir = _generate_small_dataset(tmp_path)
    artifacts_dir = tmp_path / "artifacts"
    compile_dataset(raw_dir, artifacts_dir)

    assert (artifacts_dir / "train.parquet").exists()
    assert (artifacts_dir / "val.parquet").exists()
    assert (artifacts_dir / "test.parquet").exists()

    train = pq.read_table(artifacts_dir / "train.parquet")
    val = pq.read_table(artifacts_dir / "val.parquet")
    test = pq.read_table(artifacts_dir / "test.parquet")
    total = len(train) + len(val) + len(test)
    assert total == 300


def test_stats_has_expected_fields(tmp_path):
    raw_dir = _generate_small_dataset(tmp_path)
    artifacts_dir = tmp_path / "artifacts"
    compile_dataset(raw_dir, artifacts_dir)

    stats = json.loads((artifacts_dir / "stats.json").read_text())
    assert "user_age" in stats
    assert "mu" in stats["user_age"]
    assert "sigma" in stats["user_age"]
    assert "item_price_boundaries" in stats
    assert len(stats["item_price_boundaries"]) == 9  # B=10 means 9 boundaries


def test_popularity_sums_to_one(tmp_path):
    raw_dir = _generate_small_dataset(tmp_path)
    artifacts_dir = tmp_path / "artifacts"
    compile_dataset(raw_dir, artifacts_dir)

    pop = json.loads((artifacts_dir / "popularity.json").read_text())
    total = sum(pop.values())
    assert abs(total - 1.0) < 1e-6


def test_vocab_index_zero_is_unk(tmp_path):
    raw_dir = _generate_small_dataset(tmp_path)
    artifacts_dir = tmp_path / "artifacts"
    compile_dataset(raw_dir, artifacts_dir)

    vocab = json.loads((artifacts_dir / "user_id_vocab.json").read_text())
    assert vocab["[UNK]"] == 0
