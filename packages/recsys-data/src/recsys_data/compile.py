from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from recsys_common.vocab import VocabStore


def compile_dataset(raw_dir: Path, artifacts_dir: Path) -> None:
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    users = pq.read_table(raw_dir / "users.parquet")
    items = pq.read_table(raw_dir / "items.parquet")
    interactions = pq.read_table(raw_dir / "interactions.parquet")

    # --- Build and save vocabularies ---
    user_ids = users.column("user_id").to_pylist()
    user_vocab = VocabStore.build(user_ids)
    user_vocab.save(artifacts_dir / "user_id_vocab.json")

    item_ids = items.column("item_id").to_pylist()
    item_vocab = VocabStore.build(item_ids)
    item_vocab.save(artifacts_dir / "item_id_vocab.json")

    all_categories: list[str] = items.column("item_category").to_pylist()
    for cat_list in users.column("user_historic_categories").to_pylist():
        all_categories.extend(cat_list)
    cat_vocab = VocabStore.build(all_categories)
    cat_vocab.save(artifacts_dir / "category_vocab.json")

    device_types = interactions.column("device_type").to_pylist()
    device_vocab = VocabStore.build(device_types)
    device_vocab.save(artifacts_dir / "device_type_vocab.json")

    # --- Compute and save stats ---
    user_ages = np.array(users.column("user_age").to_pylist(), dtype=np.float64)
    hours = np.array(interactions.column("hour_of_day").to_pylist(), dtype=np.float64)
    ctrs = np.array(items.column("item_historical_ctr").to_pylist(), dtype=np.float64)
    prices = np.array(items.column("item_price").to_pylist(), dtype=np.float64)

    price_boundaries = np.quantile(prices, np.linspace(0.1, 0.9, 9)).tolist()

    stats = {
        "user_age": {"mu": float(user_ages.mean()), "sigma": float(user_ages.std())},
        "hour_of_day": {"mu": float(hours.mean()), "sigma": float(hours.std())},
        "item_historical_ctr": {"mu": float(ctrs.mean()), "sigma": float(ctrs.std())},
        "item_price_boundaries": price_boundaries,
    }
    (artifacts_dir / "stats.json").write_text(json.dumps(stats, indent=2))

    # --- Compute and save popularity ---
    item_counts = Counter(interactions.column("item_id").to_pylist())
    total = sum(item_counts.values())
    popularity = {iid: count / total for iid, count in item_counts.items()}
    (artifacts_dir / "popularity.json").write_text(json.dumps(popularity, indent=2))

    # --- Split interactions into train/val/test ---
    n = len(interactions)
    indices = np.arange(n)
    rng = np.random.default_rng(42)
    rng.shuffle(indices)

    train_end = int(0.8 * n)
    val_end = int(0.9 * n)

    train_indices = indices[:train_end].tolist()
    val_indices = indices[train_end:val_end].tolist()
    test_indices = indices[val_end:].tolist()

    pq.write_table(interactions.take(train_indices), artifacts_dir / "train.parquet")
    pq.write_table(interactions.take(val_indices), artifacts_dir / "val.parquet")
    pq.write_table(interactions.take(test_indices), artifacts_dir / "test.parquet")
