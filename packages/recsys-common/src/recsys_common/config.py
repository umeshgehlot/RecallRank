from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml


@dataclass(frozen=True)
class ModelConfig:
    embedding_dim: int = 64
    user_mlp_dims: list[int] = field(default_factory=lambda: [256, 128, 64])
    item_mlp_dims: list[int] = field(default_factory=lambda: [256, 128, 64])
    deep_mlp_dims: list[int] = field(default_factory=lambda: [512, 256, 128])
    deep_dropout: float = 0.2
    price_buckets: int = 10
    num_categories: int = 20
    max_historic_categories: int = 5
    max_session_clicks: int = 3
    retrieval_top_k: int = 200
    diversity_max_per_category: int = 3


def load_config(path: Path, overrides: dict | None = None) -> dict:
    with open(path) as f:
        cfg = yaml.safe_load(f) or {}
    if overrides:
        cfg.update({k: v for k, v in overrides.items() if v is not None})
    return cfg
