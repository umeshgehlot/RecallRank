from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field


@dataclass
class PostprocessConfig:
    max_per_category: int = 3
    top_n: int = 10
    business_multipliers: dict[str, float] = field(default_factory=dict)


def postprocess(
    candidates: list[dict], config: PostprocessConfig, top_n: int | None = None
) -> list[dict]:
    candidates = [dict(c) for c in candidates]

    for c in candidates:
        mult = config.business_multipliers.get(c["category"], 1.0)
        c["score"] = c["score"] * mult

    candidates.sort(key=lambda x: x["score"], reverse=True)

    seen_ids: set[str] = set()
    deduped: list[dict] = []
    for c in candidates:
        if c["item_id"] not in seen_ids:
            seen_ids.add(c["item_id"])
            deduped.append(c)

    cat_counts: dict[str, int] = defaultdict(int)
    diverse: list[dict] = []
    for c in deduped:
        if cat_counts[c["category"]] < config.max_per_category:
            diverse.append(c)
            cat_counts[c["category"]] += 1

    limit = top_n if top_n is not None else config.top_n
    return diverse[:limit]
