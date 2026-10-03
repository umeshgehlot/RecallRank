from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq


@dataclass
class SyntheticConfig:
    num_users: int = 10_000
    num_items: int = 50_000
    num_interactions: int = 500_000
    num_categories: int = 20
    seed: int = 42


def generate_synthetic_data(cfg: SyntheticConfig, output_dir: Path) -> None:
    rng = np.random.default_rng(cfg.seed)
    output_dir.mkdir(parents=True, exist_ok=True)
    categories = [f"cat_{i}" for i in range(cfg.num_categories)]

    # Users
    user_ids = [f"user_{i}" for i in range(cfg.num_users)]
    user_ages = rng.uniform(18, 80, cfg.num_users).round(1)
    user_hist_cats = []
    for _ in range(cfg.num_users):
        n_cats = rng.integers(2, 6)
        weights = rng.dirichlet(np.ones(cfg.num_categories) * 0.3)
        chosen = rng.choice(categories, size=n_cats, replace=False, p=weights)
        user_hist_cats.append(chosen.tolist())

    # Items
    item_ids = [f"item_{i}" for i in range(cfg.num_items)]
    item_categories = rng.choice(categories, size=cfg.num_items).tolist()
    item_ctrs = rng.beta(2, 50, cfg.num_items).round(4)
    item_prices = rng.lognormal(mean=3.5, sigma=1.0, size=cfg.num_items).round(2)
    item_popularity = rng.pareto(a=1.5, size=cfg.num_items)
    item_popularity = item_popularity / item_popularity.sum()

    # Interactions
    interaction_user_ids = []
    interaction_item_ids = []
    clicked_list = []
    converted_list = []
    device_types = []
    hours = []

    user_cat_map = {uid: set(cats) for uid, cats in zip(user_ids, user_hist_cats)}
    item_cat_map = {iid: cat for iid, cat in zip(item_ids, item_categories)}

    for _ in range(cfg.num_interactions):
        uid = rng.choice(user_ids)
        iid = rng.choice(item_ids, p=item_popularity)
        user_cats = user_cat_map[uid]
        item_cat = item_cat_map[iid]
        click_prob = 0.3 if item_cat in user_cats else 0.05
        click = int(rng.random() < click_prob)
        # Conversion only possible if clicked, ~20% conversion rate on clicks
        converted = int(click == 1 and rng.random() < 0.2)

        interaction_user_ids.append(uid)
        interaction_item_ids.append(iid)
        clicked_list.append(click)
        converted_list.append(converted)
        device_types.append(rng.choice(["mobile", "desktop"]))
        hours.append(int(rng.integers(0, 24)))

    # Build session_last_3_clicks
    user_click_history: dict[str, list[str]] = {uid: [] for uid in user_ids}
    for uid, iid, c in zip(interaction_user_ids, interaction_item_ids, clicked_list):
        if c == 1:
            user_click_history[uid].append(iid)
    session_clicks = [user_click_history[uid][-3:] for uid in user_ids]

    # Write users
    users_table = pa.table(
        {
            "user_id": user_ids,
            "user_age": user_ages,
            "user_historic_categories": user_hist_cats,
            "session_last_3_clicks": session_clicks,
        }
    )
    pq.write_table(users_table, output_dir / "users.parquet")

    # Write items
    items_table = pa.table(
        {
            "item_id": item_ids,
            "item_category": item_categories,
            "item_historical_ctr": item_ctrs,
            "item_price": item_prices,
        }
    )
    pq.write_table(items_table, output_dir / "items.parquet")

    # Write interactions
    interactions_table = pa.table(
        {
            "user_id": interaction_user_ids,
            "item_id": interaction_item_ids,
            "clicked": clicked_list,
            "converted": converted_list,
            "device_type": device_types,
            "hour_of_day": hours,
        }
    )
    pq.write_table(interactions_table, output_dir / "interactions.parquet")
