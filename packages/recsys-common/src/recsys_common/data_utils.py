from __future__ import annotations

from pathlib import Path

import pyarrow.parquet as pq


def load_user_map(raw_dir: Path) -> dict[str, dict]:
    users = pq.read_table(raw_dir / "users.parquet")
    result: dict[str, dict] = {}
    for i in range(len(users)):
        uid = users.column("user_id")[i].as_py()
        result[uid] = {
            "user_age": users.column("user_age")[i].as_py(),
            "user_historic_categories": users.column("user_historic_categories")[i].as_py(),
            "session_last_3_clicks": users.column("session_last_3_clicks")[i].as_py(),
        }
    return result


def load_item_map(raw_dir: Path) -> dict[str, dict]:
    items = pq.read_table(raw_dir / "items.parquet")
    result: dict[str, dict] = {}
    for i in range(len(items)):
        iid = items.column("item_id")[i].as_py()
        result[iid] = {
            "item_category": items.column("item_category")[i].as_py(),
            "item_historical_ctr": items.column("item_historical_ctr")[i].as_py(),
            "item_price": items.column("item_price")[i].as_py(),
        }
    return result
