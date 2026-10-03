import argparse
import logging
from pathlib import Path

import pyarrow.parquet as pq
from recsys_common.schemas import RawItemFeatures, RawUserFeatures
from recsys_serving.feature_store import SqliteFeatureStore

logging.basicConfig(level=logging.INFO)


def main():
    parser = argparse.ArgumentParser(description="Populate feature store from parquet files")
    parser.add_argument("--artifacts-dir", type=Path, default=Path("artifacts"))
    parser.add_argument("--db-path", type=Path, default=Path("artifacts/features.db"))
    args = parser.parse_args()

    raw_dir = args.artifacts_dir / "raw"
    store = SqliteFeatureStore(args.db_path)

    users = pq.read_table(raw_dir / "users.parquet")
    logging.info(f"Loading {len(users)} users...")
    for i in range(len(users)):
        user = RawUserFeatures(
            user_id=users.column("user_id")[i].as_py(),
            user_age=users.column("user_age")[i].as_py(),
            user_historic_categories=users.column("user_historic_categories")[i].as_py(),
            session_last_3_clicks=users.column("session_last_3_clicks")[i].as_py(),
        )
        store.put_user_features(user.user_id, user)
        store.put_user_session(user.user_id, user.session_last_3_clicks)

    items = pq.read_table(raw_dir / "items.parquet")
    logging.info(f"Loading {len(items)} items...")
    for i in range(len(items)):
        item = RawItemFeatures(
            item_id=items.column("item_id")[i].as_py(),
            item_category=items.column("item_category")[i].as_py(),
            item_historical_ctr=items.column("item_historical_ctr")[i].as_py(),
            item_price=items.column("item_price")[i].as_py(),
        )
        store.put_item_features(item.item_id, item)

    logging.info(f"Feature store populated at {args.db_path}")


if __name__ == "__main__":
    main()
