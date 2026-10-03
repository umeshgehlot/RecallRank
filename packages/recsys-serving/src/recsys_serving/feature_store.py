from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Protocol

from recsys_common.schemas import RawItemFeatures, RawUserFeatures


class FeatureStore(Protocol):
    def get_user_features(self, user_id: str) -> RawUserFeatures | None: ...
    def get_item_features(self, item_ids: list[str]) -> dict[str, RawItemFeatures]: ...
    def get_user_session(self, user_id: str) -> list[str]: ...


class SqliteFeatureStore:
    def __init__(self, db_path: Path) -> None:
        self._conn = sqlite3.connect(str(db_path), check_same_thread=False)
        self._conn.execute("CREATE TABLE IF NOT EXISTS users (user_id TEXT PRIMARY KEY, data TEXT)")
        self._conn.execute("CREATE TABLE IF NOT EXISTS items (item_id TEXT PRIMARY KEY, data TEXT)")
        self._conn.execute(
            "CREATE TABLE IF NOT EXISTS sessions (user_id TEXT PRIMARY KEY, data TEXT)"
        )
        self._conn.commit()

    def put_user_features(self, user_id: str, features: RawUserFeatures) -> None:
        self._conn.execute(
            "INSERT OR REPLACE INTO users (user_id, data) VALUES (?, ?)",
            (user_id, features.model_dump_json()),
        )
        self._conn.commit()

    def get_user_features(self, user_id: str) -> RawUserFeatures | None:
        row = self._conn.execute("SELECT data FROM users WHERE user_id = ?", (user_id,)).fetchone()
        if row is None:
            return None
        return RawUserFeatures.model_validate_json(row[0])

    def put_item_features(self, item_id: str, features: RawItemFeatures) -> None:
        self._conn.execute(
            "INSERT OR REPLACE INTO items (item_id, data) VALUES (?, ?)",
            (item_id, features.model_dump_json()),
        )
        self._conn.commit()

    def get_item_features(self, item_ids: list[str]) -> dict[str, RawItemFeatures]:
        placeholders = ",".join("?" for _ in item_ids)
        rows = self._conn.execute(
            f"SELECT item_id, data FROM items WHERE item_id IN ({placeholders})",
            item_ids,
        ).fetchall()
        return {row[0]: RawItemFeatures.model_validate_json(row[1]) for row in rows}

    def put_user_session(self, user_id: str, click_ids: list[str]) -> None:
        self._conn.execute(
            "INSERT OR REPLACE INTO sessions (user_id, data) VALUES (?, ?)",
            (user_id, json.dumps(click_ids)),
        )
        self._conn.commit()

    def get_user_session(self, user_id: str) -> list[str]:
        row = self._conn.execute(
            "SELECT data FROM sessions WHERE user_id = ?", (user_id,)
        ).fetchone()
        if row is None:
            return []
        return list(json.loads(row[0]))
