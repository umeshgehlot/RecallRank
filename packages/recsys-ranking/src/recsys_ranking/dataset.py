from __future__ import annotations

from pathlib import Path

import pyarrow.parquet as pq
import torch
from recsys_common.config import ModelConfig
from recsys_common.data_utils import load_item_map, load_user_map
from recsys_common.processor import FeatureProcessor
from recsys_common.schemas import ContextFeatures, RawItemFeatures, RawUserFeatures
from recsys_common.transforms import pad_and_mask
from torch.utils.data import Dataset


class RankingDataset(Dataset):
    def __init__(self, artifacts_dir: Path, split: str = "train") -> None:
        self._processor = FeatureProcessor(artifacts_dir)
        self._config = ModelConfig()

        interactions = pq.read_table(artifacts_dir / f"{split}.parquet")
        self._user_ids = interactions.column("user_id").to_pylist()
        self._item_ids = interactions.column("item_id").to_pylist()
        self._clicked = interactions.column("clicked").to_pylist()
        self._converted = interactions.column("converted").to_pylist()
        self._device_types = interactions.column("device_type").to_pylist()
        self._hours = interactions.column("hour_of_day").to_pylist()

        raw_dir = artifacts_dir.parent / "raw"
        self._user_map = load_user_map(raw_dir)
        self._item_map = load_item_map(raw_dir)

    def __len__(self) -> int:
        return len(self._user_ids)

    def __getitem__(self, idx: int) -> dict:
        uid = self._user_ids[idx]
        iid = self._item_ids[idx]

        user_data = self._user_map.get(uid, {})
        user = RawUserFeatures(
            user_id=uid,
            user_age=user_data.get("user_age", 30.0),
            user_historic_categories=user_data.get("user_historic_categories", []),
            session_last_3_clicks=user_data.get("session_last_3_clicks", []),
        )
        ctx = ContextFeatures(
            device_type=self._device_types[idx],
            hour_of_day=self._hours[idx],
        )
        item_data = self._item_map[iid]
        item = RawItemFeatures(
            item_id=iid,
            item_category=item_data["item_category"],
            item_historical_ctr=item_data["item_historical_ctr"],
            item_price=item_data["item_price"],
        )

        pu = self._processor.process_user(user, ctx)
        pi = self._processor.process_item(item)

        cat_indices, cat_mask = pad_and_mask(
            pu.historic_category_indices, self._config.max_historic_categories
        )
        click_indices, click_mask = pad_and_mask(
            pu.session_click_indices, self._config.max_session_clicks
        )

        return {
            "user_id_idx": pu.user_id_idx,
            "item_id_idx": pi.item_id_idx,
            "category_idx": pi.category_idx,
            "device_type_idx": pu.device_type_idx,
            "price_bucket": pi.price_bucket,
            "user_age_scaled": pu.user_age_scaled,
            "hour_of_day_scaled": pu.hour_of_day_scaled,
            "ctr_scaled": pi.ctr_scaled,
            "historic_category_indices": cat_indices,
            "historic_category_mask": cat_mask,
            "session_click_indices": click_indices,
            "session_click_mask": click_mask,
            "clicked": float(self._clicked[idx]),
            "converted": float(self._converted[idx]),
        }


def ranking_collate_fn(batch: list[dict]) -> dict[str, torch.Tensor]:
    return {
        "user_id_idx": torch.tensor([b["user_id_idx"] for b in batch]),
        "item_id_idx": torch.tensor([b["item_id_idx"] for b in batch]),
        "category_idx": torch.tensor([b["category_idx"] for b in batch]),
        "device_type_idx": torch.tensor([b["device_type_idx"] for b in batch]),
        "price_bucket": torch.tensor([b["price_bucket"] for b in batch]),
        "user_age_scaled": torch.tensor([b["user_age_scaled"] for b in batch], dtype=torch.float),
        "hour_of_day_scaled": torch.tensor(
            [b["hour_of_day_scaled"] for b in batch], dtype=torch.float
        ),
        "ctr_scaled": torch.tensor([b["ctr_scaled"] for b in batch], dtype=torch.float),
        "historic_category_indices": torch.tensor([b["historic_category_indices"] for b in batch]),
        "historic_category_mask": torch.tensor([b["historic_category_mask"] for b in batch]),
        "session_click_indices": torch.tensor([b["session_click_indices"] for b in batch]),
        "session_click_mask": torch.tensor([b["session_click_mask"] for b in batch]),
        "clicked": torch.tensor([b["clicked"] for b in batch], dtype=torch.float),
        "converted": torch.tensor([b["converted"] for b in batch], dtype=torch.float),
    }
