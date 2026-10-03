from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from recsys_common.schemas import ContextFeatures, RawItemFeatures, RawUserFeatures
from recsys_common.transforms import log_compress, quantile_bucket, z_score
from recsys_common.vocab import VocabStore


@dataclass
class ProcessedUserFeatures:
    user_id_idx: int
    user_age_scaled: float
    historic_category_indices: list[int]
    session_click_indices: list[int]
    device_type_idx: int
    hour_of_day_scaled: float


@dataclass
class ProcessedItemFeatures:
    item_id_idx: int
    category_idx: int
    ctr_scaled: float
    price_bucket: int


class FeatureProcessor:
    def __init__(self, artifacts_dir: Path) -> None:
        self._user_id_vocab = VocabStore.load(artifacts_dir / "user_id_vocab.json")
        self._item_id_vocab = VocabStore.load(artifacts_dir / "item_id_vocab.json")
        self._category_vocab = VocabStore.load(artifacts_dir / "category_vocab.json")
        self._device_type_vocab = VocabStore.load(artifacts_dir / "device_type_vocab.json")

        stats = json.loads((artifacts_dir / "stats.json").read_text())
        self._user_age_mu = stats["user_age"]["mu"]
        self._user_age_sigma = stats["user_age"]["sigma"]
        self._hour_mu = stats["hour_of_day"]["mu"]
        self._hour_sigma = stats["hour_of_day"]["sigma"]
        self._ctr_mu = stats["item_historical_ctr"]["mu"]
        self._ctr_sigma = stats["item_historical_ctr"]["sigma"]
        self._price_boundaries = stats["item_price_boundaries"]

    def process_user(self, user: RawUserFeatures, ctx: ContextFeatures) -> ProcessedUserFeatures:
        return ProcessedUserFeatures(
            user_id_idx=self._user_id_vocab.lookup(user.user_id),
            user_age_scaled=z_score(user.user_age, self._user_age_mu, self._user_age_sigma),
            historic_category_indices=[
                self._category_vocab.lookup(c) for c in user.user_historic_categories
            ],
            session_click_indices=[
                self._item_id_vocab.lookup(item_id) for item_id in user.session_last_3_clicks
            ],
            device_type_idx=self._device_type_vocab.lookup(ctx.device_type),
            hour_of_day_scaled=z_score(ctx.hour_of_day, self._hour_mu, self._hour_sigma),
        )

    def process_item(self, item: RawItemFeatures) -> ProcessedItemFeatures:
        return ProcessedItemFeatures(
            item_id_idx=self._item_id_vocab.lookup(item.item_id),
            category_idx=self._category_vocab.lookup(item.item_category),
            ctr_scaled=z_score(
                log_compress(item.item_historical_ctr), self._ctr_mu, self._ctr_sigma
            ),
            price_bucket=quantile_bucket(item.item_price, self._price_boundaries),
        )

    @property
    def user_id_vocab_size(self) -> int:
        return self._user_id_vocab.size()

    @property
    def item_id_vocab_size(self) -> int:
        return self._item_id_vocab.size()

    @property
    def category_vocab_size(self) -> int:
        return self._category_vocab.size()

    @property
    def device_type_vocab_size(self) -> int:
        return self._device_type_vocab.size()
