from __future__ import annotations

from pathlib import Path

import numpy as np
import onnxruntime as ort
from recsys_common.config import ModelConfig
from recsys_common.processor import FeatureProcessor
from recsys_common.schemas import ContextFeatures, RawUserFeatures
from recsys_common.transforms import pad_and_mask
from recsys_retrieval.index import FaissIndex

from recsys_serving.feature_store import FeatureStore
from recsys_serving.postprocess import PostprocessConfig, postprocess


class RecommendationPipeline:
    def __init__(
        self,
        artifacts_dir: Path,
        retrieval_model_dir: Path,
        ranking_model_dir: Path,
        feature_store: FeatureStore,
        postprocess_config: PostprocessConfig | None = None,
    ) -> None:
        self._processor = FeatureProcessor(artifacts_dir)
        self._feature_store = feature_store
        self._config = ModelConfig()
        self._pp_config = postprocess_config or PostprocessConfig()

        self._user_tower = ort.InferenceSession(str(retrieval_model_dir / "user_tower.onnx"))
        self._ranking_model = ort.InferenceSession(str(ranking_model_dir / "mmoe_deepfm.onnx"))
        self._index = FaissIndex.load(retrieval_model_dir / "faiss_index")

    def recommend(
        self,
        user_id: str,
        device_type: str,
        hour_of_day: int,
        top_n: int = 10,
        retrieval_k: int = 200,
    ) -> list[dict]:
        raw_user = self._feature_store.get_user_features(user_id)
        if raw_user is None:
            raw_user = RawUserFeatures(user_id=user_id, user_age=30.0)

        session_clicks = self._feature_store.get_user_session(user_id)
        raw_user = RawUserFeatures(
            user_id=raw_user.user_id,
            user_age=raw_user.user_age,
            user_historic_categories=raw_user.user_historic_categories,
            session_last_3_clicks=(
                session_clicks[-3:] if session_clicks else raw_user.session_last_3_clicks
            ),
        )

        ctx = ContextFeatures(device_type=device_type, hour_of_day=hour_of_day)
        pu = self._processor.process_user(raw_user, ctx)

        max_cats = self._config.max_historic_categories
        max_clicks = self._config.max_session_clicks
        cat_indices, cat_mask = pad_and_mask(pu.historic_category_indices, max_cats)
        click_indices, click_mask = pad_and_mask(pu.session_click_indices, max_clicks)

        user_inputs = {
            "user_id_idx": np.array([pu.user_id_idx], dtype=np.int64),
            "user_age_scaled": np.array([pu.user_age_scaled], dtype=np.float32),
            "historic_category_indices": np.array([cat_indices], dtype=np.int64),
            "historic_category_mask": np.array([cat_mask], dtype=np.int64),
            "session_click_indices": np.array([click_indices], dtype=np.int64),
            "session_click_mask": np.array([click_mask], dtype=np.int64),
            "device_type_idx": np.array([pu.device_type_idx], dtype=np.int64),
            "hour_of_day_scaled": np.array([pu.hour_of_day_scaled], dtype=np.float32),
        }
        user_vector = self._user_tower.run(None, user_inputs)[0][0]

        candidates = self._index.query(user_vector.astype(np.float32), top_k=retrieval_k)
        candidate_ids = [c[0] for c in candidates]

        raw_items = self._feature_store.get_item_features(candidate_ids)
        if not raw_items:
            return []

        valid_ids = [iid for iid in candidate_ids if iid in raw_items]
        processed_items = [self._processor.process_item(raw_items[iid]) for iid in valid_ids]

        n = len(processed_items)
        ranking_inputs = {
            "user_id_idx": np.full(n, pu.user_id_idx, dtype=np.int64),
            "item_id_idx": np.array([pi.item_id_idx for pi in processed_items], dtype=np.int64),
            "category_idx": np.array([pi.category_idx for pi in processed_items], dtype=np.int64),
            "device_type_idx": np.full(n, pu.device_type_idx, dtype=np.int64),
            "price_bucket": np.array([pi.price_bucket for pi in processed_items], dtype=np.int64),
            "user_age_scaled": np.full(n, pu.user_age_scaled, dtype=np.float32),
            "hour_of_day_scaled": np.full(n, pu.hour_of_day_scaled, dtype=np.float32),
            "ctr_scaled": np.array([pi.ctr_scaled for pi in processed_items], dtype=np.float32),
            "historic_category_indices": np.tile(cat_indices, (n, 1)).astype(np.int64),
            "historic_category_mask": np.tile(cat_mask, (n, 1)).astype(np.int64),
            "session_click_indices": np.tile(click_indices, (n, 1)).astype(np.int64),
            "session_click_mask": np.tile(click_mask, (n, 1)).astype(np.int64),
        }

        ctr_logits, cvr_logits = self._ranking_model.run(None, ranking_inputs)
        ctr_probs = 1.0 / (1.0 + np.exp(-ctr_logits.flatten()))
        cvr_probs = 1.0 / (1.0 + np.exp(-cvr_logits.flatten()))

        result_candidates = []
        for i, iid in enumerate(valid_ids):
            item_price = raw_items[iid].item_price
            ev_score = float(ctr_probs[i] * cvr_probs[i] * item_price)
            result_candidates.append(
                {
                    "item_id": iid,
                    "score": ev_score,
                    "ctr_score": float(ctr_probs[i]),
                    "cvr_score": float(cvr_probs[i]),
                    "category": raw_items[iid].item_category,
                }
            )

        final = postprocess(result_candidates, self._pp_config, top_n=top_n)

        return [
            {
                "item_id": r["item_id"],
                "score": r["score"],
                "ctr_score": r["ctr_score"],
                "cvr_score": r["cvr_score"],
                "rank": i + 1,
            }
            for i, r in enumerate(final)
        ]
