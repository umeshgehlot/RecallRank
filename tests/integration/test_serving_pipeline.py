from __future__ import annotations

import numpy as np
import onnxruntime as ort
import pyarrow.parquet as pq
import pytest
from recsys_common.processor import FeatureProcessor
from recsys_common.schemas import RawItemFeatures, RawUserFeatures
from recsys_data.compile import compile_dataset
from recsys_data.synthetic import SyntheticConfig, generate_synthetic_data
from recsys_ranking.train import TrainConfig as RankingCfg
from recsys_ranking.train import train_ranking
from recsys_retrieval.index import FaissIndex
from recsys_retrieval.train import TrainConfig as RetrievalCfg
from recsys_retrieval.train import train_retrieval
from recsys_serving.feature_store import SqliteFeatureStore
from recsys_serving.pipeline import RecommendationPipeline


@pytest.fixture(scope="module")
def serving_pipeline(tmp_path_factory):
    base = tmp_path_factory.mktemp("serving_e2e")
    raw_dir = base / "raw"
    compiled_dir = base / "compiled"
    retrieval_dir = base / "models" / "retrieval"
    ranking_dir = base / "models" / "ranking"

    generate_synthetic_data(
        SyntheticConfig(
            num_users=80, num_items=200, num_interactions=2000, num_categories=5, seed=42
        ),
        raw_dir,
    )
    compile_dataset(raw_dir, compiled_dir)

    train_retrieval(compiled_dir, retrieval_dir, RetrievalCfg(epochs=1, batch_size=64))
    train_ranking(compiled_dir, ranking_dir, RankingCfg(epochs=1, batch_size=64))

    processor = FeatureProcessor(compiled_dir)
    item_tower = ort.InferenceSession(str(retrieval_dir / "item_tower.onnx"))
    items = pq.read_table(raw_dir / "items.parquet")

    vectors = []
    item_ids = []
    batch_size = 128
    for start in range(0, len(items), batch_size):
        end = min(start + batch_size, len(items))
        batch_items = []
        for i in range(start, end):
            item = RawItemFeatures(
                item_id=items.column("item_id")[i].as_py(),
                item_category=items.column("item_category")[i].as_py(),
                item_historical_ctr=items.column("item_historical_ctr")[i].as_py(),
                item_price=items.column("item_price")[i].as_py(),
            )
            batch_items.append(processor.process_item(item))
            item_ids.append(item.item_id)

        inputs = {
            "item_id_idx": np.array([pi.item_id_idx for pi in batch_items], dtype=np.int64),
            "category_idx": np.array([pi.category_idx for pi in batch_items], dtype=np.int64),
            "ctr_scaled": np.array([pi.ctr_scaled for pi in batch_items], dtype=np.float32),
            "price_bucket": np.array([pi.price_bucket for pi in batch_items], dtype=np.int64),
        }
        vectors.append(item_tower.run(None, inputs)[0])

    all_vectors = np.vstack(vectors).astype(np.float32)
    index = FaissIndex(all_vectors.shape[1])
    index.build(all_vectors, item_ids)
    index.save(retrieval_dir / "faiss_index")

    db_path = base / "features.db"
    store = SqliteFeatureStore(db_path)

    for i in range(len(items)):
        item = RawItemFeatures(
            item_id=items.column("item_id")[i].as_py(),
            item_category=items.column("item_category")[i].as_py(),
            item_historical_ctr=items.column("item_historical_ctr")[i].as_py(),
            item_price=items.column("item_price")[i].as_py(),
        )
        store.put_item_features(item.item_id, item)

    users = pq.read_table(raw_dir / "users.parquet")
    for i in range(len(users)):
        user = RawUserFeatures(
            user_id=users.column("user_id")[i].as_py(),
            user_age=users.column("user_age")[i].as_py(),
            user_historic_categories=users.column("user_historic_categories")[i].as_py(),
            session_last_3_clicks=users.column("session_last_3_clicks")[i].as_py(),
        )
        store.put_user_features(user.user_id, user)
        store.put_user_session(user.user_id, user.session_last_3_clicks)

    pipeline = RecommendationPipeline(
        artifacts_dir=compiled_dir,
        retrieval_model_dir=retrieval_dir,
        ranking_model_dir=ranking_dir,
        feature_store=store,
    )
    return pipeline


def test_pipeline_returns_recommendations(serving_pipeline):
    results = serving_pipeline.recommend(
        user_id="user_0",
        device_type="mobile",
        hour_of_day=14,
        top_n=10,
        retrieval_k=50,
    )

    assert len(results) > 0
    assert len(results) <= 10
    for r in results:
        assert "item_id" in r
        assert "score" in r
        assert "ctr_score" in r
        assert "cvr_score" in r
        assert "rank" in r
        assert 0.0 <= r["ctr_score"] <= 1.0
        assert 0.0 <= r["cvr_score"] <= 1.0


def test_pipeline_handles_unknown_user(serving_pipeline):
    results = serving_pipeline.recommend(
        user_id="unknown_user_999",
        device_type="desktop",
        hour_of_day=9,
        top_n=5,
        retrieval_k=50,
    )

    assert len(results) > 0
    assert len(results) <= 5


def test_grpc_module_imports():
    from recsys_serving import recsys_pb2, recsys_pb2_grpc  # noqa: F401

    assert hasattr(recsys_pb2, "RecommendRequest")
    assert hasattr(recsys_pb2, "RecommendResponse")
    assert hasattr(recsys_pb2_grpc, "RecommendationServiceStub")
    assert hasattr(recsys_pb2_grpc, "RecommendationServiceServicer")
