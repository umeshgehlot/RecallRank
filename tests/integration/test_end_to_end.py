import numpy as np
import pytest
import torch
from recsys_data.compile import compile_dataset
from recsys_data.synthetic import SyntheticConfig, generate_synthetic_data
from recsys_ranking.dataset import RankingDataset, ranking_collate_fn
from recsys_ranking.model import MMoEDeepFM
from recsys_retrieval.dataset import RetrievalDataset, retrieval_collate_fn
from recsys_retrieval.index import FaissIndex
from recsys_retrieval.loss import info_nce_loss_with_logq
from recsys_retrieval.model import ItemTower, TwoTowerModel, UserTower


@pytest.fixture(scope="module")
def compiled_artifacts(tmp_path_factory):
    base = tmp_path_factory.mktemp("e2e")
    raw_dir = base / "raw"
    compiled_dir = base / "compiled"
    cfg = SyntheticConfig(
        num_users=50, num_items=100, num_interactions=500, num_categories=5, seed=42
    )
    generate_synthetic_data(cfg, raw_dir)
    compile_dataset(raw_dir, compiled_dir)
    return compiled_dir


def test_retrieval_train_one_epoch(compiled_artifacts):
    ds = RetrievalDataset(compiled_artifacts, split="train")
    loader = torch.utils.data.DataLoader(
        ds, batch_size=32, shuffle=True, collate_fn=retrieval_collate_fn
    )

    processor = ds._processor
    model = TwoTowerModel(
        user_tower=UserTower(
            user_id_vocab_size=processor.user_id_vocab_size,
            item_id_vocab_size=processor.item_id_vocab_size,
            category_vocab_size=processor.category_vocab_size,
            device_type_vocab_size=processor.device_type_vocab_size,
            embedding_dim=16,
            mlp_dims=[32, 16],
        ),
        item_tower=ItemTower(
            item_id_vocab_size=processor.item_id_vocab_size,
            category_vocab_size=processor.category_vocab_size,
            embedding_dim=16,
            mlp_dims=[32, 16],
            price_buckets=10,
        ),
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    model.train()
    losses = []
    for user_batch, item_batch, log_q in loader:
        u, v = model(user_batch, item_batch)
        loss = info_nce_loss_with_logq(u, v, log_q)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

    assert len(losses) > 0
    assert all(val > 0 for val in losses)


def test_faiss_index_from_model(compiled_artifacts):
    ds = RetrievalDataset(compiled_artifacts, split="train")
    processor = ds._processor
    item_tower = ItemTower(
        item_id_vocab_size=processor.item_id_vocab_size,
        category_vocab_size=processor.category_vocab_size,
        embedding_dim=16,
        mlp_dims=[32, 16],
        price_buckets=10,
    )
    item_tower.eval()

    batch = {
        "item_id_idx": torch.arange(1, 11),
        "category_idx": torch.randint(1, 5, (10,)),
        "ctr_scaled": torch.randn(10),
        "price_bucket": torch.randint(0, 10, (10,)),
    }
    with torch.no_grad():
        vectors = item_tower(batch).numpy()

    ids = [f"item_{i}" for i in range(10)]
    index = FaissIndex(16)
    index.build(vectors.astype(np.float32), ids)

    results = index.query(vectors[0].astype(np.float32), top_k=5)
    assert len(results) == 5
    assert results[0][0] == "item_0"


def test_ranking_train_one_epoch(compiled_artifacts):
    ds = RankingDataset(compiled_artifacts, split="train")
    loader = torch.utils.data.DataLoader(
        ds, batch_size=32, shuffle=True, collate_fn=ranking_collate_fn
    )

    processor = ds._processor
    model = MMoEDeepFM(
        user_id_vocab_size=processor.user_id_vocab_size,
        item_id_vocab_size=processor.item_id_vocab_size,
        category_vocab_size=processor.category_vocab_size,
        device_type_vocab_size=processor.device_type_vocab_size,
        embedding_dim=16,
        price_buckets=10,
        num_experts=2,
        expert_dims=[32, 16],
        tower_dim=8,
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    ctr_loss_fn = torch.nn.BCEWithLogitsLoss()
    cvr_loss_fn = torch.nn.BCEWithLogitsLoss(reduction="none")

    model.train()
    losses = []
    for batch in loader:
        clicked = batch.pop("clicked")
        converted = batch.pop("converted")

        ctr_logits, cvr_logits = model(batch)

        ctr_loss = ctr_loss_fn(ctr_logits, clicked)
        cvr_per_sample = cvr_loss_fn(cvr_logits, converted)
        cvr_loss = (cvr_per_sample * clicked).sum() / clicked.sum().clamp(min=1)

        loss = ctr_loss + cvr_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

    assert len(losses) > 0
    assert all(val > 0 for val in losses)


def test_full_pipeline_produces_ranked_results(compiled_artifacts):
    """End-to-end: generate -> compile -> train 1 epoch -> index -> rank -> results"""
    ds = RetrievalDataset(compiled_artifacts, split="train")
    processor = ds._processor

    user_tower = UserTower(
        user_id_vocab_size=processor.user_id_vocab_size,
        item_id_vocab_size=processor.item_id_vocab_size,
        category_vocab_size=processor.category_vocab_size,
        device_type_vocab_size=processor.device_type_vocab_size,
        embedding_dim=16,
        mlp_dims=[32, 16],
    )
    item_tower = ItemTower(
        item_id_vocab_size=processor.item_id_vocab_size,
        category_vocab_size=processor.category_vocab_size,
        embedding_dim=16,
        mlp_dims=[32, 16],
        price_buckets=10,
    )
    item_tower.eval()

    item_batch = {
        "item_id_idx": torch.arange(1, 21),
        "category_idx": torch.randint(1, 5, (20,)),
        "ctr_scaled": torch.randn(20),
        "price_bucket": torch.randint(0, 10, (20,)),
    }
    with torch.no_grad():
        item_vectors = item_tower(item_batch).numpy()

    item_ids = [f"item_{i}" for i in range(20)]
    index = FaissIndex(16)
    index.build(item_vectors.astype(np.float32), item_ids)

    user_tower.eval()
    user_batch = {
        "user_id_idx": torch.tensor([1]),
        "user_age_scaled": torch.tensor([0.0]),
        "historic_category_indices": torch.zeros(1, 5, dtype=torch.long),
        "historic_category_mask": torch.tensor([[1, 0, 0, 0, 0]]),
        "session_click_indices": torch.zeros(1, 3, dtype=torch.long),
        "session_click_mask": torch.tensor([[1, 0, 0]]),
        "device_type_idx": torch.tensor([1]),
        "hour_of_day_scaled": torch.tensor([0.0]),
    }
    with torch.no_grad():
        user_vector = user_tower(user_batch).numpy()[0]

    results = index.query(user_vector.astype(np.float32), top_k=10)
    assert len(results) == 10
    assert all(isinstance(r[0], str) for r in results)
    assert all(isinstance(r[1], float) for r in results)
