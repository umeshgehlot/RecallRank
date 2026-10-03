import torch
from recsys_retrieval.model import ItemTower, TwoTowerModel, UserTower


def _user_tower_config():
    return {
        "user_id_vocab_size": 100,
        "item_id_vocab_size": 50,
        "category_vocab_size": 10,
        "device_type_vocab_size": 3,
        "embedding_dim": 16,
        "mlp_dims": [64, 32, 16],
    }


def _item_tower_config():
    return {
        "item_id_vocab_size": 50,
        "category_vocab_size": 10,
        "embedding_dim": 16,
        "mlp_dims": [64, 32, 16],
        "price_buckets": 10,
    }


def test_user_tower_output_shape():
    cfg = _user_tower_config()
    tower = UserTower(**cfg)
    batch = {
        "user_id_idx": torch.tensor([1, 2]),
        "user_age_scaled": torch.tensor([0.5, -0.3]),
        "historic_category_indices": torch.tensor([[1, 2, 0, 0, 0], [3, 0, 0, 0, 0]]),
        "historic_category_mask": torch.tensor([[1, 1, 0, 0, 0], [1, 0, 0, 0, 0]]),
        "session_click_indices": torch.tensor([[1, 2, 0], [3, 0, 0]]),
        "session_click_mask": torch.tensor([[1, 1, 0], [1, 0, 0]]),
        "device_type_idx": torch.tensor([1, 2]),
        "hour_of_day_scaled": torch.tensor([0.2, -0.5]),
    }
    out = tower(batch)
    assert out.shape == (2, 16)


def test_user_tower_output_is_l2_normalized():
    cfg = _user_tower_config()
    tower = UserTower(**cfg)
    batch = {
        "user_id_idx": torch.tensor([1, 2, 3]),
        "user_age_scaled": torch.tensor([0.5, -0.3, 1.0]),
        "historic_category_indices": torch.tensor([[1, 2, 0, 0, 0]] * 3),
        "historic_category_mask": torch.tensor([[1, 1, 0, 0, 0]] * 3),
        "session_click_indices": torch.tensor([[1, 0, 0]] * 3),
        "session_click_mask": torch.tensor([[1, 0, 0]] * 3),
        "device_type_idx": torch.tensor([1, 1, 2]),
        "hour_of_day_scaled": torch.tensor([0.2, -0.5, 0.0]),
    }
    out = tower(batch)
    norms = torch.norm(out, dim=1)
    assert torch.allclose(norms, torch.ones(3), atol=1e-5)


def test_item_tower_output_shape():
    cfg = _item_tower_config()
    tower = ItemTower(**cfg)
    batch = {
        "item_id_idx": torch.tensor([1, 2]),
        "category_idx": torch.tensor([3, 4]),
        "ctr_scaled": torch.tensor([0.1, -0.2]),
        "price_bucket": torch.tensor([5, 3]),
    }
    out = tower(batch)
    assert out.shape == (2, 16)


def test_item_tower_output_is_l2_normalized():
    cfg = _item_tower_config()
    tower = ItemTower(**cfg)
    batch = {
        "item_id_idx": torch.tensor([1, 2, 3]),
        "category_idx": torch.tensor([3, 4, 5]),
        "ctr_scaled": torch.tensor([0.1, -0.2, 0.5]),
        "price_bucket": torch.tensor([5, 3, 7]),
    }
    out = tower(batch)
    norms = torch.norm(out, dim=1)
    assert torch.allclose(norms, torch.ones(3), atol=1e-5)


def test_two_tower_model_forward():
    user_cfg = _user_tower_config()
    item_cfg = _item_tower_config()
    model = TwoTowerModel(user_tower=UserTower(**user_cfg), item_tower=ItemTower(**item_cfg))
    user_batch = {
        "user_id_idx": torch.tensor([1]),
        "user_age_scaled": torch.tensor([0.5]),
        "historic_category_indices": torch.tensor([[1, 2, 0, 0, 0]]),
        "historic_category_mask": torch.tensor([[1, 1, 0, 0, 0]]),
        "session_click_indices": torch.tensor([[1, 0, 0]]),
        "session_click_mask": torch.tensor([[1, 0, 0]]),
        "device_type_idx": torch.tensor([1]),
        "hour_of_day_scaled": torch.tensor([0.2]),
    }
    item_batch = {
        "item_id_idx": torch.tensor([1]),
        "category_idx": torch.tensor([3]),
        "ctr_scaled": torch.tensor([0.1]),
        "price_bucket": torch.tensor([5]),
    }
    u, v = model(user_batch, item_batch)
    assert u.shape == (1, 16)
    assert v.shape == (1, 16)
