import torch
import torch.nn as nn
from recsys_ranking.model import CrossLayer, DCNExpert, MMoEDeepFM


def _make_config():
    return {
        "user_id_vocab_size": 100,
        "item_id_vocab_size": 200,
        "category_vocab_size": 10,
        "device_type_vocab_size": 3,
        "embedding_dim": 16,
        "price_buckets": 10,
        "num_experts": 3,
        "expert_dims": [64, 32],
        "tower_dim": 16,
        "expert_dropout": 0.1,
        "max_historic_categories": 5,
        "max_session_clicks": 3,
    }


def _make_batch(batch_size=4):
    return {
        "user_id_idx": torch.randint(1, 100, (batch_size,)),
        "item_id_idx": torch.randint(1, 200, (batch_size,)),
        "category_idx": torch.randint(1, 10, (batch_size,)),
        "device_type_idx": torch.randint(1, 3, (batch_size,)),
        "price_bucket": torch.randint(0, 10, (batch_size,)),
        "user_age_scaled": torch.randn(batch_size),
        "hour_of_day_scaled": torch.randn(batch_size),
        "ctr_scaled": torch.randn(batch_size),
        "historic_category_indices": torch.randint(0, 10, (batch_size, 5)),
        "historic_category_mask": torch.tensor([[1, 1, 1, 0, 0]] * batch_size),
        "session_click_indices": torch.randint(0, 200, (batch_size, 3)),
        "session_click_mask": torch.tensor([[1, 1, 0]] * batch_size),
    }


def test_output_shapes():
    model = MMoEDeepFM(**_make_config())
    batch = _make_batch(8)
    ctr_logits, cvr_logits = model(batch)
    assert ctr_logits.shape == (8,)
    assert cvr_logits.shape == (8,)


def test_output_is_differentiable():
    model = MMoEDeepFM(**_make_config())
    batch = _make_batch(4)
    ctr_logits, cvr_logits = model(batch)
    loss = ctr_logits.sum() + cvr_logits.sum()
    loss.backward()
    for p in model.parameters():
        if p.requires_grad:
            assert p.grad is not None


def test_shared_embeddings_are_same_objects():
    model = MMoEDeepFM(**_make_config())
    assert model.user_id_emb is model.user_id_emb


def test_fm_component_contributes():
    torch.manual_seed(0)
    model = MMoEDeepFM(**_make_config())
    batch = _make_batch(4)

    ctr_logits, cvr_logits = model(batch)

    with torch.no_grad():
        model.fm_first_order_weights.zero_()
    ctr_logits_no_fm, cvr_logits_no_fm = model(batch)

    assert not torch.allclose(ctr_logits, ctr_logits_no_fm)


def test_num_experts():
    model = MMoEDeepFM(**_make_config())
    assert len(model.experts) == 3


def test_two_gates_exist():
    model = MMoEDeepFM(**_make_config())
    assert hasattr(model, "gate_ctr")
    assert hasattr(model, "gate_cvr")


def test_batch_size_one():
    model = MMoEDeepFM(**_make_config())
    batch = _make_batch(1)
    ctr_logits, cvr_logits = model(batch)
    assert ctr_logits.shape == (1,)
    assert cvr_logits.shape == (1,)


# --- CrossLayer tests ---

CONCAT_DIM = 7 * 16 + 3  # 115 for test config (embedding_dim=16)


def test_cross_layer_output_shape():
    layer = CrossLayer(CONCAT_DIM)
    x0 = torch.randn(4, CONCAT_DIM)
    xl = torch.randn(4, CONCAT_DIM)
    out = layer(x0, xl)
    assert out.shape == (4, CONCAT_DIM)


def test_cross_layer_math():
    layer = CrossLayer(CONCAT_DIM)
    with torch.no_grad():
        nn.init.eye_(layer.linear.weight)
        nn.init.zeros_(layer.linear.bias)
    x0 = torch.randn(4, CONCAT_DIM)
    xl = torch.randn(4, CONCAT_DIM)
    out = layer(x0, xl)
    expected = x0 * xl + xl
    assert torch.allclose(out, expected, atol=1e-5)


def test_cross_layer_low_rank_shape():
    layer = CrossLayer(CONCAT_DIM, rank=8)
    x0 = torch.randn(4, CONCAT_DIM)
    xl = torch.randn(4, CONCAT_DIM)
    out = layer(x0, xl)
    assert out.shape == (4, CONCAT_DIM)


def test_cross_layer_low_rank_fewer_params():
    full = CrossLayer(CONCAT_DIM, rank=0)
    low = CrossLayer(CONCAT_DIM, rank=8)
    full_params = sum(p.numel() for p in full.parameters())
    low_params = sum(p.numel() for p in low.parameters())
    assert low_params < full_params


# --- DCNExpert tests ---


def test_dcn_expert_output_shape():
    expert = DCNExpert(CONCAT_DIM, num_cross_layers=2, deep_dims=[64, 32], dropout=0.1)
    x = torch.randn(4, CONCAT_DIM)
    out = expert(x)
    assert out.shape == (4, 32)


def test_dcn_expert_differentiable():
    expert = DCNExpert(CONCAT_DIM, num_cross_layers=2, deep_dims=[64, 32], dropout=0.1)
    x = torch.randn(4, CONCAT_DIM)
    out = expert(x)
    out.sum().backward()
    for p in expert.parameters():
        if p.requires_grad:
            assert p.grad is not None


# --- MMoEDeepFM + DCN integration tests ---


def _make_dcn_config():
    return _make_config() | {"use_dcn": True, "num_cross_layers": 2}


def test_dcn_model_output_shapes():
    model = MMoEDeepFM(**_make_dcn_config())
    batch = _make_batch(8)
    ctr_logits, cvr_logits = model(batch)
    assert ctr_logits.shape == (8,)
    assert cvr_logits.shape == (8,)


def test_dcn_model_differentiable():
    model = MMoEDeepFM(**_make_dcn_config())
    batch = _make_batch(4)
    ctr_logits, cvr_logits = model(batch)
    loss = ctr_logits.sum() + cvr_logits.sum()
    loss.backward()
    for p in model.parameters():
        if p.requires_grad:
            assert p.grad is not None


def test_dcn_low_rank_output_shapes():
    cfg = _make_dcn_config() | {"dcn_rank": 4}
    model = MMoEDeepFM(**cfg)
    batch = _make_batch(8)
    ctr_logits, cvr_logits = model(batch)
    assert ctr_logits.shape == (8,)
    assert cvr_logits.shape == (8,)


def test_dcn_backward_compatible():
    model = MMoEDeepFM(**_make_config())
    assert isinstance(model.experts[0], nn.Sequential)


def test_dcn_expert_type():
    model = MMoEDeepFM(**_make_dcn_config())
    assert isinstance(model.experts[0], DCNExpert)


def test_dcn_batch_size_one():
    model = MMoEDeepFM(**_make_dcn_config())
    batch = _make_batch(1)
    ctr_logits, cvr_logits = model(batch)
    assert ctr_logits.shape == (1,)
    assert cvr_logits.shape == (1,)
