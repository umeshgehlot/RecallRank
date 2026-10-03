from recsys_common.config import ModelConfig
from recsys_common.schemas import ContextFeatures, RawItemFeatures, RawUserFeatures


def test_model_config_defaults():
    cfg = ModelConfig()
    assert cfg.embedding_dim == 64
    assert cfg.user_mlp_dims == [256, 128, 64]
    assert cfg.item_mlp_dims == [256, 128, 64]
    assert cfg.deep_mlp_dims == [512, 256, 128]
    assert cfg.deep_dropout == 0.2
    assert cfg.price_buckets == 10
    assert cfg.num_categories == 20


def test_raw_user_features():
    user = RawUserFeatures(
        user_id="u123",
        user_age=25.0,
        user_historic_categories=["electronics", "books"],
        session_last_3_clicks=["item_1", "item_2"],
    )
    assert user.user_id == "u123"
    assert user.user_age == 25.0
    assert len(user.user_historic_categories) == 2
    assert len(user.session_last_3_clicks) == 2


def test_raw_user_features_defaults():
    user = RawUserFeatures(user_id="u1", user_age=30.0)
    assert user.user_historic_categories == []
    assert user.session_last_3_clicks == []


def test_raw_item_features():
    item = RawItemFeatures(
        item_id="i456", item_category="electronics", item_historical_ctr=0.05, item_price=29.99
    )
    assert item.item_id == "i456"
    assert item.item_category == "electronics"


def test_context_features():
    ctx = ContextFeatures(device_type="mobile", hour_of_day=14)
    assert ctx.device_type == "mobile"
    assert ctx.hour_of_day == 14
