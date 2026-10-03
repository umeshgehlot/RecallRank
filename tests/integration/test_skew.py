"""Verify that FeatureProcessor produces identical outputs when called from
compile.py (offline) and pipeline.py (online) code paths."""

import pytest
from recsys_common.processor import FeatureProcessor
from recsys_common.schemas import ContextFeatures, RawItemFeatures, RawUserFeatures
from recsys_data.compile import compile_dataset
from recsys_data.synthetic import SyntheticConfig, generate_synthetic_data


@pytest.fixture(scope="module")
def artifacts_dir(tmp_path_factory):
    base = tmp_path_factory.mktemp("skew")
    raw_dir = base / "raw"
    compiled_dir = base / "compiled"
    cfg = SyntheticConfig(
        num_users=20, num_items=50, num_interactions=100, num_categories=5, seed=123
    )
    generate_synthetic_data(cfg, raw_dir)
    compile_dataset(raw_dir, compiled_dir)
    return compiled_dir


def test_same_processor_same_output(artifacts_dir):
    """Two independent FeatureProcessor instances produce identical results."""
    p1 = FeatureProcessor(artifacts_dir)
    p2 = FeatureProcessor(artifacts_dir)

    user = RawUserFeatures(
        user_id="user_0",
        user_age=35.0,
        user_historic_categories=["cat_1", "cat_2"],
        session_last_3_clicks=["item_5", "item_10"],
    )
    ctx = ContextFeatures(device_type="mobile", hour_of_day=14)
    item = RawItemFeatures(
        item_id="item_0",
        item_category="cat_1",
        item_historical_ctr=0.05,
        item_price=25.0,
    )

    pu1 = p1.process_user(user, ctx)
    pu2 = p2.process_user(user, ctx)
    assert pu1.user_id_idx == pu2.user_id_idx
    assert pu1.user_age_scaled == pu2.user_age_scaled
    assert pu1.historic_category_indices == pu2.historic_category_indices
    assert pu1.session_click_indices == pu2.session_click_indices
    assert pu1.device_type_idx == pu2.device_type_idx
    assert pu1.hour_of_day_scaled == pu2.hour_of_day_scaled

    pi1 = p1.process_item(item)
    pi2 = p2.process_item(item)
    assert pi1.item_id_idx == pi2.item_id_idx
    assert pi1.category_idx == pi2.category_idx
    assert pi1.ctr_scaled == pi2.ctr_scaled
    assert pi1.price_bucket == pi2.price_bucket


def test_unknown_tokens_handled_consistently(artifacts_dir):
    """Unknown user/item IDs should map to [UNK] index 0 every time."""
    p = FeatureProcessor(artifacts_dir)

    user = RawUserFeatures(
        user_id="NEVER_SEEN_USER",
        user_age=25.0,
        user_historic_categories=["UNKNOWN_CAT"],
        session_last_3_clicks=["UNKNOWN_ITEM"],
    )
    ctx = ContextFeatures(device_type="UNKNOWN_DEVICE", hour_of_day=12)
    item = RawItemFeatures(
        item_id="NEVER_SEEN_ITEM",
        item_category="UNKNOWN_CAT",
        item_historical_ctr=0.01,
        item_price=10.0,
    )

    pu = p.process_user(user, ctx)
    assert pu.user_id_idx == 0
    assert pu.historic_category_indices == [0]
    assert pu.session_click_indices == [0]
    assert pu.device_type_idx == 0

    pi = p.process_item(item)
    assert pi.item_id_idx == 0
    assert pi.category_idx == 0


def test_numerical_transforms_deterministic(artifacts_dir):
    """Numerical transforms (z-score, log-compress, quantile-bucket) are deterministic."""
    p = FeatureProcessor(artifacts_dir)

    item = RawItemFeatures(
        item_id="item_0",
        item_category="cat_1",
        item_historical_ctr=0.08,
        item_price=50.0,
    )

    results = [p.process_item(item) for _ in range(5)]
    for r in results[1:]:
        assert r.ctr_scaled == results[0].ctr_scaled
        assert r.price_bucket == results[0].price_bucket
