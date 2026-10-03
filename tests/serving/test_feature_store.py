from recsys_common.schemas import RawItemFeatures, RawUserFeatures
from recsys_serving.feature_store import SqliteFeatureStore


def test_store_and_retrieve_user(tmp_path):
    store = SqliteFeatureStore(tmp_path / "features.db")
    user = RawUserFeatures(
        user_id="u1",
        user_age=25.0,
        user_historic_categories=["electronics", "books"],
        session_last_3_clicks=["i1", "i2"],
    )
    store.put_user_features("u1", user)
    result = store.get_user_features("u1")
    assert result is not None
    assert result.user_id == "u1"
    assert result.user_age == 25.0
    assert result.user_historic_categories == ["electronics", "books"]


def test_missing_user_returns_none(tmp_path):
    store = SqliteFeatureStore(tmp_path / "features.db")
    assert store.get_user_features("nonexistent") is None


def test_store_and_retrieve_items(tmp_path):
    store = SqliteFeatureStore(tmp_path / "features.db")
    items = [
        RawItemFeatures(
            item_id="i1", item_category="electronics", item_historical_ctr=0.05, item_price=29.99
        ),
        RawItemFeatures(
            item_id="i2", item_category="books", item_historical_ctr=0.03, item_price=14.99
        ),
    ]
    for item in items:
        store.put_item_features(item.item_id, item)

    result = store.get_item_features(["i1", "i2", "i3"])
    assert "i1" in result
    assert "i2" in result
    assert "i3" not in result
    assert result["i1"].item_price == 29.99


def test_store_and_retrieve_session(tmp_path):
    store = SqliteFeatureStore(tmp_path / "features.db")
    store.put_user_session("u1", ["i10", "i20", "i30"])
    result = store.get_user_session("u1")
    assert result == ["i10", "i20", "i30"]


def test_missing_session_returns_empty(tmp_path):
    store = SqliteFeatureStore(tmp_path / "features.db")
    assert store.get_user_session("nonexistent") == []
