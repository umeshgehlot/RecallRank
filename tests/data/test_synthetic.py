import pyarrow.parquet as pq
from recsys_data.synthetic import SyntheticConfig, generate_synthetic_data


def test_generate_creates_parquet_files(tmp_path):
    cfg = SyntheticConfig(num_users=100, num_items=200, num_interactions=500, num_categories=5)
    generate_synthetic_data(cfg, tmp_path)
    assert (tmp_path / "users.parquet").exists()
    assert (tmp_path / "items.parquet").exists()
    assert (tmp_path / "interactions.parquet").exists()


def test_user_schema(tmp_path):
    cfg = SyntheticConfig(num_users=50, num_items=100, num_interactions=200, num_categories=5)
    generate_synthetic_data(cfg, tmp_path)
    users = pq.read_table(tmp_path / "users.parquet")
    assert "user_id" in users.column_names
    assert "user_age" in users.column_names
    assert "user_historic_categories" in users.column_names
    assert "session_last_3_clicks" in users.column_names
    assert len(users) == 50


def test_item_schema(tmp_path):
    cfg = SyntheticConfig(num_users=50, num_items=100, num_interactions=200, num_categories=5)
    generate_synthetic_data(cfg, tmp_path)
    items = pq.read_table(tmp_path / "items.parquet")
    assert "item_id" in items.column_names
    assert "item_category" in items.column_names
    assert "item_historical_ctr" in items.column_names
    assert "item_price" in items.column_names
    assert len(items) == 100


def test_interaction_schema(tmp_path):
    cfg = SyntheticConfig(num_users=50, num_items=100, num_interactions=200, num_categories=5)
    generate_synthetic_data(cfg, tmp_path)
    interactions = pq.read_table(tmp_path / "interactions.parquet")
    assert "user_id" in interactions.column_names
    assert "item_id" in interactions.column_names
    assert "clicked" in interactions.column_names
    assert "converted" in interactions.column_names
    assert "device_type" in interactions.column_names
    assert "hour_of_day" in interactions.column_names
    assert len(interactions) == 200


def test_item_prices_are_positive(tmp_path):
    cfg = SyntheticConfig(num_users=50, num_items=100, num_interactions=200, num_categories=5)
    generate_synthetic_data(cfg, tmp_path)
    items = pq.read_table(tmp_path / "items.parquet")
    prices = items.column("item_price").to_pylist()
    assert all(p > 0 for p in prices)


def test_conversion_only_on_clicks(tmp_path):
    cfg = SyntheticConfig(num_users=50, num_items=100, num_interactions=500, num_categories=5)
    generate_synthetic_data(cfg, tmp_path)
    interactions = pq.read_table(tmp_path / "interactions.parquet")
    clicked = interactions.column("clicked").to_pylist()
    converted = interactions.column("converted").to_pylist()
    for c, conv in zip(clicked, converted):
        if c == 0:
            assert conv == 0, "Conversion cannot happen without a click"
