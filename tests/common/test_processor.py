import json
from pathlib import Path

from recsys_common.processor import FeatureProcessor
from recsys_common.schemas import ContextFeatures, RawItemFeatures, RawUserFeatures
from recsys_common.vocab import VocabStore


def _create_artifacts(tmp_path: Path) -> Path:
    artifacts_dir = tmp_path / "artifacts"
    artifacts_dir.mkdir()

    user_vocab = VocabStore.build(["user_1", "user_2"])
    user_vocab.save(artifacts_dir / "user_id_vocab.json")

    item_vocab = VocabStore.build(["item_1", "item_2", "item_3"])
    item_vocab.save(artifacts_dir / "item_id_vocab.json")

    cat_vocab = VocabStore.build(["electronics", "books", "clothing"])
    cat_vocab.save(artifacts_dir / "category_vocab.json")

    device_vocab = VocabStore.build(["mobile", "desktop"])
    device_vocab.save(artifacts_dir / "device_type_vocab.json")

    stats = {
        "user_age": {"mu": 40.0, "sigma": 15.0},
        "hour_of_day": {"mu": 12.0, "sigma": 6.0},
        "item_historical_ctr": {"mu": 0.5, "sigma": 0.3},
        "item_price_boundaries": [5.0, 10.0, 20.0, 30.0, 50.0, 75.0, 100.0, 150.0, 200.0],
    }
    (artifacts_dir / "stats.json").write_text(json.dumps(stats))

    return artifacts_dir


def test_processor_loads_artifacts(tmp_path: Path):
    artifacts_dir = _create_artifacts(tmp_path)
    processor = FeatureProcessor(artifacts_dir)
    assert processor is not None


def test_process_user_features(tmp_path: Path):
    artifacts_dir = _create_artifacts(tmp_path)
    processor = FeatureProcessor(artifacts_dir)

    user = RawUserFeatures(
        user_id="user_1",
        user_age=25.0,
        user_historic_categories=["electronics", "books"],
        session_last_3_clicks=["item_1", "item_2"],
    )
    ctx = ContextFeatures(device_type="mobile", hour_of_day=14)

    result = processor.process_user(user, ctx)
    assert result.user_id_idx >= 1  # known user
    assert isinstance(result.user_age_scaled, float)
    assert len(result.historic_category_indices) == 2
    assert all(idx >= 1 for idx in result.historic_category_indices)
    assert len(result.session_click_indices) == 2
    assert result.device_type_idx >= 1
    assert isinstance(result.hour_of_day_scaled, float)


def test_process_item_features(tmp_path: Path):
    artifacts_dir = _create_artifacts(tmp_path)
    processor = FeatureProcessor(artifacts_dir)

    item = RawItemFeatures(
        item_id="item_1",
        item_category="electronics",
        item_historical_ctr=0.05,
        item_price=29.99,
    )

    result = processor.process_item(item)
    assert result.item_id_idx >= 1
    assert result.category_idx >= 1
    assert isinstance(result.ctr_scaled, float)
    assert 0 <= result.price_bucket <= 9


def test_unknown_user_gets_unk_index(tmp_path: Path):
    artifacts_dir = _create_artifacts(tmp_path)
    processor = FeatureProcessor(artifacts_dir)

    user = RawUserFeatures(user_id="unknown_user", user_age=30.0)
    ctx = ContextFeatures(device_type="tablet", hour_of_day=10)

    result = processor.process_user(user, ctx)
    assert result.user_id_idx == 0
    assert result.device_type_idx == 0


def test_empty_session_and_categories(tmp_path: Path):
    artifacts_dir = _create_artifacts(tmp_path)
    processor = FeatureProcessor(artifacts_dir)

    user = RawUserFeatures(user_id="user_1", user_age=30.0)
    ctx = ContextFeatures(device_type="mobile", hour_of_day=10)

    result = processor.process_user(user, ctx)
    assert result.historic_category_indices == []
    assert result.session_click_indices == []
