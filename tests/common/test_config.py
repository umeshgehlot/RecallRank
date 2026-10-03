from recsys_common.config import load_config


def test_load_config(tmp_path):
    cfg_path = tmp_path / "test.yaml"
    cfg_path.write_text("epochs: 5\nlr: 0.01\n")
    result = load_config(cfg_path)
    assert result == {"epochs": 5, "lr": 0.01}


def test_load_config_with_overrides(tmp_path):
    cfg_path = tmp_path / "test.yaml"
    cfg_path.write_text("epochs: 5\nlr: 0.01\n")
    result = load_config(cfg_path, {"epochs": 10})
    assert result == {"epochs": 10, "lr": 0.01}


def test_load_config_none_overrides_ignored(tmp_path):
    cfg_path = tmp_path / "test.yaml"
    cfg_path.write_text("epochs: 5\n")
    result = load_config(cfg_path, {"epochs": None, "lr": 0.001})
    assert result == {"epochs": 5, "lr": 0.001}
