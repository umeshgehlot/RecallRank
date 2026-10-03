import json
from pathlib import Path

from recsys_common.vocab import VocabStore


def test_build_from_tokens():
    store = VocabStore.build(["apple", "banana", "apple", "cherry"])
    assert store.lookup("apple") >= 1
    assert store.lookup("banana") >= 1
    assert store.lookup("cherry") >= 1
    assert store.lookup("unknown") == 0
    assert store.size() == 4  # 3 unique + 1 for [UNK] at index 0


def test_unk_is_index_zero():
    store = VocabStore.build(["a", "b"])
    assert store.lookup("nonexistent") == 0


def test_indices_are_contiguous():
    store = VocabStore.build(["x", "y", "z"])
    indices = {store.lookup(t) for t in ["x", "y", "z"]}
    assert indices == {1, 2, 3}


def test_save_and_load(tmp_path: Path):
    store = VocabStore.build(["cat", "dog", "fish"])
    filepath = tmp_path / "vocab.json"
    store.save(filepath)
    loaded = VocabStore.load(filepath)
    assert loaded.lookup("cat") == store.lookup("cat")
    assert loaded.lookup("dog") == store.lookup("dog")
    assert loaded.lookup("unknown") == 0
    assert loaded.size() == store.size()


def test_save_format_is_json(tmp_path: Path):
    store = VocabStore.build(["a", "b"])
    filepath = tmp_path / "vocab.json"
    store.save(filepath)
    data = json.loads(filepath.read_text())
    assert "[UNK]" in data
    assert data["[UNK]"] == 0
