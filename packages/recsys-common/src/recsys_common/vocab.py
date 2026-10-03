from __future__ import annotations

import json
from pathlib import Path


class VocabStore:
    def __init__(self, token_to_index: dict[str, int]) -> None:
        self._vocab = token_to_index

    @classmethod
    def build(cls, tokens: list[str]) -> VocabStore:
        vocab: dict[str, int] = {"[UNK]": 0}
        idx = 1
        for token in tokens:
            if token not in vocab:
                vocab[token] = idx
                idx += 1
        return cls(vocab)

    def lookup(self, token: str) -> int:
        return self._vocab.get(token, 0)

    def size(self) -> int:
        return len(self._vocab)

    def save(self, path: Path) -> None:
        path.write_text(json.dumps(self._vocab, indent=2))

    @classmethod
    def load(cls, path: Path) -> VocabStore:
        data = json.loads(path.read_text())
        return cls(data)
