from __future__ import annotations

import json
from pathlib import Path
from typing import Protocol

import faiss
import numpy as np


class VectorIndex(Protocol):
    def build(self, vectors: np.ndarray, ids: list[str]) -> None: ...
    def query(self, vector: np.ndarray, top_k: int) -> list[tuple[str, float]]: ...
    def save(self, path: Path) -> None: ...


class FaissIndex:
    def __init__(self, dim: int) -> None:
        self._dim = dim
        self._index = faiss.IndexFlatIP(dim)
        self._ids: list[str] = []

    def build(self, vectors: np.ndarray, ids: list[str]) -> None:
        faiss.normalize_L2(vectors)
        self._index.add(vectors)
        self._ids = list(ids)

    def query(self, vector: np.ndarray, top_k: int) -> list[tuple[str, float]]:
        vec = vector.reshape(1, -1).copy()
        faiss.normalize_L2(vec)
        scores, indices = self._index.search(vec, top_k)
        return [
            (self._ids[idx], float(score)) for score, idx in zip(scores[0], indices[0]) if idx >= 0
        ]

    def save(self, path: Path) -> None:
        path.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self._index, str(path / "index.faiss"))
        (path / "ids.json").write_text(json.dumps(self._ids))

    @classmethod
    def load(cls, path: Path) -> FaissIndex:
        index = faiss.read_index(str(path / "index.faiss"))
        ids = json.loads((path / "ids.json").read_text())
        obj = cls(index.d)
        obj._index = index
        obj._ids = ids
        return obj
