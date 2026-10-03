import bisect
import math


def log_compress(x: float) -> float:
    return math.log(1.0 + max(0.0, x))


def z_score(x: float, mu: float, sigma: float) -> float:
    if sigma == 0.0:
        return 0.0
    return (x - mu) / sigma


def quantile_bucket(x: float, boundaries: list[float]) -> int:
    return bisect.bisect_right(boundaries, x)


def vocab_lookup(token: str, vocab: dict[str, int]) -> int:
    return vocab.get(token, 0)


def pad_and_mask(indices: list[int], max_len: int) -> tuple[list[int], list[int]]:
    n = min(len(indices), max_len)
    padded = (indices + [0] * max_len)[:max_len]
    mask = [1] * n + [0] * (max_len - n)
    return padded, mask
