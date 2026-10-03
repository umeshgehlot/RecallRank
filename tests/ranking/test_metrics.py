import math

import numpy as np
from recsys_ranking.metrics import compute_auc, compute_gauc, compute_log_loss


def test_log_loss_perfect_predictions():
    labels = np.array([1.0, 0.0, 1.0])
    preds = np.array([0.999, 0.001, 0.999])
    loss = compute_log_loss(labels, preds)
    assert loss < 0.01


def test_log_loss_worst_predictions():
    labels = np.array([1.0, 0.0])
    preds = np.array([0.01, 0.99])
    loss = compute_log_loss(labels, preds)
    assert loss > 2.0


def test_log_loss_clamps_predictions():
    labels = np.array([1.0])
    preds = np.array([0.0])  # would cause log(0) without clamping
    loss = compute_log_loss(labels, preds)
    assert math.isfinite(loss)


def test_gauc_perfect_ranking():
    user_ids = np.array(["u1", "u1", "u1", "u2", "u2", "u2"])
    labels = np.array([1, 0, 0, 1, 0, 0])
    preds = np.array([0.9, 0.3, 0.1, 0.8, 0.4, 0.2])
    gauc = compute_gauc(user_ids, labels, preds)
    assert gauc == 1.0


def test_gauc_random_ranking():
    rng = np.random.default_rng(42)
    n = 1000
    user_ids = np.array([f"u{i % 10}" for i in range(n)])
    labels = rng.integers(0, 2, size=n).astype(float)
    preds = rng.random(n)
    gauc = compute_gauc(user_ids, labels, preds)
    assert 0.3 < gauc < 0.7


def test_gauc_skips_single_class_users():
    user_ids = np.array(["u1", "u1", "u2", "u2"])
    labels = np.array([1, 1, 1, 0])  # u1 has only positives
    preds = np.array([0.9, 0.8, 0.7, 0.3])
    gauc = compute_gauc(user_ids, labels, preds)
    # Only u2 contributes (u1 has single class)
    assert gauc == 1.0


class TestComputeAuc:
    def test_perfect_auc(self):
        labels = np.array([0, 0, 1, 1])
        preds = np.array([0.1, 0.2, 0.8, 0.9])
        assert compute_auc(labels, preds) == 1.0

    def test_random_auc_near_half(self):
        np.random.seed(42)
        labels = np.array([0, 1] * 50)
        preds = np.random.rand(100)
        auc = compute_auc(labels, preds)
        assert 0.3 < auc < 0.7

    def test_single_class_returns_zero(self):
        labels = np.array([1, 1, 1])
        preds = np.array([0.5, 0.6, 0.7])
        assert compute_auc(labels, preds) == 0.0

    def test_worst_auc(self):
        labels = np.array([1, 1, 0, 0])
        preds = np.array([0.1, 0.2, 0.8, 0.9])
        assert compute_auc(labels, preds) == 0.0
