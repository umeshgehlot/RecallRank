import numpy as np


def compute_log_loss(labels: np.ndarray, predictions: np.ndarray) -> float:
    eps = 1e-7
    preds = np.clip(predictions, eps, 1.0 - eps)
    losses = -(labels * np.log(preds) + (1.0 - labels) * np.log(1.0 - preds))
    return float(losses.mean())


def compute_gauc(user_ids: np.ndarray, labels: np.ndarray, predictions: np.ndarray) -> float:
    unique_users = np.unique(user_ids)
    weighted_auc_sum = 0.0
    weight_sum = 0.0

    for uid in unique_users:
        mask = user_ids == uid
        user_labels = labels[mask]
        user_preds = predictions[mask]

        if len(np.unique(user_labels)) < 2:
            continue

        auc = _auc(user_labels, user_preds)
        weight = float(mask.sum())
        weighted_auc_sum += auc * weight
        weight_sum += weight

    if weight_sum == 0:
        return 0.0
    return weighted_auc_sum / weight_sum


def compute_auc(labels: np.ndarray, predictions: np.ndarray) -> float:
    if len(np.unique(labels)) < 2:
        return 0.0
    return _auc(labels, predictions)


def _auc(labels: np.ndarray, predictions: np.ndarray) -> float:
    pos = predictions[labels == 1]
    neg = predictions[labels == 0]
    if len(pos) == 0 or len(neg) == 0:
        return 0.0
    concordant = sum((p > n) + 0.5 * (p == n) for p in pos for n in neg)
    return float(concordant / (len(pos) * len(neg)))
