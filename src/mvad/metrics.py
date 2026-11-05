from __future__ import annotations

import numpy as np
from sklearn.metrics import roc_auc_score

from mvad.injection import COLLECTIVE, NORMAL, POINT


def auroc(scores: np.ndarray, truth: np.ndarray) -> float:
    return float(roc_auc_score(truth, scores))


def auroc_by_type(scores: np.ndarray, anomaly_type: np.ndarray) -> dict[str, float]:
    """AUROC for each anomaly type, computed against the normal rows only."""
    result = {}
    for name, kind in (("point", POINT), ("collective", COLLECTIVE)):
        mask = (anomaly_type == NORMAL) | (anomaly_type == kind)
        truth = anomaly_type[mask] == kind
        if truth.any() and not truth.all():
            result[name] = auroc(scores[mask], truth)
    return result


def standardize(scores: np.ndarray, mean: float, std: float) -> np.ndarray:
    return (scores - mean) / (std if std > 0 else 1.0)


def episodes(truth: np.ndarray) -> list[tuple[int, int]]:
    """Half-open (start, end) index ranges of consecutive positive labels."""
    padded = np.concatenate(([0], truth.astype(np.int8), [0]))
    edges = np.flatnonzero(np.diff(padded))
    return list(zip(edges[::2].tolist(), edges[1::2].tolist(), strict=True))


def point_adjust(predictions: np.ndarray, truth: np.ndarray) -> np.ndarray:
    """Mark a whole anomaly episode as detected if any sample in it was flagged."""
    adjusted = predictions.astype(bool).copy()
    for start, end in episodes(truth):
        if adjusted[start:end].any():
            adjusted[start:end] = True
    return adjusted


def classification_metrics(predictions: np.ndarray, truth: np.ndarray) -> dict[str, float]:
    predictions, truth = predictions.astype(bool), truth.astype(bool)
    tp = int((predictions & truth).sum())
    fp = int((predictions & ~truth).sum())
    fn = int((~predictions & truth).sum())
    tn = int((~predictions & ~truth).sum())
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "accuracy": (tp + tn) / len(truth),
    }


def point_adjusted_metrics(predictions: np.ndarray, truth: np.ndarray) -> dict[str, float]:
    return classification_metrics(point_adjust(predictions, truth), truth)
