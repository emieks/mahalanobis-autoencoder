import numpy as np
import pytest

from mvad.injection import COLLECTIVE, NORMAL, POINT
from mvad.metrics import (
    auroc_by_type,
    classification_metrics,
    episodes,
    point_adjust,
    point_adjusted_metrics,
    standardize,
)


def test_episodes_are_half_open_ranges():
    truth = np.array([0, 1, 1, 0, 0, 1, 0, 1])
    assert episodes(truth) == [(1, 3), (5, 6), (7, 8)]
    assert episodes(np.zeros(4)) == []


def test_point_adjust_fills_detected_episodes_only():
    truth = np.array([0, 1, 1, 1, 0, 1, 1, 0])
    predictions = np.array([0, 0, 1, 0, 1, 0, 0, 0])
    adjusted = point_adjust(predictions, truth)
    assert adjusted.tolist() == [False, True, True, True, True, False, False, False]


def test_point_adjusted_metrics():
    truth = np.array([0, 1, 1, 1, 0, 1, 1, 0])
    predictions = np.array([0, 0, 1, 0, 1, 0, 0, 0])
    plain = classification_metrics(predictions, truth)
    adjusted = point_adjusted_metrics(predictions, truth)
    assert plain["recall"] == pytest.approx(1 / 5)
    assert adjusted["recall"] == pytest.approx(3 / 5)
    assert adjusted["precision"] == pytest.approx(3 / 4)


def test_auroc_by_type_uses_normal_rows_as_negatives():
    anomaly_type = np.array([NORMAL, NORMAL, POINT, COLLECTIVE, NORMAL, POINT])
    scores = np.array([0.1, 0.2, 0.9, 0.05, 0.3, 0.8])
    result = auroc_by_type(scores, anomaly_type)
    assert result["point"] == pytest.approx(1.0)
    assert result["collective"] == pytest.approx(0.0)


def test_auroc_by_type_skips_missing_types():
    anomaly_type = np.array([NORMAL, NORMAL, POINT])
    assert set(auroc_by_type(np.array([0.1, 0.2, 0.9]), anomaly_type)) == {"point"}


def test_standardize():
    assert np.allclose(standardize(np.array([1.0, 3.0]), 1.0, 2.0), [0.0, 1.0])
    assert np.allclose(standardize(np.array([1.0, 3.0]), 1.0, 0.0), [0.0, 2.0])
