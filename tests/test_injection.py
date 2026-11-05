import numpy as np
import pytest

from mvad.injection import (
    COLLECTIVE,
    LABEL,
    NORMAL,
    POINT,
    CollectiveAnomaly,
    PointAnomaly,
    collective_cosine_similarity,
    collective_patterns,
    inject,
    plan_anomalies,
)
from tests.conftest import COUPLED, SENSORS


def make_plan(frame, max_magnitude=5.0, seed=1):
    return plan_anomalies(
        frame, SENSORS, COUPLED, 0.05, 30, 300, max_magnitude, np.random.default_rng(seed)
    )


def test_collective_patterns_cover_every_proper_subset():
    patterns = collective_patterns(["a", "b", "c"])
    assert len(patterns) == 6
    assert all(0 < len(p) < 3 for p in patterns)


def test_plan_alternates_and_does_not_overlap(frame):
    plan = make_plan(frame)
    assert all(isinstance(a, PointAnomaly) for a in plan[::2])
    assert all(isinstance(a, CollectiveAnomaly) for a in plan[1::2])
    assert all(30 <= a.end - a.start <= 300 for a in plan)

    ordered = sorted(plan, key=lambda a: a.start)
    assert all(a.end <= b.start for a, b in zip(ordered, ordered[1:], strict=False))
    covered = sum(a.end - a.start for a in plan)
    assert covered >= 0.05 * len(frame)
    assert covered < 0.05 * len(frame) + 300


def test_point_anomaly_displaces_by_magnitude_in_global_std(frame):
    plan = make_plan(frame)
    injected = inject(frame, plan, 3.0, COUPLED, noise=0.0, rng=np.random.default_rng(0))
    for anomaly in plan:
        if isinstance(anomaly, PointAnomaly):
            rows = slice(anomaly.start, anomaly.end)
            offset = injected[anomaly.column].iloc[rows] - frame[anomaly.column].iloc[rows]
            expected = anomaly.sign * 3.0 * frame[anomaly.column].std()
            assert np.allclose(offset, expected)


def test_point_anomalies_keep_non_negative_channels_non_negative(frame):
    plan = make_plan(frame)
    injected = inject(frame, plan, 5.0, COUPLED, noise=0.0, rng=np.random.default_rng(0))
    assert (injected[SENSORS] >= 0).all().all()


def test_labels_match_plan(frame):
    plan = make_plan(frame)
    injected = inject(frame, plan, 1.0, COUPLED, 0.05, np.random.default_rng(0))
    labels = injected[LABEL].to_numpy()
    expected = np.full(len(frame), NORMAL)
    for anomaly in plan:
        kind = POINT if isinstance(anomaly, PointAnomaly) else COLLECTIVE
        expected[anomaly.start : anomaly.end] = kind
    assert (labels == expected).all()
    assert injected.drop(columns=LABEL).shape == frame.shape


def test_collective_anomalies_stay_within_normal_range(frame):
    plan = make_plan(frame)
    injected = inject(frame, plan, 1.0, COUPLED, noise=0.0, rng=np.random.default_rng(0))
    for anomaly in plan:
        if isinstance(anomaly, CollectiveAnomaly):
            block = injected[COUPLED].iloc[anomaly.start : anomaly.end]
            assert (block.min() >= frame[COUPLED].min()).all()
            assert (block.max() <= frame[COUPLED].max()).all()
            for column in COUPLED:
                level = block[column].iloc[0]
                if column in anomaly.low:
                    assert level == pytest.approx(frame[column].min() + 0.1)
                else:
                    assert level == pytest.approx(frame[column].mean())


def test_cosine_similarity_is_between_zero_and_one(frame):
    plan = make_plan(frame)
    injected = inject(frame, plan, 1.0, COUPLED, 0.05, np.random.default_rng(0))
    similarity = collective_cosine_similarity(frame, injected, plan)
    assert 0.0 < similarity <= 1.0
