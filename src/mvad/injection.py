from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations

import numpy as np
import pandas as pd

LABEL = "anomaly_type"
NORMAL, POINT, COLLECTIVE = 0, 1, 2

LOW_LEVEL_OFFSET = 0.1


@dataclass(frozen=True)
class PointAnomaly:
    start: int
    end: int
    column: str
    sign: int


@dataclass(frozen=True)
class CollectiveAnomaly:
    start: int
    end: int
    low: tuple[str, ...]


Anomaly = PointAnomaly | CollectiveAnomaly


def collective_patterns(coupled: list[str]) -> list[tuple[str, ...]]:
    """Every way of driving some, but not all, of the coupled channels low."""
    return [subset for size in range(1, len(coupled)) for subset in combinations(coupled, size)]


def plan_anomalies(
    reference: pd.DataFrame,
    sensors: list[str],
    coupled: list[str],
    anomaly_fraction: float,
    min_length: int,
    max_length: int,
    max_magnitude: float,
    rng: np.random.Generator,
) -> list[Anomaly]:
    """Choose non-overlapping intervals until `anomaly_fraction` of the rows are covered.

    Point and collective anomalies alternate. A plan is applied at several magnitudes, so
    point anomalies are validated at the largest one: a displacement that would push a
    non-negative channel below zero is redrawn.
    """
    occupied = np.zeros(len(reference), dtype=bool)
    target = int(anomaly_fraction * len(reference))
    patterns = collective_patterns(coupled)
    values = {c: reference[c].to_numpy(dtype=float) for c in sensors}
    stds = {c: float(values[c].std()) for c in sensors}
    non_negative = {c: bool((values[c] >= 0).all()) for c in sensors}

    plan: list[Anomaly] = []
    covered = 0
    while covered < target:
        start, end = _free_interval(occupied, min_length, max_length, rng)
        if len(plan) % 2 == 0:
            plan.append(_point_anomaly(start, end, values, stds, non_negative, max_magnitude, rng))
        else:
            plan.append(CollectiveAnomaly(start, end, patterns[rng.integers(len(patterns))]))
        covered += end - start
    return plan


def inject(
    reference: pd.DataFrame,
    plan: list[Anomaly],
    magnitude: float,
    coupled: list[str],
    noise: float,
    rng: np.random.Generator,
) -> pd.DataFrame:
    """Apply a plan at a point-anomaly magnitude (in global standard deviations).

    Point anomalies displace one channel by a constant offset. Collective anomalies hold the
    chosen coupled channels just above their minimum and the rest at their mean, so each
    channel stays in its normal range while the relationship between them breaks. Gaussian
    noise proportional to the injected level is added to everything. The returned frame has
    an `anomaly_type` label column.
    """
    df = reference.copy()
    touched = sorted({a.column for a in plan if isinstance(a, PointAnomaly)} | set(coupled))
    df[touched] = df[touched].astype(float)
    stds = reference[touched].std()
    means = reference[coupled].mean()
    lows = reference[coupled].min() + LOW_LEVEL_OFFSET
    labels = np.zeros(len(df), dtype=np.int8)

    for anomaly in plan:
        rows = slice(anomaly.start, anomaly.end)
        length = anomaly.end - anomaly.start
        if isinstance(anomaly, PointAnomaly):
            col = df.columns.get_loc(anomaly.column)
            offset = anomaly.sign * magnitude * stds[anomaly.column]
            df.iloc[rows, col] = _with_noise(df.iloc[rows, col].to_numpy() + offset, noise, rng)
            labels[rows] = POINT
        else:
            for column in coupled:
                level = lows[column] if column in anomaly.low else means[column]
                col = df.columns.get_loc(column)
                df.iloc[rows, col] = _with_noise(np.full(length, level), noise, rng)
            labels[rows] = COLLECTIVE

    df[LABEL] = labels
    return df


def collective_cosine_similarity(
    reference: pd.DataFrame, injected: pd.DataFrame, plan: list[Anomaly]
) -> float:
    """Mean cosine similarity between original and injected values of the channels driven low."""
    similarities = []
    for anomaly in plan:
        if not isinstance(anomaly, CollectiveAnomaly):
            continue
        columns = list(anomaly.low)
        u = reference[columns].to_numpy(dtype=float)[anomaly.start : anomaly.end].ravel()
        v = injected[columns].to_numpy(dtype=float)[anomaly.start : anomaly.end].ravel()
        norm = np.linalg.norm(u) * np.linalg.norm(v)
        if norm > 0:
            similarities.append(u @ v / norm)
    return float(np.mean(similarities)) if similarities else float("nan")


def _free_interval(
    occupied: np.ndarray, min_length: int, max_length: int, rng: np.random.Generator
) -> tuple[int, int]:
    for _ in range(100):
        length = int(rng.integers(min_length, max_length + 1))
        start = int(rng.integers(0, len(occupied) - length))
        if not occupied[start : start + length].any():
            occupied[start : start + length] = True
            return start, start + length
    raise RuntimeError("could not place a non-overlapping anomaly interval")


def _point_anomaly(
    start: int,
    end: int,
    values: dict[str, np.ndarray],
    stds: dict[str, float],
    non_negative: dict[str, bool],
    max_magnitude: float,
    rng: np.random.Generator,
) -> PointAnomaly:
    columns = [c for c in values if stds[c] > 0]
    for _ in range(100):
        column = columns[rng.integers(len(columns))]
        sign = int(rng.choice((-1, 1)))
        displaced = values[column][start:end] + sign * max_magnitude * stds[column]
        if non_negative[column] and (displaced < 0).any():
            continue
        return PointAnomaly(start, end, column, sign)
    raise RuntimeError(f"could not place a valid point anomaly in rows {start}-{end}")


def _with_noise(values: np.ndarray, noise: float, rng: np.random.Generator) -> np.ndarray:
    std = noise * np.abs(values).mean()
    return values + rng.normal(0.0, std, size=len(values)) if std > 0 else values
