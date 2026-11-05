from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


def temporal_split(
    df: pd.DataFrame, train_fraction: float, val_fraction: float
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    n = len(df)
    train_end = int(n * train_fraction)
    val_end = int(n * (train_fraction + val_fraction))
    return df.iloc[:train_end], df.iloc[train_end:val_end], df.iloc[val_end:]


def features(df: pd.DataFrame, columns: list[str]) -> np.ndarray:
    return df[columns].to_numpy(dtype=np.float32)


class Scaler:
    """Column-wise standardization to zero mean and unit variance."""

    def __init__(self, mean: np.ndarray, scale: np.ndarray):
        self.mean = np.asarray(mean, dtype=np.float32)
        self.scale = np.asarray(scale, dtype=np.float32)

    @classmethod
    def fit(cls, x: np.ndarray) -> Scaler:
        scale = x.std(axis=0)
        scale[scale == 0] = 1.0
        return cls(x.mean(axis=0), scale)

    def transform(self, x: np.ndarray) -> np.ndarray:
        return ((x - self.mean) / self.scale).astype(np.float32)

    def save(self, path: Path) -> None:
        path.write_text(json.dumps({"mean": self.mean.tolist(), "scale": self.scale.tolist()}))

    @classmethod
    def load(cls, path: Path) -> Scaler:
        raw = json.loads(path.read_text())
        return cls(raw["mean"], raw["scale"])
