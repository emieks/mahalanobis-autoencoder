from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.stats import chi2

from mvad.data import Scaler, features
from mvad.model import AutoEncoder


@torch.no_grad()
def _batched(
    fn: Callable[[torch.Tensor], torch.Tensor],
    x: np.ndarray,
    device: torch.device | str,
    batch_size: int = 8192,
) -> np.ndarray:
    out = []
    for i in range(0, len(x), batch_size):
        batch = torch.from_numpy(x[i : i + batch_size]).to(device)
        out.append(fn(batch).cpu().numpy())
    return np.concatenate(out)


def latents(model: AutoEncoder, x: np.ndarray, device: torch.device | str) -> np.ndarray:
    model.eval()
    return _batched(model.encode, x, device)


def reconstruction_error(
    model: AutoEncoder, x: np.ndarray, device: torch.device | str
) -> np.ndarray:
    """Per-row mean squared reconstruction error."""
    model.eval()
    return _batched(lambda batch: ((model(batch) - batch) ** 2).mean(dim=1), x, device)


class MahalanobisProfile:
    """Mean and inverse covariance of the latent representations of normal data."""

    def __init__(self, mean: np.ndarray, inv_cov: np.ndarray):
        self.mean = np.asarray(mean, dtype=np.float64)
        self.inv_cov = np.asarray(inv_cov, dtype=np.float64)

    @classmethod
    def fit(cls, z: np.ndarray, ridge: float = 1e-6) -> MahalanobisProfile:
        z = z.astype(np.float64)
        cov = np.cov(z, rowvar=False) + ridge * np.eye(z.shape[1])
        return cls(z.mean(axis=0), np.linalg.inv(cov))

    @property
    def dof(self) -> int:
        return len(self.mean)

    def distance(self, z: np.ndarray) -> np.ndarray:
        diff = z.astype(np.float64) - self.mean
        return np.sqrt(np.einsum("ij,jk,ik->i", diff, self.inv_cov, diff))

    def chi2_threshold(self, alpha: float) -> float:
        """Distance above which a sample is rejected at significance `alpha`.

        Assumes the normal latents are Gaussian, so the squared distance is chi-squared
        distributed with one degree of freedom per latent dimension.
        """
        return float(np.sqrt(chi2.ppf(1 - alpha, df=self.dof)))

    def save(self, path: Path) -> None:
        np.savez(path, mean=self.mean, inv_cov=self.inv_cov)

    @classmethod
    def load(cls, path: Path) -> MahalanobisProfile:
        raw = np.load(path)
        return cls(raw["mean"], raw["inv_cov"])


def mahalanobis_distance(
    model: AutoEncoder, profile: MahalanobisProfile, x: np.ndarray, device: torch.device | str
) -> np.ndarray:
    return profile.distance(latents(model, x, device))


def ensemble(*scores: np.ndarray) -> np.ndarray:
    """Average of min-max normalized scores."""
    normalized = [(s - s.min()) / (s.max() - s.min() + 1e-8) for s in scores]
    return np.mean(normalized, axis=0)


class StaticBaseline:
    """Fixed per-channel limits at `sigma` standard deviations from the training mean.

    The score is the largest exceedance of any channel, zero when all are within limits.
    """

    def __init__(self, columns: list[str], lower: np.ndarray, upper: np.ndarray):
        self.columns = list(columns)
        self.lower = np.asarray(lower, dtype=np.float64)
        self.upper = np.asarray(upper, dtype=np.float64)

    @classmethod
    def fit(cls, train: pd.DataFrame, columns: list[str], sigma: float) -> StaticBaseline:
        x = train[columns].to_numpy(dtype=np.float64)
        mean, std = x.mean(axis=0), x.std(axis=0)
        return cls(columns, mean - sigma * std, mean + sigma * std)

    def score(self, df: pd.DataFrame) -> np.ndarray:
        x = df[self.columns].to_numpy(dtype=np.float64)
        exceedance = np.maximum(self.lower - x, x - self.upper)
        return np.maximum(exceedance, 0).max(axis=1)

    def save(self, path: Path) -> None:
        path.write_text(
            json.dumps(
                {
                    "columns": self.columns,
                    "lower": self.lower.tolist(),
                    "upper": self.upper.tolist(),
                }
            )
        )

    @classmethod
    def load(cls, path: Path) -> StaticBaseline:
        raw = json.loads(path.read_text())
        return cls(raw["columns"], raw["lower"], raw["upper"])


class MahalanobisDetector:
    """The deployable detector: scaler, encoder and latent profile with a chi-squared threshold."""

    def __init__(
        self,
        model: AutoEncoder,
        scaler: Scaler,
        profile: MahalanobisProfile,
        columns: list[str],
        alpha: float = 0.05,
        device: torch.device | str = "cpu",
    ):
        self.model = model
        self.scaler = scaler
        self.profile = profile
        self.columns = columns
        self.threshold = profile.chi2_threshold(alpha)
        self.device = device

    @classmethod
    def load(
        cls, model_dir: Path, columns: list[str], alpha: float = 0.05, device="cpu"
    ) -> MahalanobisDetector:
        return cls(
            AutoEncoder.load(model_dir / "autoencoder.pt", device),
            Scaler.load(model_dir / "scaler.json"),
            MahalanobisProfile.load(model_dir / "mahalanobis_profile.npz"),
            columns,
            alpha,
            device,
        )

    def score(self, df: pd.DataFrame) -> np.ndarray:
        x = self.scaler.transform(features(df, self.columns))
        return mahalanobis_distance(self.model, self.profile, x, self.device)

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        return self.score(df) > self.threshold
