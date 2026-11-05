from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch

from mvad.config import Config
from mvad.data import Scaler, features
from mvad.injection import COLLECTIVE, LABEL, POINT
from mvad.metrics import auroc_by_type, point_adjusted_metrics, standardize
from mvad.model import AutoEncoder
from mvad.scoring import (
    MahalanobisProfile,
    StaticBaseline,
    ensemble,
    mahalanobis_distance,
    reconstruction_error,
)

METHODS = ("baseline", "reconstruction", "mahalanobis", "ensemble")


@dataclass
class Scorers:
    model: AutoEncoder
    scaler: Scaler
    profile: MahalanobisProfile
    baseline: StaticBaseline
    columns: list[str]
    device: torch.device | str

    @classmethod
    def load(cls, cfg: Config, device: torch.device | str) -> Scorers:
        return cls(
            model=AutoEncoder.load(cfg.model_dir / "autoencoder.pt", device),
            scaler=Scaler.load(cfg.model_dir / "scaler.json"),
            profile=MahalanobisProfile.load(cfg.model_dir / "mahalanobis_profile.npz"),
            baseline=StaticBaseline.load(cfg.model_dir / "static_baseline.json"),
            columns=cfg.data.features,
            device=device,
        )

    def score(self, df: pd.DataFrame) -> dict[str, np.ndarray]:
        x = self.scaler.transform(features(df, self.columns))
        recon = reconstruction_error(self.model, x, self.device)
        mahal = mahalanobis_distance(self.model, self.profile, x, self.device)
        return {
            "baseline": self.baseline.score(df),
            "reconstruction": recon,
            "mahalanobis": mahal,
            "ensemble": ensemble(mahal, recon),
        }


def evaluate(cfg: Config, device: torch.device | str, log: Callable[[str], None] = print) -> dict:
    """Score every injected test set and report AUROC per anomaly type and point-adjusted
    metrics on point anomalies at standardized score thresholds.

    Scores are standardized by their mean and standard deviation on the validation set, so a
    threshold k flags samples scoring more than k standard deviations above normal.
    """
    scorers = Scorers.load(cfg, device)
    val_scores = scorers.score(pd.read_csv(cfg.data_dir / "val.csv"))
    stats = {m: (float(s.mean()), float(s.std())) for m, s in val_scores.items()}

    results = {"thresholds": cfg.evaluation.thresholds, "magnitudes": {}}
    for magnitude in cfg.injection.magnitudes:
        test = pd.read_csv(cfg.data_dir / f"test_z{magnitude:g}.csv")
        anomaly_type = test[LABEL].to_numpy()
        point_rows = anomaly_type != COLLECTIVE
        point_truth = anomaly_type[point_rows] == POINT

        per_method = {}
        for method, scores in scorers.score(test).items():
            standardized = standardize(scores[point_rows], *stats[method])
            per_method[method] = {
                "auroc": auroc_by_type(scores, anomaly_type),
                "point_adjusted": {
                    f"{k:g}": point_adjusted_metrics(standardized > k, point_truth)
                    for k in cfg.evaluation.thresholds
                },
            }
        results["magnitudes"][f"{magnitude:g}"] = per_method
        log(f"z={magnitude:g}  " + _summary(per_method))

    cfg.results_dir.mkdir(parents=True, exist_ok=True)
    out = cfg.results_dir / "evaluation.json"
    out.write_text(json.dumps(results, indent=2))
    log(f"Saved {out}")
    return results


def _summary(per_method: dict) -> str:
    return "  ".join(
        f"{m}: point {r['auroc'].get('point', float('nan')):.3f} "
        f"collective {r['auroc'].get('collective', float('nan')):.3f}"
        for m, r in per_method.items()
    )
