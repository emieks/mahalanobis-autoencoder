from __future__ import annotations

import argparse
import csv

import numpy as np
import pandas as pd
import torch

from mvad.config import Config, load_config
from mvad.data import Scaler, features, temporal_split
from mvad.evaluation import evaluate
from mvad.injection import collective_cosine_similarity, inject, plan_anomalies
from mvad.model import AutoEncoder
from mvad.scoring import MahalanobisProfile, StaticBaseline, latents
from mvad.training import train


def split(cfg: Config) -> None:
    df = pd.read_csv(cfg.raw_data)[cfg.data.features]
    cfg.data_dir.mkdir(parents=True, exist_ok=True)
    parts = temporal_split(df, cfg.data.train_fraction, cfg.data.val_fraction)
    for name, part in zip(("train", "val", "test"), parts, strict=True):
        part.to_csv(cfg.data_dir / f"{name}.csv", index=False)
        print(f"{name}: {len(part)} rows")


def train_model(cfg: Config, device: torch.device) -> None:
    columns = cfg.data.features
    train_df = pd.read_csv(cfg.data_dir / "train.csv")
    val_df = pd.read_csv(cfg.data_dir / "val.csv")
    scaler = Scaler.fit(features(pd.read_csv(cfg.raw_data), columns))
    train_x = scaler.transform(features(train_df, columns))
    val_x = scaler.transform(features(val_df, columns))

    cfg.model_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = cfg.model_dir / "autoencoder.pt"
    model = AutoEncoder(len(columns), cfg.model.hidden_dims, cfg.model.dropout)
    print(f"Training on {device}")
    history = train(model, train_x, val_x, cfg.training, checkpoint, device)

    with open(cfg.model_dir / "training_log.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["epoch", "train_loss", "val_loss"])
        writer.writeheader()
        writer.writerows(history)

    model = AutoEncoder.load(checkpoint, device)
    MahalanobisProfile.fit(latents(model, train_x, device)).save(
        cfg.model_dir / "mahalanobis_profile.npz"
    )
    StaticBaseline.fit(train_df, cfg.data.sensors, cfg.evaluation.baseline_sigma).save(
        cfg.model_dir / "static_baseline.json"
    )
    scaler.save(cfg.model_dir / "scaler.json")
    print(f"Saved model, scaler, Mahalanobis profile and static baseline to {cfg.model_dir}")


def inject_anomalies(cfg: Config) -> None:
    inj = cfg.injection
    reference = pd.read_csv(cfg.data_dir / "test.csv")
    plan = plan_anomalies(
        reference,
        cfg.data.sensors,
        inj.coupled,
        inj.anomaly_fraction,
        inj.min_length,
        inj.max_length,
        max(inj.magnitudes),
        np.random.default_rng(inj.seed),
    )
    print(f"{len(plan)} anomaly intervals covering {sum(a.end - a.start for a in plan)} rows")

    for magnitude in inj.magnitudes:
        rng = np.random.default_rng(inj.seed)
        injected = inject(reference, plan, magnitude, inj.coupled, inj.noise, rng)
        path = cfg.data_dir / f"test_z{magnitude:g}.csv"
        injected.to_csv(path, index=False)
        print(f"z={magnitude:g}: {path}")

    print(
        "Collective anomaly cosine similarity: "
        f"{collective_cosine_similarity(reference, injected, plan):.3f}"
    )


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="mvad")
    parser.add_argument("--config", default="configs/default.toml")
    parser.add_argument("--cpu", action="store_true", help="run on CPU even if CUDA is available")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("split", help="split raw data into train, validation and test sets")
    commands.add_parser("train", help="train the autoencoder and fit the detectors")
    commands.add_parser("inject", help="inject synthetic anomalies into the test set")
    commands.add_parser("evaluate", help="score the injected test sets")
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    match args.command:
        case "split":
            split(cfg)
        case "train":
            train_model(cfg, device)
        case "inject":
            inject_anomalies(cfg)
        case "evaluate":
            evaluate(cfg, device)
