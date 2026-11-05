from __future__ import annotations

import tomllib
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class DataConfig:
    time_column: str
    sensors: list[str]
    train_fraction: float
    val_fraction: float

    @property
    def features(self) -> list[str]:
        return [self.time_column, *self.sensors]


@dataclass(frozen=True)
class ModelConfig:
    hidden_dims: list[int]
    dropout: float


@dataclass(frozen=True)
class TrainingConfig:
    epochs: int
    batch_size: int
    learning_rate: float
    early_stopping_patience: int
    lr_patience: int
    lr_factor: float


@dataclass(frozen=True)
class InjectionConfig:
    seed: int
    anomaly_fraction: float
    min_length: int
    max_length: int
    noise: float
    magnitudes: list[float]
    coupled: list[str]


@dataclass(frozen=True)
class EvaluationConfig:
    thresholds: list[float]
    baseline_sigma: float
    chi2_alpha: float


@dataclass(frozen=True)
class Config:
    raw_data: Path
    data_dir: Path
    model_dir: Path
    results_dir: Path
    data: DataConfig
    model: ModelConfig
    training: TrainingConfig
    injection: InjectionConfig
    evaluation: EvaluationConfig


def load_config(path: str | Path = "configs/default.toml") -> Config:
    raw = tomllib.loads(Path(path).read_text(encoding="utf-8"))
    paths = raw["paths"]
    return Config(
        raw_data=Path(paths["raw_data"]),
        data_dir=Path(paths["data_dir"]),
        model_dir=Path(paths["model_dir"]),
        results_dir=Path(paths["results_dir"]),
        data=DataConfig(**raw["data"]),
        model=ModelConfig(**raw["model"]),
        training=TrainingConfig(**raw["training"]),
        injection=InjectionConfig(**raw["injection"]),
        evaluation=EvaluationConfig(**raw["evaluation"]),
    )
