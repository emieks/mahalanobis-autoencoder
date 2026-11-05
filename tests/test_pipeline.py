import json
from pathlib import Path

import pandas as pd

from mvad.cli import main
from mvad.scoring import MahalanobisDetector
from tests.conftest import FEATURES, make_frame

CONFIG = """
[paths]
raw_data = "data/data.csv"
data_dir = "data"
model_dir = "models"
results_dir = "results"

[data]
time_column = "Time"
sensors = ["Vibration Radial", "Vibration Axial", "Humidity", "Temperature", "Shaft speed"]
train_fraction = 0.70
val_fraction = 0.15

[model]
hidden_dims = [16, 8, 4]
dropout = 0.2

[training]
epochs = 2
batch_size = 256
learning_rate = 1e-3
early_stopping_patience = 5
lr_patience = 1
lr_factor = 0.1

[injection]
seed = 42
anomaly_fraction = 0.05
min_length = 30
max_length = 100
noise = 0.05
magnitudes = [1, 5]
coupled = ["Vibration Radial", "Vibration Axial", "Shaft speed"]

[evaluation]
thresholds = [0.5, 2.5]
baseline_sigma = 3.0
chi2_alpha = 0.05
"""


def test_end_to_end(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    Path("data").mkdir()
    make_frame(20_000).to_csv("data/data.csv", index=False)
    Path("config.toml").write_text(CONFIG)

    for command in ("split", "train", "inject", "evaluate"):
        main(["--config", "config.toml", "--cpu", command])

    assert (tmp_path / "data/test_z1.csv").exists()
    assert (tmp_path / "data/test_z5.csv").exists()
    results = json.loads((tmp_path / "results/evaluation.json").read_text())
    assert set(results["magnitudes"]) == {"1", "5"}
    for method in ("baseline", "reconstruction", "mahalanobis", "ensemble"):
        entry = results["magnitudes"]["5"][method]
        assert set(entry["auroc"]) == {"point", "collective"}
        assert set(entry["point_adjusted"]) == {"0.5", "2.5"}

    detector = MahalanobisDetector.load(tmp_path / "models", FEATURES)
    test = pd.read_csv(tmp_path / "data/test_z5.csv")
    flags = detector.predict(test)
    assert flags.shape == (len(test),)
    assert flags.dtype == bool
