# Mahalanobis Autoencoder

Unsupervised anomaly detection in multivariate time series, scoring samples by the
Mahalanobis distance of their latent representation instead of the reconstruction error.

This is the code for my Bachelor's thesis *Anomaly Detection in Multivariate Time Series
Using Reconstruction Autoencoder with Mahalanobis Distance* (B.Sc. in Computer Science with
Specialization in Intelligent Systems, Mälardalen University, June 2026), carried out with
Leine & Linde AB. The report is available on [DiVA](https://mdu.diva-portal.org/smash/record.jsf?pid=diva2%3A2076078).

## Overview

Reconstruction autoencoders are a common choice for unsupervised anomaly detection in
sensor data, but they have a weakness that follows from the manifold hypothesis: samples
that lie close to the learned data manifold are reconstructed well even when they are
anomalous, so subtle anomalies pass undetected.

This project replaces the reconstruction error with the Mahalanobis distance computed in
the autoencoder's latent space. The model is trained on three months of operating data from
a rotary encoder in an industrial installation and evaluated against synthetic point and
collective anomalies injected into a held-out test period. Four detectors are compared:

- reconstruction error of the autoencoder
- Mahalanobis distance of the latent representation from the training distribution
- an ensemble averaging the two normalized scores
- static per-channel limits at three standard deviations from the training mean

The two autoencoder scores trade off against each other. Reconstruction error is the
stronger detector of isolated point anomalies, while the Mahalanobis distance is stronger on
collective anomalies that break the correlation between channels without pushing any single
channel out of its normal range. The static baseline catches large point anomalies but
cannot represent collective ones at all. The ensemble gives the most balanced performance
across both anomaly types.

| AUROC | Point $z=1$ | $z=2$ | $z=3$ | $z=4$ | $z=5$ | Collective |
|---|---|---|---|---|---|---|
| Static thresholds | 0.599 | 0.697 | 0.837 | 0.951 | 0.979 | 0.480 |
| Reconstruction AE | 0.766 | 0.924 | 0.979 | 0.985 | 0.986 | 0.541 |
| Mahalanobis AE | 0.612 | 0.743 | 0.848 | 0.943 | 0.973 | 0.579 |
| Ensemble AE | 0.689 | 0.894 | 0.965 | 0.983 | 0.986 | 0.581 |

Point anomalies displace one channel by $z$ global standard deviations. Collective anomalies
hold one or two of the coupled channels (shaft speed, radial and axial vibration) near their
minimum while the others stay at their mean.

## Method

The autoencoder is a symmetric fully connected network with a 32-16-8-4 encoder, Leaky ReLU
activations, and batch normalization and dropout after the first two encoder layers. It is
trained with MSE loss and Adam, with early stopping on validation loss. After training, the
training set is passed through the encoder and the mean and covariance of the latent
vectors form a Mahalanobis profile. The Mahalanobis score of a sample is the distance of its
latent vector from that profile.

For evaluation, each detector's scores are standardized by their mean and standard
deviation, and a threshold $k$ flags samples more than $k$ standard deviations above normal.
AUROC is reported per anomaly type and point-adjusted F1 at several $k$. For deployment, the
Mahalanobis detector uses a threshold from the chi-squared distribution with one degree of
freedom per latent dimension (`MahalanobisDetector` in `mvad/scoring.py`).

## Data

The data used in the thesis is confidential and is not included. The pipeline expects a CSV
of normal operation at `data/data.csv`, one row per timestep. The column names, including
the `coupled` channels that the collective anomalies are built from, are set in
`configs/default.toml`.

## Usage

```bash
pip install -e .
mvad split
mvad train
mvad inject
mvad evaluate
```

`split` writes train, validation and test sets to `data/`. `train` saves the autoencoder,
scaler, Mahalanobis profile and static baseline to `models/`. `inject` writes one injected
test set per magnitude to `data/test_z{z}.csv`, and `evaluate` writes
`results/evaluation.json`. All settings are in `configs/default.toml`; pass `--config` to
use another file.

To use the trained Mahalanobis detector on new data:

```python
import pandas as pd
from mvad.config import load_config
from mvad.scoring import MahalanobisDetector

cfg = load_config()
detector = MahalanobisDetector.load(cfg.model_dir, cfg.data.features, alpha=0.05)
flags = detector.predict(pd.read_csv("new_readings.csv"))
```

## Development

```bash
pip install -e .[dev]
ruff check . && ruff format --check .
pytest
```

## License

[MIT](LICENSE)
