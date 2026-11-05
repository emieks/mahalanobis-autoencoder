import numpy as np
import pandas as pd
import pytest
import torch

from mvad.model import AutoEncoder
from mvad.scoring import (
    MahalanobisProfile,
    StaticBaseline,
    ensemble,
    mahalanobis_distance,
    reconstruction_error,
)


def test_reconstruction_error_is_per_row_mse():
    model = AutoEncoder(3, [2]).eval()
    x = np.random.default_rng(0).normal(size=(20, 3)).astype(np.float32)
    errors = reconstruction_error(model, x, "cpu")
    assert errors.shape == (20,)
    assert (errors >= 0).all()


def test_mahalanobis_profile_scales_by_covariance():
    rng = np.random.default_rng(0)
    z = rng.normal(size=(5000, 2)) * [1.0, 10.0]
    profile = MahalanobisProfile.fit(z)
    assert profile.distance(profile.mean[None, :])[0] == pytest.approx(0.0)
    along_narrow, along_wide = profile.distance(np.array([[1.0, 0.0], [0.0, 10.0]]))
    assert along_narrow == pytest.approx(along_wide, rel=0.1)


def test_mahalanobis_distance_through_model_and_save_load(tmp_path):
    model = AutoEncoder(4, [3, 2]).eval()
    x = np.random.default_rng(0).normal(size=(200, 4)).astype(np.float32)
    profile = MahalanobisProfile.fit(model.encode(torch.from_numpy(x)).detach().numpy())
    profile.save(tmp_path / "profile.npz")
    loaded = MahalanobisProfile.load(tmp_path / "profile.npz")
    original = mahalanobis_distance(model, profile, x, "cpu")
    assert np.allclose(original, mahalanobis_distance(model, loaded, x, "cpu"))


def test_chi2_threshold_matches_table():
    profile = MahalanobisProfile(np.zeros(4), np.eye(4))
    assert profile.chi2_threshold(0.05) ** 2 == pytest.approx(9.488, abs=1e-3)
    assert profile.chi2_threshold(0.01) ** 2 == pytest.approx(13.277, abs=1e-3)


def test_ensemble_averages_normalized_scores():
    a = np.array([0.0, 5.0, 10.0])
    b = np.array([100.0, 100.0, 300.0])
    assert np.allclose(ensemble(a, b), [0.0, 0.25, 1.0], atol=1e-6)


def test_static_baseline_scores_exceedance(tmp_path):
    train = pd.DataFrame({"a": [0.0, 2.0, 4.0, 6.0], "b": [10.0, 10.0, 10.0, 10.0]})
    baseline = StaticBaseline.fit(train, ["a", "b"], sigma=1.0)
    baseline.save(tmp_path / "baseline.json")
    baseline = StaticBaseline.load(tmp_path / "baseline.json")

    inside = pd.DataFrame({"a": [3.0], "b": [10.0]})
    outside = pd.DataFrame({"a": [3.0], "b": [12.5]})
    assert baseline.score(inside)[0] == 0.0
    assert baseline.score(outside)[0] == pytest.approx(2.5)
