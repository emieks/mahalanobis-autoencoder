import torch
from torch import nn

from mvad.model import AutoEncoder


def test_shapes():
    model = AutoEncoder(6, [32, 16, 8, 4]).eval()
    x = torch.randn(10, 6)
    assert model.encode(x).shape == (10, 4)
    assert model(x).shape == (10, 6)
    assert model.latent_dim == 4


def test_regularization_only_on_first_two_encoder_layers():
    model = AutoEncoder(6, [32, 16, 8, 4])
    assert sum(isinstance(m, nn.BatchNorm1d) for m in model.encoder) == 2
    assert sum(isinstance(m, nn.Dropout) for m in model.encoder) == 2
    assert not any(isinstance(m, nn.BatchNorm1d | nn.Dropout) for m in model.decoder)
    assert isinstance(model.encoder[-1], nn.Linear)
    assert isinstance(model.decoder[-1], nn.Linear)


def test_save_and_load_round_trip(tmp_path):
    model = AutoEncoder(6, [8, 4], dropout=0.1).eval()
    model.save(tmp_path / "ae.pt")
    loaded = AutoEncoder.load(tmp_path / "ae.pt")
    x = torch.randn(5, 6)
    assert loaded.hidden_dims == [8, 4]
    assert torch.allclose(model(x), loaded(x))
