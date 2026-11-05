from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import torch
from torch import nn


class AutoEncoder(nn.Module):
    """Symmetric fully connected autoencoder.

    Batch normalization and dropout are applied after the first two encoder layers only.
    The bottleneck and the output layer are linear.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dims: Sequence[int] = (32, 16, 8, 4),
        dropout: float = 0.2,
        negative_slope: float = 0.01,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dims = list(hidden_dims)
        self.dropout = dropout

        encoder: list[nn.Module] = []
        prev = input_dim
        for i, dim in enumerate(self.hidden_dims):
            encoder.append(nn.Linear(prev, dim))
            if i < len(self.hidden_dims) - 1:
                if i < 2:
                    encoder.append(nn.BatchNorm1d(dim))
                encoder.append(nn.LeakyReLU(negative_slope))
                if i < 2:
                    encoder.append(nn.Dropout(dropout))
            prev = dim
        self.encoder = nn.Sequential(*encoder)

        decoder: list[nn.Module] = []
        for dim in reversed(self.hidden_dims[:-1]):
            decoder.extend([nn.Linear(prev, dim), nn.LeakyReLU(negative_slope)])
            prev = dim
        decoder.append(nn.Linear(prev, input_dim))
        self.decoder = nn.Sequential(*decoder)

    @property
    def latent_dim(self) -> int:
        return self.hidden_dims[-1]

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.encoder(x))

    def save(self, path: Path) -> None:
        torch.save(
            {
                "input_dim": self.input_dim,
                "hidden_dims": self.hidden_dims,
                "dropout": self.dropout,
                "state_dict": self.state_dict(),
            },
            path,
        )

    @classmethod
    def load(cls, path: Path, device: torch.device | str = "cpu") -> AutoEncoder:
        checkpoint = torch.load(path, map_location=device)
        model = cls(checkpoint["input_dim"], checkpoint["hidden_dims"], checkpoint["dropout"])
        model.load_state_dict(checkpoint["state_dict"])
        return model.to(device).eval()
