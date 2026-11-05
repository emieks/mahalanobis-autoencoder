from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from mvad.config import TrainingConfig
from mvad.model import AutoEncoder


def train(
    model: AutoEncoder,
    train_x: np.ndarray,
    val_x: np.ndarray,
    cfg: TrainingConfig,
    checkpoint: Path,
    device: torch.device,
    log: Callable[[str], None] = print,
) -> list[dict[str, float]]:
    """Train with MSE loss, saving the best validation checkpoint. Returns the loss history."""
    train_loader = DataLoader(
        TensorDataset(torch.from_numpy(train_x)), batch_size=cfg.batch_size, shuffle=True
    )
    val_loader = DataLoader(TensorDataset(torch.from_numpy(val_x)), batch_size=cfg.batch_size)

    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=cfg.lr_factor, patience=cfg.lr_patience
    )
    loss_fn = nn.MSELoss()

    history = [
        {
            "epoch": 0,
            "train_loss": _evaluate(model, train_loader, loss_fn, device),
            "val_loss": _evaluate(model, val_loader, loss_fn, device),
        }
    ]
    log(_format(history[-1], cfg.epochs))

    best_val = float("inf")
    epochs_without_improvement = 0
    for epoch in range(1, cfg.epochs + 1):
        model.train()
        total = 0.0
        for (batch,) in train_loader:
            batch = batch.to(device)
            loss = loss_fn(model(batch), batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total += loss.item()

        val_loss = _evaluate(model, val_loader, loss_fn, device)
        history.append(
            {"epoch": epoch, "train_loss": total / len(train_loader), "val_loss": val_loss}
        )
        scheduler.step(val_loss)

        if val_loss < best_val:
            best_val = val_loss
            epochs_without_improvement = 0
            model.save(checkpoint)
            log(_format(history[-1], cfg.epochs) + "  *")
        else:
            epochs_without_improvement += 1
            log(_format(history[-1], cfg.epochs))
            if epochs_without_improvement >= cfg.early_stopping_patience:
                log(f"No improvement for {cfg.early_stopping_patience} epochs, stopping.")
                break

    return history


@torch.no_grad()
def _evaluate(
    model: AutoEncoder, loader: DataLoader, loss_fn: nn.Module, device: torch.device
) -> float:
    model.eval()
    total = 0.0
    for (batch,) in loader:
        batch = batch.to(device)
        total += loss_fn(model(batch), batch).item()
    return total / len(loader)


def _format(entry: dict[str, float], epochs: int) -> str:
    return (
        f"epoch {entry['epoch']:>3}/{epochs}  "
        f"train {entry['train_loss']:.6f}  val {entry['val_loss']:.6f}"
    )
