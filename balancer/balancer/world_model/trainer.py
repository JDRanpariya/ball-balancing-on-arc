"""
trainer.py  -  WorldModelTrainer
=================================
Handles training, validation, testing, checkpointing, and WandB logging
for the ArcBall LSTM world model.

Separation of concerns
-----------------------
  dataset.py  - data loading / normalisation
  model.py    - architecture + loss
  trainer.py  - training loop (this file)
  train.py    - entry point / config

The 80-model data-scaling sweep lives in paper/ram/scripts/wm_data_sweep.py.
"""

from __future__ import annotations

import os
from typing import Optional

import numpy as np
import torch
from sklearn.metrics import mean_absolute_error, r2_score
from torch.utils.data import DataLoader
from tqdm import tqdm

from .model import LSTMWorldModel, WorldModelConfig


def _wandb():
    """Lazy import of wandb - only needed when actually training."""
    import wandb
    return wandb


class WorldModelTrainer:

    STATE_NAMES = ["cart_pos", "cart_vel", "ball_pos", "ball_vel"]

    def __init__(
        self,
        cfg:            WorldModelConfig,
        train_loader:   DataLoader,
        val_loader:     DataLoader,
        test_loader:    DataLoader,
        epochs:         int   = 80,
        lr:             float = 1e-4,
        checkpoint_dir: str   = "checkpoints",
        save_freq:      int   = 10,
        patience:       int   = 10,       # early stopping patience
    ):
        self.device         = "cuda" if torch.cuda.is_available() else "cpu"
        self.epochs         = epochs
        self.checkpoint_dir = checkpoint_dir
        self.save_freq      = save_freq
        self.patience       = patience
        self.train_loader   = train_loader
        self.val_loader     = val_loader
        self.test_loader    = test_loader

        self.model     = LSTMWorldModel(cfg).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)

        os.makedirs(checkpoint_dir, exist_ok=True)
        if os.environ.get("WANDB_MODE") != "disabled": _wandb().watch(self.model, log="all")

    # -- helpers ---------------------------------------------------------------

    def _batch_to_device(self, batch):
        return [t.to(self.device) for t in batch]

    def _build_input(self, s: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        return torch.cat([s, a], dim=-1)

    def _save_checkpoint(self, epoch: int, train_loss: float, val_loss: float, tag: str):
        path = os.path.join(self.checkpoint_dir, f"{tag}.pth")
        torch.save({
            "epoch":                epoch,
            "model_state_dict":     self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "train_loss":           train_loss,
            "val_loss":             val_loss,
        }, path)
        return path

    # -- train -----------------------------------------------------------------

    def train(self):
        best_val_loss    = float("inf")
        epochs_no_improve = 0

        for epoch in range(1, self.epochs + 1):
            train_loss           = self._train_epoch(epoch)
            val_loss, val_metrics = self._val_epoch(epoch)

            # -- logging -----------------------------------------------
            log = {"epoch": epoch, "train_loss": train_loss, **{f"val_{k}": v for k, v in val_metrics.items()}}
            if os.environ.get("WANDB_MODE") != "disabled": _wandb().log(log)

            print(
                f"Epoch {epoch}/{self.epochs} | "
                f"train_loss: {train_loss:.5f} | "
                f"val_loss: {val_loss:.5f} | "
                f"val_r2: {val_metrics['r2']:.4f} | "
                f"val_mae: {val_metrics['mae']:.4f}"
            )

            # -- checkpointing -----------------------------------------
            if val_loss < best_val_loss:
                best_val_loss     = val_loss
                epochs_no_improve = 0
                path = self._save_checkpoint(epoch, train_loss, val_loss, "best_model")
                print(f"  -> best model saved -> {path}")
            else:
                epochs_no_improve += 1
                print(f"  -> no improvement ({epochs_no_improve}/{self.patience})")

            if epoch % self.save_freq == 0 or epoch == self.epochs:
                path = self._save_checkpoint(epoch, train_loss, val_loss, f"epoch_{epoch:04d}")
                print(f"  -> checkpoint saved  -> {path}")

            # -- early stopping ----------------------------------------
            if epochs_no_improve >= self.patience:
                print(f"  Early stopping triggered at epoch {epoch} (patience={self.patience})")
                break

    def _train_epoch(self, epoch: int) -> float:
        self.model.train()
        total_loss = 0.0

        for s, a, delta in tqdm(self.train_loader, desc=f"Epoch {epoch} [train]", leave=False):
            s, a, delta = self._batch_to_device([s, a, delta])
            x = self._build_input(s, a)

            self.optimizer.zero_grad()
            pred, _ = self.model(x)
            loss, _ = self.model.loss(pred, delta)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()

            total_loss += loss.item() * s.size(0)

        return total_loss / len(self.train_loader.dataset)

    # -- validate --------------------------------------------------------------

    def _val_epoch(self, epoch: int) -> tuple[float, dict]:
        self.model.eval()
        total_loss = 0.0
        all_true, all_pred = [], []

        with torch.no_grad():
            for s, a, delta in tqdm(self.val_loader, desc=f"Epoch {epoch} [val]  ", leave=False):
                s, a, delta = self._batch_to_device([s, a, delta])
                x = self._build_input(s, a)

                pred, _ = self.model(x)
                loss, _ = self.model.loss(pred, delta)
                total_loss += loss.item() * s.size(0)

                all_true.append(delta[:, -1, :].cpu())
                all_pred.append(pred[:,  -1, :].cpu())

        val_loss = total_loss / len(self.val_loader.dataset)

        yt = torch.cat(all_true).numpy().reshape(-1)
        yp = torch.cat(all_pred).numpy().reshape(-1)
        metrics = {
            "loss": val_loss,
            "r2":   float(r2_score(yt, yp)),
            "mae":  float(mean_absolute_error(yt, yp)),
        }
        return val_loss, metrics

    # -- test ------------------------------------------------------------------

    def test(self):
        best_path = os.path.join(self.checkpoint_dir, "best_model.pth")
        if os.path.exists(best_path):
            ckpt = torch.load(best_path, map_location=self.device)
            self.model.load_state_dict(ckpt["model_state_dict"])
            print(f"Loaded best model from {best_path}")
        else:
            print("No saved checkpoint found - testing with current weights.")

        self.model.eval()
        all_true, all_pred = [], []

        with torch.no_grad():
            for s, a, delta in tqdm(self.test_loader, desc="[test]", leave=False):
                s, a, delta = self._batch_to_device([s, a, delta])
                B, T, _ = s.shape
                hc = None

                state      = s[:, 0, :]
                batch_pred = []

                for t in range(T):
                    x_t    = torch.cat([state, a[:, t, :]], dim=-1).unsqueeze(1)
                    pred_t, hc = self.model(x_t, hc)
                    delta_t    = pred_t[:, 0, :]
                    state      = state + delta_t
                    batch_pred.append(delta_t.unsqueeze(1))

                all_pred.append(torch.cat(batch_pred, dim=1).cpu())
                all_true.append(delta.cpu())

        yt = torch.cat(all_true).numpy()
        yp = torch.cat(all_pred).numpy()

        r2  = r2_score(yt.reshape(-1), yp.reshape(-1))
        mae = mean_absolute_error(yt.reshape(-1), yp.reshape(-1))
        print(f"Test | R²: {r2:.4f} | MAE: {mae:.4f}")
        if os.environ.get("WANDB_MODE") != "disabled": _wandb().log({"test_r2": r2, "test_mae": mae})

        self._plot_timeseries(yt, yp)

    # -- plotting --------------------------------------------------------------

    def _plot_timeseries(self, yt: np.ndarray, yp: np.ndarray, max_steps: int = 2000):
        import matplotlib.pyplot as plt

        yt_flat = yt.reshape(-1, yt.shape[-1])
        yp_flat = yp.reshape(-1, yp.shape[-1])
        n   = len(yt_flat)
        idx = np.linspace(0, n - 1, min(n, max_steps)).astype(int)

        fig, axes = plt.subplots(4, 1, figsize=(12, 10), sharex=True)
        for d, ax in enumerate(axes):
            ax.plot(yt_flat[idx, d], label="true", lw=1)
            ax.plot(yp_flat[idx, d], label="pred", lw=1, alpha=0.8)
            ax.set_ylabel(self.STATE_NAMES[d])
            ax.legend(loc="upper right", fontsize="small")
        axes[-1].set_xlabel("step (downsampled)")
        plt.suptitle("Δstate: true vs predicted (autoregressive test)")
        plt.tight_layout()

        fig_path = os.path.join(self.checkpoint_dir, "true_vs_pred.png")
        fig.savefig(fig_path, dpi=150)
        if os.environ.get("WANDB_MODE") != "disabled": _wandb().log({"true_vs_pred": _wandb().Image(fig)})
        plt.close(fig)
        print(f"Plot saved -> {fig_path}")
