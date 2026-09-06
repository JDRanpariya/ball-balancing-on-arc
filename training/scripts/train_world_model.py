#!/usr/bin/env python3
"""
Train the LSTM world model for ArcBall.

Canonical entry point for world-model training.  The model architecture,
dataset loader, and training loop live in ``balancer.world_model``.

Usage
-----
From the repository root::

    python scripts/train_world_model.py
    python scripts/train_world_model.py --epochs 50 --export --export-version v1

The ``--export`` flag copies the best checkpoint and normalisation stats to
``evaluation/models/world_model/<version>/``.
"""

import argparse
import os
import random
import shutil
from pathlib import Path

import numpy as np
import torch
import wandb
from torch.utils.data import DataLoader

from balancer.world_model import ArcBallDataset, WorldModelConfig, WorldModelTrainer

# -- Config --------------------------------------------------------------------

_REPO = Path(__file__).resolve().parents[2]

# Default is the pre_neg 1M collection: this trains the v1 world model that
# PPO-WM deploys. MPPI's world model (v12) instead trains on calib196_160 -
# pass --data-path .../arcball_cont_1M_calib196_160.h5 --export-version v12.
DATA_PATH   = str(_REPO / "data" / "dataset" / "arcball_cont_1M_pre_neg.h5")
BATCH_SIZE  = 256
TRAIN_SEQ   = 10
EPOCHS      = 50
LR          = 1e-4
SAVE_FREQ   = 10
TAIL_ROWS   = None   # use full dataset; set to int to use last N rows

MODEL_CFG = WorldModelConfig(
    input_dim=5, state_dim=4, hidden_dim=64,
    num_layers=2, dropout=0.2,
    state_weights=[1.0, 5.0, 1.0, 5.0],
)


def export_checkpoint(checkpoint_dir: str, version: str):
    """Copy best model + norm stats to evaluation/models/world_model/<version>/."""
    dst = _REPO / "evaluation" / "models" / "world_model" / version
    dst.mkdir(parents=True, exist_ok=True)

    best = Path(checkpoint_dir) / "best_model.pth"
    if best.exists():
        shutil.copy2(best, dst / "best_model.pth")
        print(f"Exported best_model.pth -> {dst}/")

    stats_src = Path(checkpoint_dir).parent / "norm_stats.npz"
    if stats_src.exists():
        shutil.copy2(stats_src, dst / "norm_stats.npz")
        print(f"Exported norm_stats.npz -> {dst}/")


def main():
    parser = argparse.ArgumentParser(description="Train ArcBall LSTM world model")
    parser.add_argument("--epochs", type=int, default=EPOCHS, help="Number of epochs")
    parser.add_argument("--lr", type=float, default=LR, help="Learning rate")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE, help="Batch size")
    parser.add_argument("--data-path", default=DATA_PATH, help="Path to HDF5 dataset")
    parser.add_argument("--export", action="store_true", help="Export best model to evaluation/")
    parser.add_argument("--export-version", default="v1", help="Version subdirectory")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--no-wandb", action="store_true",
                        help="Disable wandb logging (no login/offline mode needed)")
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    if args.no_wandb:
        os.environ["WANDB_MODE"] = "disabled"
    wandb.init(project="arcball-world-model")
    run_id = getattr(wandb.run, "id", None) or "local"
    checkpoint_dir = os.path.join("experiments", f"run_{run_id}", "checkpoints")

    # -- Data --------------------------------------------------------------
    train_ds = ArcBallDataset(args.data_path, mode="train", seq_len=TRAIN_SEQ, tail_rows=TAIL_ROWS)
    val_ds   = ArcBallDataset(args.data_path, mode="val",   seq_len=TRAIN_SEQ, tail_rows=TAIL_ROWS)
    test_ds  = ArcBallDataset(args.data_path, mode="test",  seq_len=TRAIN_SEQ, tail_rows=TAIL_ROWS)

    # Save normalisation stats now so the checkpoint dir is complete even if
    # --export is used before training finishes (predictor.py loads these).
    stats_path = Path(checkpoint_dir).parent / "norm_stats.npz"
    stats_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(stats_path,
             s_mean=train_ds.s_mean, s_std=train_ds.s_std,
             delta_mean=train_ds.delta_mean, delta_std=train_ds.delta_std)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,  num_workers=4, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)
    test_loader  = DataLoader(test_ds,  batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)

    # -- Train -------------------------------------------------------------
    trainer = WorldModelTrainer(
        cfg=MODEL_CFG, train_loader=train_loader, val_loader=val_loader,
        test_loader=test_loader, epochs=args.epochs, lr=args.lr,
        checkpoint_dir=checkpoint_dir, save_freq=SAVE_FREQ,
    )
    trainer.train()
    trainer.test()

    # -- Export ------------------------------------------------------------
    if args.export:
        export_checkpoint(checkpoint_dir, args.export_version)

    wandb.finish()


if __name__ == "__main__":
    main()
