#!/usr/bin/env python3
"""
Data-scaling sweep for LSTM world model.
No input noise, no LR scheduler, patience-based early stopping, clean data.

Trains on incremental data sizes from the clean dataset, evaluates each.
Saves everything to unique paths - no overriding.

Usage:
  conda activate balancer
  PYTHONPATH=/path/to/balancer WANDB_MODE=disabled python3 paper/ram/scripts/wm_data_sweep.py
"""

import os, sys, json, time, subprocess, shutil
from pathlib import Path
import numpy as np

REPO        = Path(__file__).resolve().parents[3]
BALANCER    = REPO / "balancer"
EXP_SCRIPT  = REPO / "paper" / "ram" / "scripts"
SIM_VAL_DIR = REPO / "paper" / "data" / "sim_validation"
PYTHON      = sys.executable

os.environ["PYTHONPATH"] = f"{BALANCER}:{os.environ.get('PYTHONPATH', '')}"

from balancer.world_model import ArcBallDataset, WorldModelConfig, WorldModelTrainer

# Capped at 1,000,000 rows - the full size of the shipped
# arcball_cont_1M_calib196_160.h5 dataset (the original 2.3M/3M collection
# campaigns are not shipped with this release).
SIZES = [10000, 25000, 50000, 70000, 100000, 150000, 200000, 300000, 400000, 500000, 1000000]

def train_and_eval(size):
    """Train WM on first `size` rows of clean dataset, then evaluate."""
    result_dir = REPO / "paper" / "data" / "wm_data_sweep" / str(size)
    ckpt_path  = result_dir / "model.pth"
    stats_path = result_dir / "norm_stats.npz"
    result_dir.mkdir(parents=True, exist_ok=True)

    # -- Skip if already complete --------------------------------------
    if (result_dir / "exp2_results.npz").exists() and ckpt_path.exists():
        print(f"[{size}] Already complete, skipping")
        return True

    # -- Train ---------------------------------------------------------
    print(f"\n{'='*60}")
    print(f"[{size}] Training on first {size} rows of clean dataset")
    print(f"{'='*60}")

    from torch.utils.data import DataLoader

    DATA_PATH = str(REPO / "data" / "dataset" / "arcball_cont_1M_calib196_160.h5")

    train_ds = ArcBallDataset(DATA_PATH, mode="train", seq_len=10, max_rows=size)
    val_ds   = ArcBallDataset(DATA_PATH, mode="val",   seq_len=10, max_rows=size)
    test_ds  = ArcBallDataset(DATA_PATH, mode="test",  seq_len=10, max_rows=size)

    train_loader = DataLoader(train_ds, batch_size=256, shuffle=True,  num_workers=4, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=256, shuffle=False, num_workers=4, pin_memory=True)
    test_loader  = DataLoader(test_ds,  batch_size=256, shuffle=False, num_workers=4, pin_memory=True)

    MODEL_CFG = WorldModelConfig(
        input_dim=5, state_dim=4, hidden_dim=64,
        num_layers=2, dropout=0.2,
        state_weights=[1.0, 5.0, 1.0, 5.0],
    )

    ckpt_dir = result_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    trainer = WorldModelTrainer(
        cfg=MODEL_CFG, train_loader=train_loader, val_loader=val_loader,
        test_loader=test_loader, epochs=200, lr=1e-4,
        checkpoint_dir=str(ckpt_dir), save_freq=10, patience=10,
    )

    t0 = time.time()
    trainer.train()
    train_time = time.time() - t0

    # -- Save best model + stats ---------------------------------------
    # The trainer saves to checkpoint_dir; copy to result_dir
    best_src = ckpt_dir / "best_model.pth"
    if best_src.exists():
        shutil.copy2(best_src, ckpt_path)
    # Also save norm_stats from dataset
    np.savez(stats_path,
             s_mean=train_ds.s_mean, s_std=train_ds.s_std,
             delta_mean=train_ds.delta_mean, delta_std=train_ds.delta_std)

    print(f"[{size}] Trained ({train_time:.0f}s)")

    # -- Run test ------------------------------------------------------
    trainer.test()

    # -- Evaluate with Exp1 + Exp2 ------------------------------------
    if not ckpt_path.exists():
        print(f"[{size}] FAIL - no checkpoint")
        return False

    env = {**os.environ, "WM_CKPT_OVERRIDE": str(ckpt_path)}

    for name, script in [("exp1", "exp1_short_horizon_prediction.py"),
                          ("exp2", "exp2_long_horizon_prediction.py")]:
        print(f"[{size}] Running {name}...")
        t0 = time.time()
        r = subprocess.run([str(PYTHON), str(EXP_SCRIPT / script)], env=env,
                          capture_output=True, text=True)
        elapsed = time.time() - t0
        with open(result_dir / f"{name}_log.txt", "w") as f:
            f.write(r.stdout + "\n--- STDERR ---\n" + r.stderr)
        src = SIM_VAL_DIR / f"{name}_results.npz"
        if src.exists():
            shutil.copy2(src, result_dir / f"{name}_results.npz")
            print(f"[{size}] {name} OK ({elapsed:.0f}s)")
        else:
            print(f"[{size}] {name} FAIL - no results")

    return True


def main():
    import wandb
    wandb.init(mode="disabled")
    results = {}
    for size in SIZES:
        ok = train_and_eval(size)
        results[str(size)] = {"ok": ok}
        with open(REPO / "paper" / "data" / "wm_data_sweep" / "progress.json", "w") as f:
            json.dump(results, f, indent=2)

    wandb.finish()
    print(f"\n{'='*60}")
    print("All done!")
    print(f"Results in {REPO / 'paper' / 'data' / 'wm_data_sweep'}/")

if __name__ == "__main__":
    main()
