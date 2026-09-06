#!/usr/bin/env python3
"""
Build a flat 11-column HDF5 dataset from hardware evaluation JSONs.

Scans evaluation JSONs and produces the flat transition format consumed
by offline RL (IQL) training. Supports filtering by date, controller
family, and success/failure.

Usage:
    # Build the deployed IQL input (post-recalibration, May 20+):
    python data/build_flat_dataset.py --out data/dataset/arcball_post_recalib_flat

    # Build with a step cap (for data-sweep experiments):
    python data/build_flat_dataset.py --max-steps 250000 --out /tmp/sweep_subset

    # Preview without writing:
    python data/build_flat_dataset.py --dry-run

    # Use a custom eval archive:
    python data/build_flat_dataset.py --archive-dir /path/to/eval_results/real
"""

import argparse
import glob
import json
import os
import re
import sys
import time
from collections import Counter
from pathlib import Path

import h5py
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_EVAL_DIR = ROOT / "evaluation" / "results" / "real"
DEFAULT_OUT = ROOT / "data" / "dataset" / "arcball_post_recalib_flat.h5"

MIN_STEPS = 5
DIP_BAND_RAD = 0.007  # |ball_angle| threshold for dip_reward=1

# Match folder paths containing dates >= 20260520 (post-recalibration)
POST_RECALIB_PAT = re.compile(r"2026052[0-9]|20260[6-9][0-9]{2}")


def load_trials(json_path: str):
    """Yield (obs_arr, actions_arr, controller_name, success) per valid trial."""
    with open(json_path) as f:
        data = json.load(f)
    for ctrl_name, trials in data["controllers"].items():
        for trial in trials:
            states = trial.get("states", [])
            actions = trial.get("actions", [])
            n = min(len(states), len(actions))
            if n < MIN_STEPS:
                continue
            obs = np.array(states[:n], dtype=np.float32)    # (n, 4)
            acts = np.array(actions[:n], dtype=np.float32)  # (n,)
            # Sanity bounds
            if (np.abs(obs[:, 0]).max() > 2.0
                    or np.abs(obs[:, 1]).max() > 5.0
                    or np.any(np.isnan(obs))
                    or np.any(np.isinf(obs))):
                continue
            yield obs, acts, ctrl_name, trial.get("success", None)


_TS_PAT = re.compile(r"(20\d{6})_(\d{6})")


def _path_ts(path: str):
    """Extract the embedded YYYYMMDDHHMMSS timestamp from a run-folder path."""
    m = _TS_PAT.search(path)
    return (m.group(1) + m.group(2)) if m else None


def collect_jsons(eval_dir: Path, date_filter: str | None,
                  after: str | None = None, before: str | None = None):
    """Find all data.json files, filtered by an explicit [after, before)
    timestamp window (YYYYMMDDHHMMSS) when given, else by the date pattern."""
    # Sort chronologically by the embedded run timestamp (recording order),
    # NOT by pathname, so the rebuilt transition stream matches the shipped
    # HDF5 row order exactly (training forms consecutive 600-step windows).
    all_jsons = sorted(
        glob.glob(str(eval_dir / "**" / "data.json"), recursive=True),
        key=lambda p: (_path_ts(p) or "", p),
    )
    if after or before:
        filtered = []
        for f in all_jsons:
            ts = _path_ts(f)
            if ts is None:
                continue
            if after and ts < after:
                continue
            if before and ts >= before:
                continue
            filtered.append(f)
    elif date_filter == "post_recalib":
        filtered = [f for f in all_jsons if POST_RECALIB_PAT.search(f)]
    else:
        filtered = all_jsons
    return all_jsons, filtered


def build(eval_dir: Path, out_path: Path, dry_run: bool,
          max_steps: int | None, date_filter: str | None,
          after: str | None = None, before: str | None = None):
    # Safety check - never silently overwrite
    if out_path.exists() and not dry_run:
        print(f"ERROR: {out_path} already exists. Delete it manually to rebuild.")
        sys.exit(1)

    all_jsons, filtered_jsons = collect_jsons(eval_dir, date_filter, after, before)
    print(f"Eval dir: {eval_dir}")
    print(f"Total JSONs: {len(all_jsons)}")
    print(f"Filtered JSONs: {len(filtered_jsons)}"
          + (f" ({date_filter})" if date_filter else ""))

    all_obs, all_next_obs, all_actions = [], [], []
    total_transitions = 0
    total_trials = 0
    n_success = 0
    n_fail = 0
    ctrl_counter = Counter()

    for jf in filtered_jsons:
        if max_steps is not None and total_transitions >= max_steps:
            break
        for obs, acts, ctrl_name, success in load_trials(jf):
            if max_steps is not None and total_transitions >= max_steps:
                break
            n = len(obs)
            remaining = max_steps - total_transitions if max_steps else n
            if n - 1 > remaining and max_steps:
                continue  # skip partial episode to keep episodes clean

            all_obs.append(obs[:-1])
            all_next_obs.append(obs[1:])
            all_actions.append(acts[:-1])
            total_transitions += n - 1
            total_trials += 1
            ctrl_counter[ctrl_name] += 1
            if success:
                n_success += 1
            else:
                n_fail += 1

    print(f"\nTrials: {total_trials}  (success={n_success}, fail={n_fail}, "
          f"SR={100*n_success/max(total_trials,1):.1f}%)")
    print(f"Transitions: {total_transitions:,}")

    # Guard against empty output
    if total_transitions == 0:
        print(
            f"No hardware evaluation runs found under {eval_dir} (0 transitions).\n"
            "These raw per-trial JSONs come from on-hardware data collection\n"
            "(evaluation/scripts/eval.py, data/collect_data.py) and are not shipped\n"
            "with the archive. You do NOT need this builder to train the offline-RL\n"
            "policy: the processed dataset is already provided at\n"
            "  data/dataset/arcball_real_demonstrations/arcball_post_recalib_flat.h5\n"
            "and is the default input to training/offline_rl/offline_train_cont.py.")
        sys.exit(1)

    print("\nTop 10 controllers:")
    for ctrl, cnt in ctrl_counter.most_common(10):
        print(f"  {ctrl}: {cnt} trials")

    if dry_run:
        print("\n[dry-run] No file written.")
        return

    obs_arr  = np.concatenate(all_obs,      axis=0).astype(np.float32)
    nobs_arr = np.concatenate(all_next_obs, axis=0).astype(np.float32)
    acts_arr = np.concatenate(all_actions,  axis=0).astype(np.float32)

    # dip_reward: ball at physical center
    ball_angle_next = nobs_arr[:, 2]
    dip_reward = (np.abs(ball_angle_next) <= DIP_BAND_RAD).astype(np.float32)
    stored_reward = np.zeros(len(obs_arr), dtype=np.float32)

    # 11-column layout matching offline_train_cont.py:
    # [prev_state(4), action(1), next_state(4), reward(1), dip_reward(1)]
    dataset = np.column_stack([
        obs_arr,
        acts_arr.reshape(-1, 1),
        nobs_arr,
        stored_reward.reshape(-1, 1),
        dip_reward.reshape(-1, 1),
    ]).astype(np.float32)

    print(f"\nOutput shape: {dataset.shape}")
    print(f"Dip reward fraction: {dip_reward.mean()*100:.1f}%")
    print(f"Action range: [{acts_arr.min():.3f}, {acts_arr.max():.3f}]")
    print(f"Ball angle range: [{ball_angle_next.min():.4f}, {ball_angle_next.max():.4f}]")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(out_path, "w") as f:
        f.create_dataset("dataset", data=dataset, compression="gzip", chunks=True)
        f.attrs["description"] = (
            "Flat 11-column hardware demonstration transitions "
            "for offline RL (IQL) training."
        )
        f.attrs["source_dir"] = str(eval_dir)
        f.attrs["n_jsons"] = len(filtered_jsons)
        f.attrs["n_trials"] = total_trials
        f.attrs["n_success"] = n_success
        f.attrs["n_fail"] = n_fail
        f.attrs["n_steps"] = total_transitions
        f.attrs["columns"] = (
            "prev_cart_pos, prev_cart_vel, prev_ball_pos, prev_ball_vel, action, "
            "cur_cart_pos, cur_cart_vel, cur_ball_pos, cur_ball_vel, reward, dip_reward"
        )
        if date_filter:
            f.attrs["date_filter"] = date_filter
        f.attrs["build_timestamp"] = time.strftime("%Y-%m-%d %H:%M:%S")

    size_mb = out_path.stat().st_size / (1024 * 1024)
    print(f"\nWritten: {out_path} ({size_mb:.1f} MB)")


if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--archive-dir", "--eval-dir", default=str(DEFAULT_EVAL_DIR),
                    help="Path to eval results (default: evaluation/results/real)")
    p.add_argument("--out", default=str(DEFAULT_OUT),
                    help="Output HDF5 path (must not already exist)")
    p.add_argument("--max-steps", type=int, default=None,
                    help="Cap total transitions (for data-sweep experiments)")
    p.add_argument("--date-filter", default="post_recalib",
                    choices=["all", "post_recalib"],
                    help="Filter JSONs by date pattern (ignored if --after/--before set)")
    # Exact recording window that reproduces the released IQL training corpus
    # (arcball_post_recalib_flat.h5: 1,415 trials / 333,497 transitions). The
    # 15:03 upper cutoff is 47 min before the deployed checkpoint's benchmark
    # run, so the training set is provably disjoint from IQL's 50 eval trials.
    p.add_argument("--after", default="20260520000000",
                    help="Include trials at/after this timestamp (YYYYMMDDHHMMSS)")
    p.add_argument("--before", default="20260528150300",
                    help="Include trials strictly before this timestamp")
    p.add_argument("--dry-run", action="store_true",
                    help="Preview stats without writing")
    args = p.parse_args()
    # An explicit "--date-filter all" overrides the default IQL-window cutoffs.
    after, before = args.after, args.before
    if args.date_filter == "all":
        after = before = None
    build(Path(args.archive_dir), Path(args.out), args.dry_run,
          args.max_steps, args.date_filter, after, before)
