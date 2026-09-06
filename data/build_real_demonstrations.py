#!/usr/bin/env python3
"""
Build arcball_real_demonstrations.h5 from evaluation hardware trial JSONs.

Scans all evaluation/results/real/**/data.json files and assembles a
self-documenting HDF5 dataset of controller demonstrations (success + failure).

Idempotent: re-run after new trials to regenerate with everything included.

Usage:
    python build_real_demonstrations.py                  # build full dataset
    python build_real_demonstrations.py --dry-run        # preview without writing
    python build_real_demonstrations.py --success-only   # only successful episodes
"""

import argparse
import glob
import json
import os
import sys
import time
from pathlib import Path

import h5py
import numpy as np

# -- Project imports --
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "balancer"))
from balancer.hardware.constants import SYSTEM


# ============================================================================
# CONSTANTS
# ============================================================================

EVAL_RESULTS_DIR = Path(__file__).resolve().parent.parent / "evaluation" / "results" / "real"
OUTPUT_DIR = Path(__file__).resolve().parent / "dataset" / "arcball_real_demonstrations"
OUTPUT_HDF5 = OUTPUT_DIR / "arcball_real_demonstrations.h5"

MIN_EPISODE_STEPS = 5  # Skip trivially short episodes

# Controller family classification
FAMILY_MAP = {
    "PIDController": "PID",
    "OffsetCorrectedPID": "PID",
    "PIDMargin": "PID",
    "LQRController": "LQR",
    "MPCController": "MPC",
    "NMPCController": "NMPC",
    "SMCController": "SMC",
    "SMC_Force": "SMC",
    "MPPIController": "MPPI",
    "RLController": "RL",
    "RLHardwareController": "RL",
}


# ============================================================================
# HELPERS
# ============================================================================

def classify_family(controller_name: str) -> str:
    """Map controller tag to family name."""
    for prefix, family in FAMILY_MAP.items():
        if controller_name.startswith(prefix):
            return family
    return "Other"


def extract_episodes(json_path: str, success_only: bool = False):
    """Yield episode dicts from a single evaluation JSON."""
    with open(json_path) as f:
        data = json.load(f)

    ts = data.get("Ts", 0.05)
    source = os.path.relpath(json_path, EVAL_RESULTS_DIR)

    for ctrl_name, trials in data["controllers"].items():
        family = classify_family(ctrl_name)
        for trial in trials:
            if success_only and not trial.get("success", False):
                continue

            states = trial.get("states", [])
            actions = trial.get("actions", [])
            timestamps = trial.get("timestamps", [])

            n = min(len(states), len(actions))
            if n < MIN_EPISODE_STEPS:
                continue

            states = states[:n]
            actions = actions[:n]
            timestamps = timestamps[:n] if len(timestamps) >= n else [
                i * ts for i in range(n)
            ]

            # Sanity check: skip episodes with clearly corrupt sensor data
            obs_arr = np.array(states, dtype=np.float32)
            if (np.abs(obs_arr[:, 0]).max() > 2.0 or
                    np.abs(obs_arr[:, 1]).max() > 5.0 or
                    np.any(np.isnan(obs_arr)) or
                    np.any(np.isinf(obs_arr))):
                continue

            yield {
                "observations": obs_arr,
                "actions": np.array(actions, dtype=np.float32),
                "timestamps": np.array(timestamps, dtype=np.float32),
                # Metadata
                "controller_name": ctrl_name,
                "controller_family": family,
                "success": trial.get("success", False),
                "settling_time_s": trial.get("settling_time") or float("nan"),
                "difficulty": trial.get("difficulty", "unknown"),
                "initial_cart_pos_m": trial.get("cart_target_m", 0.0),
                "ise_theta": trial.get("ise_theta", float("nan")),
                "ise_u": trial.get("ise_u", float("nan")),
                "duration_s": timestamps[-1] if timestamps else n * ts,
                "source_file": source,
                "Ts": ts,
            }


# ============================================================================
# BUILD DATASET
# ============================================================================

def build_dataset(success_only: bool = False, dry_run: bool = False):
    """Scan JSONs and write HDF5."""

    json_files = sorted(glob.glob(str(EVAL_RESULTS_DIR / "**" / "data.json"), recursive=True))
    print(f"Found {len(json_files)} result files in {EVAL_RESULTS_DIR}")

    if not json_files:
        print("No data.json files found. Nothing to do.")
        return

    # -- Collect all episodes --
    episodes = []
    for jf in json_files:
        for ep in extract_episodes(jf, success_only=success_only):
            episodes.append(ep)

    print(f"Extracted {len(episodes)} episodes ({sum(e['success'] for e in episodes)} successful)")
    total_steps = sum(len(e["observations"]) for e in episodes)
    print(f"Total timesteps: {total_steps:,}")

    if dry_run:
        # Print summary by family
        from collections import Counter
        fam_counts = Counter(e["controller_family"] for e in episodes)
        print("\nBy controller family:")
        for fam, count in sorted(fam_counts.items(), key=lambda x: -x[1]):
            print(f"  {fam:<8} {count:>5} episodes")
        return

    # -- Build arrays --
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Flat arrays
    all_obs = []
    all_next_obs = []
    all_actions = []
    all_terminals = []
    all_episode_ids = []
    episode_boundaries = [0]

    # Per-episode metadata
    meta_controller_name = []
    meta_controller_family = []
    meta_episode_length = []
    meta_success = []
    meta_settling_time = []
    meta_difficulty = []
    meta_initial_cart_pos = []
    meta_ise_theta = []
    meta_ise_u = []
    meta_duration = []
    meta_source_file = []

    for ep_idx, ep in enumerate(episodes):
        obs = ep["observations"]
        acts = ep["actions"]
        n = len(obs)

        # Flat transition arrays: (s_t, a_t, s_{t+1})
        # Last step has no next_obs, so we use n-1 transitions
        all_obs.append(obs[:-1])
        all_next_obs.append(obs[1:])
        all_actions.append(acts[:-1])

        terms = np.zeros(n - 1, dtype=bool)
        terms[-1] = True  # Mark last transition as terminal
        all_terminals.append(terms)

        all_episode_ids.append(np.full(n - 1, ep_idx, dtype=np.int32))
        episode_boundaries.append(episode_boundaries[-1] + n - 1)

        # Metadata
        meta_controller_name.append(ep["controller_name"])
        meta_controller_family.append(ep["controller_family"])
        meta_episode_length.append(n)
        meta_success.append(ep["success"])
        meta_settling_time.append(ep["settling_time_s"])
        meta_difficulty.append(ep["difficulty"])
        meta_initial_cart_pos.append(ep["initial_cart_pos_m"])
        meta_ise_theta.append(ep["ise_theta"])
        meta_ise_u.append(ep["ise_u"])
        meta_duration.append(ep["duration_s"])
        meta_source_file.append(ep["source_file"])

    # Concatenate
    flat_obs = np.concatenate(all_obs, axis=0)
    flat_next_obs = np.concatenate(all_next_obs, axis=0)
    flat_actions = np.concatenate(all_actions, axis=0)
    flat_terminals = np.concatenate(all_terminals, axis=0)
    flat_episode_ids = np.concatenate(all_episode_ids, axis=0)
    episode_boundaries = np.array(episode_boundaries, dtype=np.int64)

    print(f"Flat arrays shape: {flat_obs.shape[0]:,} transitions")

    # -- Write HDF5 --
    print(f"Writing {OUTPUT_HDF5}...")
    with h5py.File(OUTPUT_HDF5, "w") as f:

        # -- Episodes (indexed by group) --
        ep_grp = f.create_group("episodes")
        for ep_idx, ep in enumerate(episodes):
            g = ep_grp.create_group(str(ep_idx))
            g.create_dataset("observations", data=ep["observations"], compression="gzip")
            g.create_dataset("actions", data=ep["actions"], compression="gzip")
            g.create_dataset("timestamps", data=ep["timestamps"], compression="gzip")

        # -- Flat view --
        flat_grp = f.create_group("flat")
        flat_grp.create_dataset("observations", data=flat_obs, compression="gzip")
        flat_grp.create_dataset("next_observations", data=flat_next_obs, compression="gzip")
        flat_grp.create_dataset("actions", data=flat_actions, compression="gzip")
        flat_grp.create_dataset("terminals", data=flat_terminals, compression="gzip")
        flat_grp.create_dataset("episode_ids", data=flat_episode_ids, compression="gzip")
        flat_grp.create_dataset("episode_boundaries", data=episode_boundaries)

        # -- Episode metadata --
        meta_grp = f.create_group("episode_metadata")
        dt_str = h5py.string_dtype()
        meta_grp.create_dataset("controller_name", data=meta_controller_name, dtype=dt_str)
        meta_grp.create_dataset("controller_family", data=meta_controller_family, dtype=dt_str)
        meta_grp.create_dataset("episode_length", data=np.array(meta_episode_length, dtype=np.int32))
        meta_grp.create_dataset("success", data=np.array(meta_success, dtype=bool))
        meta_grp.create_dataset("settling_time_s", data=np.array(meta_settling_time, dtype=np.float32))
        meta_grp.create_dataset("difficulty", data=meta_difficulty, dtype=dt_str)
        meta_grp.create_dataset("initial_cart_pos_m", data=np.array(meta_initial_cart_pos, dtype=np.float32))
        meta_grp.create_dataset("ise_theta", data=np.array(meta_ise_theta, dtype=np.float32))
        meta_grp.create_dataset("ise_u", data=np.array(meta_ise_u, dtype=np.float32))
        meta_grp.create_dataset("duration_s", data=np.array(meta_duration, dtype=np.float32))
        meta_grp.create_dataset("source_file", data=meta_source_file, dtype=dt_str)

        # -- System constants --
        sys_grp = f.create_group("system")
        sys_grp.attrs["arc_radius_m"] = SYSTEM.ARC_RADIUS_M
        sys_grp.attrs["cart_mass_kg"] = SYSTEM.CART_MASS_KG
        sys_grp.attrs["ball_mass_kg"] = SYSTEM.BALL_MASS_KG
        sys_grp.attrs["ball_radius_m"] = SYSTEM.BALL_RADIUS_M
        sys_grp.attrs["ball_limit_rad"] = SYSTEM.BALL_LIMIT
        sys_grp.attrs["cart_limit_m"] = SYSTEM.CART_LIMIT
        sys_grp.attrs["track_half_m"] = SYSTEM.track_half
        sys_grp.attrs["max_cart_vel_ms"] = SYSTEM.MAX_CART_VEL
        sys_grp.attrs["control_freq_hz"] = 20.0
        sys_grp.attrs["Ts_s"] = 0.05
        sys_grp.attrs["settling_band_rad"] = SYSTEM.SETTLING_BAND
        sys_grp.attrs["settling_duration_s"] = SYSTEM.SETTLING_DURATION
        sys_grp.attrs["fail_time_s"] = SYSTEM.FAIL_TIME
        sys_grp.attrs["gravity"] = SYSTEM.GRAVITY

        # -- Root attributes --
        f.attrs["description"] = (
            "Real hardware demonstrations from a ball-on-arc balancing system. "
            "Contains trajectories from classical controllers (PID, LQR, MPC, NMPC, SMC) "
            "and learned policies (RL). Includes both successful balancing episodes and "
            "failures. Collected on physical hardware with ToF distance sensors at 20 Hz."
        )
        f.attrs["version"] = "1.0"
        f.attrs["n_episodes"] = len(episodes)
        f.attrs["n_episodes_success"] = sum(meta_success)
        f.attrs["n_episodes_failure"] = len(episodes) - sum(meta_success)
        f.attrs["total_timesteps"] = int(flat_obs.shape[0])
        f.attrs["state_labels"] = ["cart_pos_m", "cart_vel_ms", "ball_angle_rad", "ball_angvel_rads"]
        f.attrs["action_description"] = "Cart velocity command in m/s, range [-0.9, 0.9]"
        f.attrs["collection_period"] = "2026-04 to 2026-05"
        f.attrs["build_timestamp"] = time.strftime("%Y-%m-%d %H:%M:%S")
        f.attrs["source_json_count"] = len(json_files)
        f.attrs["source_dir"] = str(EVAL_RESULTS_DIR)

    file_size = os.path.getsize(OUTPUT_HDF5) / (1024 * 1024)
    print(f"Done. {OUTPUT_HDF5} ({file_size:.1f} MB)")
    print(f"  Episodes: {len(episodes)} ({sum(meta_success)} success, {len(episodes)-sum(meta_success)} failure)")
    print(f"  Transitions: {flat_obs.shape[0]:,}")
    print(f"  Controller families: {sorted(set(meta_controller_family))}")


# ============================================================================
# CLI
# ============================================================================

def show_status():
    """Show what's changed since last build."""
    if not OUTPUT_HDF5.exists():
        print("No dataset built yet. Run without --status to build.")
        return

    with h5py.File(OUTPUT_HDF5, "r") as f:
        build_time = f.attrs.get("build_timestamp", "unknown")
        n_eps = f.attrs.get("n_episodes", 0)
        n_success = f.attrs.get("n_episodes_success", 0)
        n_files = f.attrs.get("source_json_count", 0)
        total_steps = f.attrs.get("total_timesteps", 0)

    print(f"Last build: {build_time}")
    print(f"  Episodes: {n_eps} ({n_success} success)")
    print(f"  Transitions: {total_steps:,}")
    print(f"  Source JSONs at build time: {n_files}")
    print()

    # Count current JSONs
    json_files = sorted(glob.glob(str(EVAL_RESULTS_DIR / "**" / "data.json"), recursive=True))
    new_files = [f for f in json_files if os.path.getmtime(f) > os.path.getmtime(OUTPUT_HDF5)]
    print(f"Current source JSONs: {len(json_files)}")
    print(f"New/modified since last build: {len(new_files)}")
    if new_files:
        print("\nNew files:")
        for nf in new_files[:15]:
            print(f"  {os.path.relpath(nf, EVAL_RESULTS_DIR)}")
        if len(new_files) > 15:
            print(f"  ... and {len(new_files) - 15} more")
        print(f"\nRun 'python build_real_demonstrations.py' to rebuild with new data.")
    else:
        print("Dataset is up to date.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Build real hardware demonstration dataset from evaluation trials.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--dry-run", action="store_true",
                        help="Preview stats without writing HDF5")
    parser.add_argument("--success-only", action="store_true",
                        help="Only include successful episodes")
    parser.add_argument("--status", action="store_true",
                        help="Show dataset status and new files since last build")
    args = parser.parse_args()

    if args.status:
        show_status()
    else:
        build_dataset(success_only=args.success_only, dry_run=args.dry_run)
