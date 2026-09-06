#!/usr/bin/env python3
"""
Exp 2: Long-Horizon Open-Loop Prediction (h=5..600, N=50)

Compares best analytical model vs world model across full RL-episode length.
Output: supplementary/data/sim_validation/exp2_results.npz

Dataset: data/dataset/arcball_cont_1M_calib196_160.h5 (the last-1M slice of the
original 2.3M calib196 collection campaign; the part the v12 world model was
trained on - the earlier/larger campaign files are not shipped)
"""

import sys, time
from pathlib import Path

import os
import h5py
import numpy as np
import torch

_REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_REPO / "balancer"))

from balancer.core.dynamics import dynamics_velocity_input, DEFAULT_PARAMS
from balancer.world_model import WorldModelPredictor

# -- Constants --------------------------------------------------------------
DT = 0.05
V_MAX = 0.9
CART_LIMIT = 0.7765
BALL_LIMIT = 0.081

DATASET_PATH = _REPO / "data" / "dataset" / "arcball_cont_1M_calib196_160.h5"
WM_CKPT      = Path(os.environ.get("WM_CKPT_OVERRIDE",
                 str(_REPO / "evaluation" / "models" / "world_model" / "v12" / "best_model.pth")))
OUT_DIR      = _REPO / "paper" / "data" / "sim_validation"

RNG = np.random.default_rng(42)

HORIZONS = [5, 10, 20, 50, 100, 200, 300, 400, 600]
N_WANT = 50  # general starts (no IC filter)


# -- Helpers (same as Exp 1) -----------------------------------------------

def load_dataset(path):
    """Load HDF5, return (states, actions, next_states).

    arcball_cont_1M_calib196_160.h5 is already the last-1M slice of the
    original 2.3M calib196 campaign where v12 was trained, so no further
    slicing is needed.
    """
    with h5py.File(str(path), "r") as f:
        raw = f["dataset"][:]
    states = raw[:, 0:4].astype(np.float64)
    actions = raw[:, 6].astype(np.float64)
    next_states = raw[:, 7:11].astype(np.float64)
    print(f"  Dataset: {len(states):,} transitions")
    return states, actions, next_states


def find_starts(states, next_states, max_h, n_want):
    """General starts (any IC), no resets in segment."""
    N = len(states)
    cart_jump = np.max(np.abs(states[1:, 0:2] - next_states[:-1, 0:2]), axis=1)
    is_reset = np.concatenate([[False], cart_jump > 0.05])
    cum_reset = np.cumsum(is_reset)

    # Restrict to the held-out test partition (last 10% of the 80/10/10 split
    # used to train the v12 world model) so the evaluation is out-of-sample.
    candidates = RNG.permutation(np.arange(int(0.9 * N), N - max_h))
    valid = []
    for s in candidates:
        if cum_reset[s + max_h] - cum_reset[s] == 0:
            valid.append(int(s))
        if len(valid) >= n_want:
            break
    valid.sort()
    print(f"  Found {len(valid)}/{n_want} general starts")
    return valid


def enforce_limits(s, restitution=0.3):
    x, xd, th, thd = s
    if abs(x) > CART_LIMIT:
        x = np.clip(x, -CART_LIMIT, CART_LIMIT)
        if (x >= CART_LIMIT and xd > 0) or (x <= -CART_LIMIT and xd < 0):
            xd = 0.0
    if abs(th) > BALL_LIMIT:
        th = np.clip(th, -BALL_LIMIT, BALL_LIMIT)
        if np.sign(th) * thd > 0:
            thd = -restitution * thd
    return np.array([x, xd, th, thd])


def soft_limit_vcmd(v_cmd, x):
    if x >= CART_LIMIT and v_cmd > 0:
        return 0.0
    if x <= -CART_LIMIT and v_cmd < 0:
        return 0.0
    return v_cmd


# -- Analytical model (best config: tau=0.15 + delay=1) --------------------

def analytical_rollout(s0, actions, params=None, delay_steps=1, filter_tau=0.01):
    """Open-loop N-step analytical rollout with 1-step command delay and a
    PT1 velocity observation filter (the best analytical config from Exp 1)."""
    if params is None:
        params = dict(DEFAULT_PARAMS)
        params["tau_v"] = 0.15
    alpha = DT / (filter_tau + DT)
    delay_buffer = [0.0] * delay_steps
    traj = [s0.copy()]
    s = s0.copy()
    filt_xd, filt_thd = s[1], s[3]
    for a in actions:
        delay_buffer.append(a)
        v_cmd = delay_buffer.pop(0)
        v_cmd = np.clip(float(v_cmd), -V_MAX, V_MAX)
        v_cmd = soft_limit_vcmd(v_cmd, s[0])
        ds = dynamics_velocity_input(s, v_cmd, params)
        s_next = enforce_limits(s + DT * ds)
        filt_xd = alpha * s_next[1] + (1 - alpha) * filt_xd
        filt_thd = alpha * s_next[3] + (1 - alpha) * filt_thd
        s = np.array([s_next[0], filt_xd, s_next[2], filt_thd])
        traj.append(s.copy())
    return np.array(traj)


# -- World model -----------------------------------------------------------

def wm_rollout(s0, actions, predictor):
    """Open-loop N-step world model rollout with LSTM hidden state."""
    traj = [s0.copy()]
    s = s0.copy()
    hc = None
    for a in actions:
        s_t = torch.tensor(s, dtype=torch.float32, device=predictor.device).unsqueeze(0)
        a_t = torch.tensor([[float(a)]], dtype=torch.float32, device=predictor.device)
        s_norm = predictor.normalise_state(s_t)
        a_norm = predictor.normalise_action(a_t)
        x = torch.cat([s_norm, a_norm], dim=-1).unsqueeze(1)
        delta_norm, hc = predictor.model(x, hc)
        delta = predictor.denormalise_delta(delta_norm[:, 0, :])
        s = (s_t + delta).squeeze(0).detach().cpu().numpy()
        traj.append(s.copy())
    return np.array(traj)


# -- Main -------------------------------------------------------------------

def run_exp2():
    print("=" * 60)
    print("  EXP 2: Long-Horizon Open-Loop Prediction")
    print("=" * 60)

    print("\nLoading dataset...")
    states, actions, next_states = load_dataset(DATASET_PATH)
    real_trajs = np.concatenate([states[:1], next_states], axis=0)

    max_h = max(HORIZONS)
    starts = find_starts(states, next_states, max_h, N_WANT)
    print(f"  Using {len(starts)} segments, max_h={max_h}\n")

    # Load world model
    predictor = None
    if WM_CKPT.exists():
        print("Loading world model...")
        try:
            predictor = WorldModelPredictor(str(WM_CKPT))
            print(f"  {predictor}")
        except Exception as e:
            print(f"  WARNING: could not load world model: {e}")
    else:
        print(f"  WARNING: world model not found at {WM_CKPT}")

    results = {}

    # Analytical model
    name = "Analytical (best)"
    results[name] = {}
    t0 = time.time()
    for h in HORIZONS:
        term_errs, traj_errs = [], []
        for idx in starts:
            if idx + h >= len(states):
                continue
            s0 = states[idx]
            acts = actions[idx:idx + h]
            real_traj = real_trajs[idx:idx + h + 1]
            pred_traj = analytical_rollout(s0, acts)
            abs_err = np.abs(pred_traj - real_traj)
            term_errs.append(abs_err[-1])
            traj_errs.append(abs_err.mean(axis=0))
        if term_errs:
            results[name][h] = {
                "terminal_mae": np.mean(term_errs, axis=0),
                "traj_mae": np.mean(traj_errs, axis=0),
                "total_mae": np.mean(traj_errs).sum(),
                "n_segments": len(term_errs),
            }
    elapsed = time.time() - t0
    final = results[name].get(max_h, {})
    if final:
        print(f"  {name:<25s} h={max_h}: total={final['total_mae']:.4f} ({elapsed:.1f}s)")

    # World model
    if predictor is not None:
        name = "WorldModel (LSTM)"
        results[name] = {}
        t0 = time.time()
        for h in HORIZONS:
            term_errs, traj_errs = [], []
            for idx in starts:
                if idx + h >= len(states):
                    continue
                s0 = states[idx]
                acts = actions[idx:idx + h]
                real_traj = real_trajs[idx:idx + h + 1]
                pred_traj = wm_rollout(s0, acts, predictor)
                abs_err = np.abs(pred_traj - real_traj)
                term_errs.append(abs_err[-1])
                traj_errs.append(abs_err.mean(axis=0))
            if term_errs:
                results[name][h] = {
                    "terminal_mae": np.mean(term_errs, axis=0),
                    "traj_mae": np.mean(traj_errs, axis=0),
                    "total_mae": np.mean(traj_errs).sum(),
                    "n_segments": len(term_errs),
                }
        elapsed = time.time() - t0
        final = results[name].get(max_h, {})
        if final:
            print(f"  {name:<25s} h={max_h}: total={final['total_mae']:.4f} ({elapsed:.1f}s)")

    # Save
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUT_DIR / "exp2_results.npz"
    flat = {}
    for name, horizons in results.items():
        for h, metrics in horizons.items():
            for k, v in metrics.items():
                safe_name = name.replace(" ", "_").replace("(", "").replace(")", "")
                flat[f"{safe_name}_h{h}_{k}"] = v
    np.savez(path, **flat)
    print(f"\n  Saved {path}")
    print(f"  {len(flat)} arrays\n")
    return results


if __name__ == "__main__":
    run_exp2()
