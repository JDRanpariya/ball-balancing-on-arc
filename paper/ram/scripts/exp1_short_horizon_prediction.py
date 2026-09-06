#!/usr/bin/env python3
"""
Exp 1: Short-Horizon Open-Loop Prediction (h=5..50, N=100)

Compares analytical model variants + world model against real hardware data.
Output: supplementary/data/sim_validation/exp1_results.npz

Uses only the main paper repo:
  - Dataset: data/dataset/arcball_cont_1M_calib196_160.h5 (the last-1M slice of the
    original 2.3M calib196 collection campaign; the part the v12 world model was
    trained on - the earlier/larger campaign files are not shipped)
  - Analytical model: balancer.core.dynamics.dynamics_velocity_input
  - World model: balancer.world_model.WorldModelPredictor (v12 checkpoint)
"""

import sys, time, warnings
from pathlib import Path

import os
import h5py
import numpy as np
import torch

# -- Project paths ----------------------------------------------------------
_REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_REPO / "balancer"))

from balancer.core.dynamics import dynamics_velocity_input, DEFAULT_PARAMS
from balancer.world_model import WorldModelPredictor

# -- Constants --------------------------------------------------------------
DT = 0.05
V_MAX = 0.9
CART_LIMIT = 0.7765
BALL_LIMIT = 0.081
STATE_LABELS = ["x", "xd", "th", "thd"]

DATASET_PATH = _REPO / "data" / "dataset" / "arcball_cont_1M_calib196_160.h5"
WM_CKPT      = Path(os.environ.get("WM_CKPT_OVERRIDE",
                 str(_REPO / "evaluation" / "models" / "world_model" / "v12" / "best_model.pth")))
OUT_DIR      = _REPO / "paper" / "data" / "sim_validation"

RNG = np.random.default_rng(42)

HORIZONS = [5, 10, 15, 20, 30, 50]
N_WANT = 100
THETA_MAX = 0.04  # centre-only for Exp 1


# -- Helpers ----------------------------------------------------------------

def load_dataset(path):
    """Load HDF5, return (states, actions, next_states).

    arcball_cont_1M_calib196_160.h5 is already the last-1M slice of the original
    2.3M calib196 collection campaign (the part the v12 world model was trained
    on), so no further slicing is needed for a fair comparison between
    analytical and learned models.
    """
    with h5py.File(str(path), "r") as f:
        raw = f["dataset"][:]
    # 15-col layout: [0:4]=state, [6]=action, [7:11]=next_state
    states      = raw[:, 0:4].astype(np.float64)
    actions     = raw[:, 6].astype(np.float64)
    next_states = raw[:, 7:11].astype(np.float64)
    print(f"  Dataset: {len(states):,} transitions")
    return states, actions, next_states


def find_center_starts(states, next_states, max_h, n_want, theta_max=0.04):
    """Find start indices where |theta| < theta_max, no resets in segment."""
    N = len(states)
    # Detect resets
    cart_jump = np.max(np.abs(states[1:, 0:2] - next_states[:-1, 0:2]), axis=1)
    is_reset = np.concatenate([[False], cart_jump > 0.05])
    cum_reset = np.cumsum(is_reset)

    candidates = np.where(np.abs(states[:, 2]) < theta_max)[0]
    # Restrict to the validation partition (rows 80-90% of the 80/10/10 split
    # used to train the v12 world model). This keeps the analytical-model
    # selection here out-of-sample AND disjoint from the Exp 2 test-partition
    # comparison (rows 90-100%).
    candidates = candidates[(candidates < N - max_h)
                            & (candidates >= int(0.8 * N))
                            & (candidates < int(0.9 * N))]
    RNG.shuffle(candidates)

    valid = []
    for s in candidates:
        if cum_reset[s + max_h] - cum_reset[s] == 0:
            valid.append(int(s))
        if len(valid) >= n_want:
            break
    valid.sort()
    print(f"  Found {len(valid)}/{n_want} centre starts (|θ| < {theta_max})")
    return valid


def enforce_limits(s, restitution=0.3):
    """Clamp cart position + ball angle, apply inelastic bounce."""
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
    """Zero velocity command if cart at limit and moving outward."""
    if x >= CART_LIMIT and v_cmd > 0:
        return 0.0
    if x <= -CART_LIMIT and v_cmd < 0:
        return 0.0
    return v_cmd


# -- Model rollouts ---------------------------------------------------------

def analytical_step(s, v_cmd, params):
    """One Euler step of velocity-input dynamics + clipping."""
    v_cmd = np.clip(v_cmd, -V_MAX, V_MAX)
    v_cmd = soft_limit_vcmd(v_cmd, s[0])
    ds = dynamics_velocity_input(s, v_cmd, params)
    s_new = s + DT * ds
    return enforce_limits(s_new)


def analytical_rollout(s0, actions, params):
    """Open-loop N-step analytical rollout."""
    traj = [s0.copy()]
    s = s0.copy()
    for a in actions:
        s = analytical_step(s, a, params)
        traj.append(s.copy())
    return np.array(traj)


def analytical_step_with_delay(s, v_cmd, params, delay_buffer):
    """Analytical step with command delay buffer."""
    # Push current command into buffer, pop oldest
    delay_buffer.append(v_cmd)
    effective_cmd = delay_buffer.pop(0)
    return analytical_step(s, effective_cmd, params), delay_buffer


def analytical_rollout_with_delay(s0, actions, params, delay_steps=1):
    """Open-loop rollout with command delay."""
    delay_buffer = [0.0] * delay_steps
    traj = [s0.copy()]
    s = s0.copy()
    for a in actions:
        s, delay_buffer = analytical_step_with_delay(s, a, params, delay_buffer)
        traj.append(s.copy())
    return np.array(traj)


def analytical_rollout_with_delay_filter(s0, actions, params, delay_steps=1, filter_tau=0.01):
    """Open-loop rollout with command delay + PT1 velocity observation filter.
    
    The PT1 filter smooths velocity observations: 
    filtered = alpha * raw + (1-alpha) * prev_filtered
    where alpha = DT / (filter_tau + DT)
    """
    alpha = DT / (filter_tau + DT)
    delay_buffer = [0.0] * delay_steps
    traj = [s0.copy()]
    s = s0.copy()
    # Initialise filtered velocity to the initial state's velocity
    filt_xd, filt_thd = s[1], s[3]
    for a in actions:
        # Apply delay
        delay_buffer.append(a)
        v_cmd = delay_buffer.pop(0)
        # Step dynamics
        v_cmd = np.clip(float(v_cmd), -V_MAX, V_MAX)
        v_cmd = soft_limit_vcmd(v_cmd, s[0])
        ds = dynamics_velocity_input(s, v_cmd, params)
        s_next = enforce_limits(s + DT * ds)
        # Apply PT1 filter to velocity components
        filt_xd = alpha * s_next[1] + (1 - alpha) * filt_xd
        filt_thd = alpha * s_next[3] + (1 - alpha) * filt_thd
        s = np.array([s_next[0], filt_xd, s_next[2], filt_thd])
        traj.append(s.copy())
    return np.array(traj)


def wm_rollout(s0, actions, predictor):
    """Open-loop N-step world model rollout with LSTM hidden state."""
    traj = [s0.copy()]
    s = s0.copy()
    hc = None  # LSTM hidden state, maintained across steps
    for a in actions:
        s_t = torch.tensor(s, dtype=torch.float32, device=predictor.device).unsqueeze(0)
        a_t = torch.tensor([[float(a)]], dtype=torch.float32, device=predictor.device)
        s_norm = predictor.normalise_state(s_t)
        a_norm = predictor.normalise_action(a_t)
        x = torch.cat([s_norm, a_norm], dim=-1).unsqueeze(1)  # (1, 1, 5)
        delta_norm, hc = predictor.model(x, hc)
        delta = predictor.denormalise_delta(delta_norm[:, 0, :])
        s = (s_t + delta).squeeze(0).detach().cpu().numpy()
        traj.append(s.copy())
    return np.array(traj)


# -- Metrics ----------------------------------------------------------------

def compute_errors(pred_traj, real_traj):
    """Return (terminal_mae, traj_mae) both shape (4,)."""
    abs_err = np.abs(pred_traj - real_traj)
    terminal = abs_err[-1]
    traj_mean = abs_err.mean(axis=0)
    return terminal, traj_mean


# -- Build model configs ---------------------------------------------------

def build_models(predictor):
    """Return dict of {name: rollout_fn} for all variants."""
    models = {}
    p_bare = dict(DEFAULT_PARAMS)
    p_bare["tau_v"] = 0.06

    p_full = dict(DEFAULT_PARAMS)
    p_full["tau_v"] = 0.15

    models["Bare (tau=0.06)"]     = lambda s0, acts: analytical_rollout(s0, acts, p_bare)
    models["+tau=0.15"]           = lambda s0, acts: analytical_rollout(s0, acts, p_full)
    models["+Delay (1 step)"]     = lambda s0, acts: analytical_rollout_with_delay(s0, acts, p_full, 1)
    models["+Delay+Obs filter"]   = lambda s0, acts: analytical_rollout_with_delay_filter(s0, acts, p_full, 1)

    # World model
    if predictor is not None:
        models["WorldModel (LSTM)"] = lambda s0, acts, p=predictor: wm_rollout(s0, acts, p)

    return models


# -- Main -------------------------------------------------------------------

def run_exp1():
    print("=" * 60)
    print("  EXP 1: Short-Horizon Open-Loop Prediction")
    print("=" * 60)

    # Load dataset
    print("\nLoading dataset...")
    states, actions, next_states = load_dataset(DATASET_PATH)
    real_trajs = np.concatenate([states[:1], next_states], axis=0)

    # Find segments
    max_h = max(HORIZONS)
    starts = find_center_starts(states, next_states, max_h, N_WANT, THETA_MAX)
    if len(starts) < 10:
        print("  WARNING: too few segments, using all available")
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

    # Build models
    models = build_models(predictor)

    # Run
    results = {}
    for name, rollout_fn in models.items():
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

                pred_traj = rollout_fn(s0, acts)
                t_err, tr_err = compute_errors(pred_traj, real_traj)
                term_errs.append(t_err)
                traj_errs.append(tr_err)

            if term_errs:
                results[name][h] = {
                    "terminal_mae": np.mean(term_errs, axis=0),
                    "traj_mae": np.mean(traj_errs, axis=0),
                    "n_segments": len(term_errs),
                }
        elapsed = time.time() - t0
        final = results[name].get(max_h, {})
        if final:
            tm = final["terminal_mae"]
            print(f"  {name:<22s} h={max_h}: "
                  f"x={tm[0]:.4f} xd={tm[1]:.4f} "
                  f"th={tm[2]:.5f} thd={tm[3]:.4f}  ({elapsed:.1f}s)")

    # Save
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUT_DIR / "exp1_results.npz"
    flat = {}
    for name, horizons in results.items():
        for h, metrics in horizons.items():
            for k, v in metrics.items():
                safe_name = name.replace(" ", "_").replace("(", "").replace(")", "")
                flat[f"{safe_name}_h{h}_{k}"] = v
    np.savez(path, **flat)
    print(f"\n  Saved {path}")
    print(f"  {len(flat)} arrays, {len(results)} models\n")
    return results


if __name__ == "__main__":
    run_exp1()
