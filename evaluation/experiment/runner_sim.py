"""
Simulation experiment runner - mirrors the hardware runner structure.

Uses the existing ExperimentConfig (experiment/config.py) without modification.
"""

from __future__ import annotations

import json
import time
import os
from pathlib import Path
from typing import List, Optional

import numpy as np

from balancer.core.dynamics import NonLinearDynamics
from .config import ExperimentConfig
from .initial_conditions import get_hardware_trial_configs
from balancer.hardware.constants import SYSTEM


# ===========================================================================

# CONSTANTS (match hardware limits)

# ===========================================================================

CART_MIN, CART_MAX = -SYSTEM.CART_LIMIT, SYSTEM.CART_LIMIT
THETA_MIN, THETA_MAX = -SYSTEM.BALL_LIMIT, SYSTEM.BALL_LIMIT
CART_LIMIT = SYSTEM.CART_LIMIT
DISCRETE_THRESHOLD = SYSTEM.DISCRETE_THRESHOLD

# Default IC sampling ranges (legacy - used only if stratified=False)

DEFAULT_CART_RANGE = (-0.75, 0.75)
DEFAULT_THETA_RANGE = (-0.074, 0.074)


# ===========================================================================

# SETTLING TIME DETECTION

# ===========================================================================

def _compute_settling_time(
    theta_hist: np.ndarray,
    time_hist: np.ndarray,
    band: float = SYSTEM.SETTLING_BAND,
    duration: float = SYSTEM.SETTLING_DURATION,
) -> Optional[float]:
    """
    Return the first time after which |θ| < band
    for at least ``duration`` seconds continuously.  Returns None if never settled.

    MATCHES tuning/simulate.py:settling_time() exactly:
    time-based comparison, returns the START of the settled window.
    """
    n = len(theta_hist)
    if n == 0:
        return None

    theta_abs = np.abs(theta_hist)
    in_band = theta_abs < band          # strict '<' matches tuning
    enter_time = None

    for i in range(n):
        if in_band[i]:
            if enter_time is None:
                enter_time = time_hist[i]
            # Time-based check (not sample-counting)
            if time_hist[i] - enter_time >= duration:
                return float(enter_time)
        else:
            enter_time = None

    return None


# ===========================================================================

# SINGLE TRIAL

# ===========================================================================

def _sample_ic(rng: np.random.Generator,
               cart_range=DEFAULT_CART_RANGE,
               theta_range=DEFAULT_THETA_RANGE) -> np.ndarray:
    cart = rng.uniform(*cart_range)
    theta = rng.uniform(*theta_range)
    return np.array([cart, 0.0, theta, 0.0])


def run_trial(
    controller,
    s0: np.ndarray,
    cfg: ExperimentConfig,
    rng: Optional[np.random.Generator] = None,
) -> dict:
    """
    Run one simulation trial.

    Returns a dict with the same schema as the hardware runner.
    """
    if rng is None:
        rng = np.random.default_rng()
    motor = getattr(cfg, 'motor_model', 'first_order')
    dyn = NonLinearDynamics(
        kinematics_integrator="rk4",
        tau=cfg.Ts,
        action_type=cfg.action_type,
        motor_model=motor,
        velocity_lag_tau=0.15,
        command_delay_steps=1,
        jerk_limit=50.0,
        accel_limit=15.0,
        decel_limit=15.0,
    )

    controller.reset()
    steps = int(cfg.run_time / cfg.Ts)
    s = s0.copy()

    max_vals = np.array([SYSTEM.CART_LIMIT, 1.0, SYSTEM.BALL_LIMIT, 1.0])

    states: List[list] = []
    timestamps: List[float] = []
    actions: List = []
    ctrl_times: List[float] = []

    prev_noisy_pos = s0.copy()
    first_step = True

    for k in range(steps):
        t = (k + 1) * cfg.Ts

        obs = s.copy()
        if cfg.noise_sigma > 0:
            # Add noise to positions only (mimics ToF/encoder sensor noise)
            pos_noise_cart = rng.normal(
                0, cfg.noise_sigma * SYSTEM.CART_LIMIT)
            pos_noise_ball = rng.normal(
                0, cfg.noise_sigma * SYSTEM.BALL_LIMIT)

            noisy_cart_pos = s[0] + pos_noise_cart
            noisy_ball_pos = s[2] + pos_noise_ball

            if first_step:
                # On first step, use true velocity (no derivative yet)
                noisy_cart_vel = s[1]
                noisy_ball_vel = s[3]
                first_step = False
            else:
                # Derive velocity from noisy positions (mimics VelocityEstimator)
                noisy_cart_vel = (noisy_cart_pos - prev_noisy_pos[0]) / cfg.Ts
                noisy_ball_vel = (noisy_ball_pos - prev_noisy_pos[2]) / cfg.Ts

            obs = np.array([noisy_cart_pos, noisy_cart_vel,
                           noisy_ball_pos, noisy_ball_vel])
            obs = np.clip(obs, -max_vals, max_vals)

            # Keep noisy-position history current so noisy_vel doesn't diverge
            prev_noisy_pos[0] = noisy_cart_pos
            prev_noisy_pos[2] = noisy_ball_pos

        t0 = time.perf_counter()
        u = controller.step(obs, cart_limit=CART_LIMIT)
        ctrl_dt_ms = (time.perf_counter() - t0) * 1000.0
        ctrl_times.append(ctrl_dt_ms)

        # Dynamics step
        if cfg.action_type == "discrete":
            if u > DISCRETE_THRESHOLD:
                action = 2
            elif u < -DISCRETE_THRESHOLD:
                action = 0
            else:
                action = 1
            s = dyn(s, action)
            actions.append(action)
        else:
            s = dyn(s, u)
            actions.append(float(u))

        if np.any(np.isnan(s)) or np.any(np.isinf(s)):
            break

        # Hard walls
        if s[0] <= CART_MIN or s[0] >= CART_MAX:
            s[0] = np.clip(s[0], CART_MIN, CART_MAX)
            s[1] = 0.0
        if s[2] <= THETA_MIN or s[2] >= THETA_MAX:
            s[2] = np.clip(s[2], THETA_MIN, THETA_MAX)
            s[3] = 0.0

        states.append(s.tolist())
        timestamps.append(t)

    # -- Post-process ----------------------------------------------
    if len(states) == 0:
        return {
            "settling_time": None,
            "success": False,
            "overshoot": float(SYSTEM.BALL_LIMIT),
            "violations": 0,
            "states": [],
            "timestamps": [],
            "actions": actions,
            "controller_time_ms": ctrl_times,
            "loop_period_ms": [],
            "initial_condition": s0.tolist(),
        }
    states_arr = np.array(states)
    time_arr = np.array(timestamps)
    theta_arr = states_arr[:, 2]

    # Settling time - matches tuning/simulate.py exactly
    settling = _compute_settling_time(
        theta_arr, time_arr,
        band=cfg.settling_band,
        duration=cfg.settling_duration,
    )

    # Overshoot + boundary violations - matches tuning/simulate.py exactly
    overshoot, boundary_violations = _compute_overshoot(
        theta_arr, cfg.settling_band,
    )

    ise_theta, ise_vel, ise_u = _compute_ise(
        time_arr, theta_arr, np.array(actions, dtype=float)
    )

    return {
        # -- eval metrics ----------------------------------------------
        "settling_time":  settling,
        "success":        settling is not None,
        "overshoot":      overshoot,
        "violations":     boundary_violations,
        # -- ISE metrics -----------------------------------------------
        "ise_theta":      ise_theta,
        "ise_vel":        ise_vel,
        "ise_u":          ise_u,
        # -- trajectory ------------------------------------------------
        "states":               states,
        "timestamps":           timestamps,
        "actions":              actions,
        "controller_time_ms":   ctrl_times,
        "loop_period_ms":       [cfg.Ts * 1000.0] * len(ctrl_times),
        "initial_condition":    s0.tolist(),
    }


def _compute_overshoot(theta_arr, tol):
    """Max |θ| after first entering settling band + boundary hit count.
    Identical to tuning/simulate.py:compute_overshoot."""
    theta_abs = np.abs(theta_arr)

    boundary_violations = int(np.sum(
        theta_abs >= (SYSTEM.BALL_LIMIT - 1e-4)
    ))

    in_band = theta_abs < tol
    first_entry = np.argmax(in_band)

    if not in_band[first_entry]:
        return float(np.nanmax(theta_abs)), boundary_violations

    post_entry = theta_abs[first_entry:]
    overshoot = float(np.nanmax(post_entry))

    return overshoot, boundary_violations


def _compute_ise(time_arr, theta_arr, u_arr, sigma_gate=0.020):
    """Compute ISE metrics from logged arrays."""
    if len(time_arr) < 2:
        return 0.0, 0.0, 0.0

    dt = np.diff(time_arr)
    T = max(time_arr[-1] - time_arr[0], 1e-6)
    n = min(len(theta_arr), len(u_arr), len(dt) + 1)

    theta = theta_arr[:n]
    u = u_arr[:n]
    dt = dt[:n-1]

    # finite-differenced velocity (matches hardware)
    theta_dot = np.diff(theta) / np.maximum(dt, 1e-6)
    gate = np.exp(-theta[:-1]**2 / (2 * sigma_gate**2))

    ise_theta = float(np.sum(theta[:-1]**2 * dt) / T)
    ise_vel = float(np.sum(theta_dot**2 * gate * dt) / T)
    ise_u = float(np.sum(u[:-1]**2 * dt) / T)

    return ise_theta, ise_vel, ise_u


# ===========================================================================

# EXPERIMENT RUNNER

# ===========================================================================

def _controller_key(ctrl) -> str:
    cls = type(ctrl).__name__
    name = getattr(ctrl, "model_name", "")
    if name and name != cls:
        return f"{cls}_{name}"
    return cls


def _save_json(data: dict, path: str):
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(data, f, indent=2, default=str)
    os.replace(tmp, path)


def _generate_stratified_ics(
    num_trials: int,
    seed: int = 42,
) -> List[np.ndarray]:
    """
    Generate stratified initial conditions matching hardware protocol.

    Uses the same difficulty-stratified cart positions as hardware trials.
    Ball angle is sampled uniformly in [-0.074, 0.074] for each trial
    (in hardware this comes from natural ball placement variance).

    All controllers see the SAME IC sequence (deterministic from seed).
    """
    rng = np.random.default_rng(seed)
    trial_configs = get_hardware_trial_configs(num_trials)

    ics = []
    for cfg_ic in trial_configs:
        cart_pos = cfg_ic["start_position"]
        # Ball angle: sample uniformly, respecting expected_theta_sign if set
        theta_sign = cfg_ic["expected_theta_sign"]
        if theta_sign > 0:
            theta = rng.uniform(0.02, 0.074)
        elif theta_sign < 0:
            theta = rng.uniform(-0.074, -0.02)
        else:
            theta = rng.uniform(-0.074, 0.074)
        ics.append(np.array([cart_pos, 0.0, theta, 0.0]))

    return ics


def run_sim_experiment(
    controllers: list,
    cfg: ExperimentConfig,
    seed: int = 42,
    checkpoint_every: int = 5,
    stratified: bool = True,
    cart_range=DEFAULT_CART_RANGE,
    theta_range=DEFAULT_THETA_RANGE,
) -> dict:
    """
    Run a full simulation experiment (all controllers × all trials).

    Uses cfg.output_dir, cfg.checkpoint_path, cfg.resume from your
    ExperimentConfig - same structure as the hardware runner.

    Args:
        stratified: If True (default), use difficulty-stratified ICs matching
                    the hardware protocol. If False, use uniform random sampling
                    from cart_range × theta_range (legacy behaviour).
    """
    rng = np.random.default_rng(seed)

    # Pre-generate ICs so all controllers see the same sequence
    if stratified:
        ic_list = _generate_stratified_ics(cfg.num_trials, seed=seed)
    else:
        ic_list = [_sample_ic(rng, cart_range, theta_range)
                   for _ in range(cfg.num_trials)]

    # Ensure output directory exists
    os.makedirs(cfg.output_dir, exist_ok=True)

    # Save config alongside results
    import yaml
    with open(cfg.config_path, "w") as f:
        yaml.dump(cfg.to_dict(), f, default_flow_style=False)

    # Load existing checkpoint if resuming
    results: dict = {"experiment": cfg.name, "controllers": {}}
    if cfg.resume and os.path.exists(cfg.checkpoint_path):
        with open(cfg.checkpoint_path, "r") as f:
            results = json.load(f)
        print(f"  Resuming from checkpoint ({cfg.checkpoint_path})")

    for ctrl in controllers:
        key = _controller_key(ctrl)

        if key not in results["controllers"]:
            results["controllers"][key] = []

        existing = len(results["controllers"][key])
        remaining = cfg.num_trials - existing
        if remaining <= 0:
            print(
                f"  {key}: already has {existing}/{cfg.num_trials} trials - skipping")
            continue

        print(
            f"\n  > {key}  ({remaining} trials, action_type={cfg.action_type})")

        for trial_idx in range(existing, cfg.num_trials):
            s0 = ic_list[trial_idx]
            result = run_trial(ctrl, s0, cfg, rng=rng)

            st_str = (f"{result['settling_time']:.2f}s"
                      if result["settling_time"] is not None else "FAIL")
            print(f"    trial {trial_idx+1:3d}/{cfg.num_trials}  "
                  f"IC={np.array2string(s0, precision=3)}  "
                  f"settle={st_str}  overshoot={result['overshoot']:.4f}")

            results["controllers"][key].append(result)

            if (trial_idx + 1) % checkpoint_every == 0:
                _save_json(results, cfg.checkpoint_path)

        _save_json(results, cfg.checkpoint_path)

    _save_json(results, cfg.checkpoint_path)
    print(f"\n  [OK] Results saved to {cfg.checkpoint_path}")
    return results
