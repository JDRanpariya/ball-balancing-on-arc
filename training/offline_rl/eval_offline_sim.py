#!/usr/bin/env python3
"""
Simulation evaluation for offline RL (d3rlpy; Seno & Imai, 2022) continuous
policies.

Runs trained d3rlpy models through the physics simulator and reports
settling time, success rate, and trajectory statistics in the canonical
JSON format (see evaluation/RESULT_FORMAT.md).

Usage:
    # Evaluate a single model
    python eval_offline_sim.py --model-path ./cont_cql_model.d3

    # Evaluate multiple models
    python eval_offline_sim.py \\
        --model-path ./cont_cql_model.d3 ./cont_iql_model.d3

    # Custom number of trials and output dir
    python eval_offline_sim.py \\
        --model-path ./cont_cql_model.d3 \\
        --num-trials 100 --results-dir ./results

    # Compare against classical controllers
    python eval_offline_sim.py \\
        --model-path ./cont_cql_model.d3 --include-classical
"""

import argparse
import json
import logging
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# System constants (self-contained for portability)
# ---------------------------------------------------------------------------
CART_LIMIT = 0.7765
BALL_LIMIT = 0.0810
SETTLING_BAND = 0.01      # rad
SETTLING_DURATION = 1.0   # s
FAIL_TIME = 30.0          # s
DT = 0.05                 # 20 Hz control

# State indices
CART_X, CART_DOT, BALL_X, BALL_DOT = 0, 1, 2, 3

# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)-8s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    stream=sys.stdout,
)
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Lightweight dynamics (no balancer dependency)
# ---------------------------------------------------------------------------

class SimpleDynamics:
    """
    Minimal nonlinear ball-on-arc dynamics for evaluation.
    Matches balancer.core.dynamics.NonLinearDynamics defaults.
    """

    def __init__(self, dt: float = 0.05):
        self.dt = dt
        self.gravity = 9.81
        self.arc_radius = 2.101
        self.friction_cart = 0.005
        self.mu_ball_rolling = 0.001
        self.mu_ball_viscous = 0.08
        self.tau_v = 0.15  # velocity lag
        self.max_cart_vel = 0.9

    def step(self, state: np.ndarray, action: float) -> np.ndarray:
        """
        RK4 integration step.

        state: [cart_pos, cart_vel, ball_angle, ball_ang_vel]
        action: continuous in [-1, 1] -> maps to target cart velocity
        """
        x, x_dot, theta, theta_dot = state

        # Action -> target velocity
        v_target = float(np.clip(action, -1.0, 1.0)) * self.max_cart_vel

        # First-order velocity lag (motor model)
        x_dot_new = x_dot + (v_target - x_dot) * (self.dt / self.tau_v)
        x_dot_new = np.clip(x_dot_new, -self.max_cart_vel, self.max_cart_vel)

        # Cart position update
        x_new = x + x_dot_new * self.dt

        # Ball dynamics: theta_ddot = (g/R)*sin(theta) - (a_cart/R)*cos(theta) - friction
        # Cart acceleration
        a_cart = (x_dot_new - x_dot) / self.dt

        def _ball_deriv(th, th_dot):
            gravity_term = (self.gravity / self.arc_radius) * np.sin(th)
            cart_term = -(a_cart / self.arc_radius) * np.cos(th)
            friction = (self.mu_ball_rolling * np.sign(th_dot)
                        + self.mu_ball_viscous * th_dot)
            return gravity_term + cart_term - friction

        # RK4 for ball
        k1_v = _ball_deriv(theta, theta_dot)
        k1_p = theta_dot

        k2_v = _ball_deriv(theta + 0.5*self.dt*k1_p, theta_dot + 0.5*self.dt*k1_v)
        k2_p = theta_dot + 0.5*self.dt*k1_v

        k3_v = _ball_deriv(theta + 0.5*self.dt*k2_p, theta_dot + 0.5*self.dt*k2_v)
        k3_p = theta_dot + 0.5*self.dt*k2_v

        k4_v = _ball_deriv(theta + self.dt*k3_p, theta_dot + self.dt*k3_v)
        k4_p = theta_dot + self.dt*k3_v

        theta_new = theta + (self.dt/6.0)*(k1_p + 2*k2_p + 2*k3_p + k4_p)
        theta_dot_new = theta_dot + (self.dt/6.0)*(k1_v + 2*k2_v + 2*k3_v + k4_v)

        return np.array([x_new, x_dot_new, theta_new, theta_dot_new], dtype=np.float32)


# ---------------------------------------------------------------------------
# Settling time detection
# ---------------------------------------------------------------------------

def compute_settling_time(
    theta_hist: np.ndarray,
    time_hist: np.ndarray,
    band: float = SETTLING_BAND,
    duration: float = SETTLING_DURATION,
) -> Optional[float]:
    """Return first time |θ| stays within band for duration seconds."""
    in_band = np.abs(theta_hist) < band
    enter_time = None
    for i in range(len(theta_hist)):
        if in_band[i]:
            if enter_time is None:
                enter_time = time_hist[i]
            if time_hist[i] - enter_time >= duration:
                return float(enter_time)
        else:
            enter_time = None
    return None


# ---------------------------------------------------------------------------
# d3rlpy model wrapper
# ---------------------------------------------------------------------------

class D3RLPyPolicy:
    """Wraps a d3rlpy model for step-by-step inference."""

    def __init__(self, model_path: str, device: str = "cpu"):
        import d3rlpy
        self.model = d3rlpy.load_learnable(model_path, device=device)
        self.name = Path(model_path).stem

    def predict(self, state: np.ndarray) -> float:
        obs = state.reshape(1, -1).astype(np.float32)
        action = self.model.predict(obs)
        return float(np.clip(action.flat[0], -1.0, 1.0))


# ---------------------------------------------------------------------------
# Single trial runner
# ---------------------------------------------------------------------------

def run_trial(
    policy: D3RLPyPolicy,
    dyn: SimpleDynamics,
    init_state: np.ndarray,
    max_steps: int,
) -> dict:
    """Run one trial. Returns dict with trajectory and metrics."""
    state = init_state.copy()
    n_steps = int(FAIL_TIME / DT)
    if max_steps:
        n_steps = min(n_steps, max_steps)

    states = [state.copy()]
    actions = []
    time_hist = [0.0]

    for step in range(n_steps):
        action = policy.predict(state)
        state = dyn.step(state, action)
        states.append(state.copy())
        actions.append(action)
        time_hist.append((step + 1) * DT)

        # Check if ball fell off
        if abs(state[BALL_X]) > BALL_LIMIT:
            break
        # Check if cart hit wall
        if abs(state[CART_X]) > CART_LIMIT:
            state[CART_X] = np.clip(state[CART_X], -CART_LIMIT, CART_LIMIT)
            state[CART_DOT] = 0.0

    states = np.array(states)
    time_hist = np.array(time_hist)
    theta_hist = states[:, BALL_X]

    settling = compute_settling_time(theta_hist, time_hist)
    settled = settling is not None
    duration = time_hist[-1]
    ball_fell = abs(states[-1, BALL_X]) > BALL_LIMIT

    return {
        "settling_time": settling if settled else FAIL_TIME,
        "settled": settled,
        "ball_fell": ball_fell,
        "duration": duration,
        "theta_rmse": float(np.sqrt(np.mean(theta_hist**2))),
        "theta_max": float(np.max(np.abs(theta_hist))),
        "actions_mean": float(np.mean(np.abs(actions))) if actions else 0.0,
        "states": states.tolist(),
        "actions": actions,
    }


# ---------------------------------------------------------------------------
# Full evaluation
# ---------------------------------------------------------------------------

def evaluate_model(
    model_path: str,
    num_trials: int = 50,
    seed: int = 42,
    device: str = "cpu",
    save_trajectories: bool = False,
) -> dict:
    """Evaluate a single d3rlpy model across multiple trials."""
    rng = np.random.default_rng(seed)
    policy = D3RLPyPolicy(model_path, device=device)
    dyn = SimpleDynamics(dt=DT)

    log.info("Evaluating %s (%d trials)", policy.name, num_trials)

    # Generate initial conditions (same distribution as eval_sim.py)
    cart_margin = 0.9
    ball_margin = 0.77
    results = []

    for trial in range(num_trials):
        cart_pos = rng.uniform(-CART_LIMIT * cart_margin, CART_LIMIT * cart_margin)
        ball_pos = rng.uniform(-BALL_LIMIT * ball_margin, BALL_LIMIT * ball_margin)
        init_state = np.array([cart_pos, 0.0, ball_pos, 0.0], dtype=np.float32)

        trial_result = run_trial(policy, dyn, init_state, max_steps=int(FAIL_TIME / DT))

        if not save_trajectories:
            del trial_result["states"]
            del trial_result["actions"]

        trial_result["trial"] = trial
        trial_result["init_cart_pos"] = float(cart_pos)
        trial_result["init_ball_pos"] = float(ball_pos)
        results.append(trial_result)

    # Aggregate metrics
    settling_times = [r["settling_time"] for r in results]
    settled_mask = [r["settled"] for r in results]
    fell_mask = [r["ball_fell"] for r in results]

    summary = {
        "controller": policy.name,
        "model_path": str(model_path),
        "num_trials": num_trials,
        "success_rate": float(np.mean(settled_mask)),
        "fall_rate": float(np.mean(fell_mask)),
        "settling_time_mean": float(np.mean(settling_times)),
        "settling_time_median": float(np.median(settling_times)),
        "settling_time_std": float(np.std(settling_times)),
        "settling_time_settled_mean": float(
            np.mean([s for s, ok in zip(settling_times, settled_mask) if ok])
        ) if any(settled_mask) else FAIL_TIME,
        "theta_rmse_mean": float(np.mean([r["theta_rmse"] for r in results])),
        "action_effort_mean": float(np.mean([r["actions_mean"] for r in results])),
        "seed": seed,
        "dt": DT,
        "episodes": results,
    }

    log.info("  Success rate: %.1f%%  |  Settling (settled): %.2fs  |  Fall rate: %.1f%%",
             summary["success_rate"] * 100,
             summary["settling_time_settled_mean"],
             summary["fall_rate"] * 100)

    return summary


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--model-path", type=str, nargs="+", required=True,
                        help="Path(s) to d3rlpy .d3 model files")
    parser.add_argument("--num-trials", type=int, default=50)
    parser.add_argument("--results-dir", type=str, default="./results")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cpu",
                        choices=["cpu", "cuda"])
    parser.add_argument("--save-trajectories", action="store_true",
                        help="Include full state/action trajectories in output JSON")
    parser.add_argument("--include-classical", action="store_true",
                        help="Also evaluate PID/LQR for comparison (requires balancer)")

    args = parser.parse_args()
    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    all_results = []

    for model_path in args.model_path:
        if not Path(model_path).exists():
            log.error("Model not found: %s", model_path)
            continue

        result = evaluate_model(
            model_path=model_path,
            num_trials=args.num_trials,
            seed=args.seed,
            device=args.device,
            save_trajectories=args.save_trajectories,
        )
        all_results.append(result)

        # Save individual result
        out_path = results_dir / f"{result['controller']}_eval.json"
        with open(out_path, "w") as f:
            json.dump(result, f, indent=2, default=str)
        log.info("  Saved -> %s", out_path)

    # Summary table
    if all_results:
        print(f"\n{'='*70}")
        print(f"{'Controller':<25s} {'Success%':>8s} {'Settle(s)':>10s} {'Fall%':>7s} {'θ_RMSE':>8s}")
        print(f"{'-'*70}")
        for r in all_results:
            print(f"{r['controller']:<25s} "
                  f"{r['success_rate']*100:>7.1f}% "
                  f"{r['settling_time_settled_mean']:>9.2f}s "
                  f"{r['fall_rate']*100:>6.1f}% "
                  f"{r['theta_rmse_mean']:>8.4f}")
        print(f"{'='*70}")


if __name__ == "__main__":
    main()
