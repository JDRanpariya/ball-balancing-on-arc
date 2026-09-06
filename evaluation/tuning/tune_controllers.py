"""
Unified controller tuning using CMA-ES.

Gain optimization uses the CMA-ES evolution strategy via the pycma
implementation (Hansen et al.).

Supports three tuning modes via --action-type:

    cont     - tune against smooth velocity plant (best for CMA-ES)

Output structure:
    tuning/
    +-- pid/
    |   +-- best_params.json
    |   +-- convergence.png
    |   +-- pid_ic0.png ... pid_icN.png     (primary mode plots)
    |   +-- summary.json
    +-- lqr/
    |   +-- ...
    +-- comparison/
        +-- ...
"""

from simulate import (
    simulate_controller,
    simulate_all_ics,
    get_tuning_initial_conditions,
    compute_metrics,
    compute_metrics_ise,
    plot_comparison,
    plot_batch_results,
    CART_LIMIT,
    FAIL_TIME,
)
from balancer.core.linear_dynamics import get_discrete_system_velocity
from balancer.core.dynamics import DEFAULT_PARAMS
from scipy.linalg import solve_discrete_are
from cma import CMAEvolutionStrategy
from joblib import Parallel, delayed
import multiprocessing
import matplotlib.pyplot as plt
import os
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")


# ============================================================================

# CONSTANTS

# ============================================================================

TUNING_ROOT = "tuning"

TUNING_SIM_T = 30.0   # full 30s sim for accurate tuning
EVAL_SIM_T = FAIL_TIME  # 30s for final validation

# ============================================================================

# HELPERS

# ============================================================================


def _ensure_dir(path):
    os.makedirs(path, exist_ok=True)
    return path


def _save_params(ctrl_name, param_dict, save_dir=None):
    if save_dir is None:
        save_dir = os.path.join(TUNING_ROOT, ctrl_name)
    _ensure_dir(save_dir)
    path = os.path.join(save_dir, "best_params.json")
    with open(path, "w") as f:
        json.dump(param_dict, f, indent=2)
    print(f"  Parameters saved to {path}")
    return path


def _save_convergence(ctrl_name, cost_history, save_dir=None):
    if save_dir is None:
        save_dir = os.path.join(TUNING_ROOT, ctrl_name)
    _ensure_dir(save_dir)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.semilogy(cost_history, linewidth=1.5)
    ax.set_xlabel("Generation")
    ax.set_ylabel("Best Cost (log)")
    ax.set_title(f"{ctrl_name.upper()} - CMA-ES Convergence")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    path = os.path.join(save_dir, "convergence.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Convergence plot saved to {path}")


def _save_best_sim(ctrl_name, controller, Ts=0.05, save_dir=None,
                   action_type="cont"):
    """Simulate best controller on all ICs and save plots + summary.

    Always produces results for the requested action_type.
    If action_type is "cont", also produces discrete validation plots.
    If action_type is "discrete", also produces cont validation plots.
    If action_type is "both", produces both sets.
    """
    if save_dir is None:
        save_dir = os.path.join(TUNING_ROOT, ctrl_name)
    _ensure_dir(save_dir)

    results = simulate_all_ics(controller, Ts=Ts, sim_T=EVAL_SIM_T,
                               action_type="cont")
    plot_batch_results(results, ctrl_name, save_dir=save_dir)

    # Summary uses primary mode(s)
    summary = []
    for res in results:
        # compute ISE alongside old metrics
        theta_dot = np.diff(res["theta"]) / np.diff(res["time"])
        ise_m = compute_metrics_ise(
            res["time"], res["theta"], res["u"], theta_dot
        )
        entry = {"ic_index": res["ic_index"], "s0": res["s0"].tolist()}
        entry.update(res["metrics"])      # old metrics (settling_time etc.)
        entry.update({                    # new ISE metrics
            "ise_theta": ise_m["ise_theta"],
            "ise_vel":   ise_m["ise_vel"],
            "ise_u":     ise_m["ise_u"],
        })
        summary.append(entry)

    avg = {
        "avg_settling_time": float(np.mean(
            [r["metrics"]["settling_time"] for r in results])),
        "avg_ise_theta": float(np.mean(
            [s["ise_theta"] for s in summary])),
        "avg_ise_vel":   float(np.mean(
            [s["ise_vel"] for s in summary])),
        "avg_ise_u":     float(np.mean(
            [s["ise_u"] for s in summary])),
        # keep old fields for comparison plots
        "avg_overshoot": float(np.mean(
            [r["metrics"]["overshoot"] for r in results])),
        "avg_energy": float(np.mean(
            [r["metrics"]["energy"] for r in results])),
    }

    out = {
        "controller": ctrl_name,
        "tuned_on": action_type,
        "averages": avg,
        "per_ic": summary,
    }
    path = os.path.join(save_dir, "summary.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"  Summary saved to {path}")

    return avg


def _run_cmaes(objective, x0, sigma0, opts, ctrl_name, n_jobs=None):
    """
    CMA-ES with parallel population evaluation.
    n_jobs: number of parallel workers (default: all physical cores).
    """
    if n_jobs is None:
        n_jobs = multiprocessing.cpu_count()

    es = CMAEvolutionStrategy(x0, sigma0, opts)
    cost_history = []

    while not es.stop():
        solutions = es.ask()

        # Each candidate is independent - evaluate in parallel
        fitnesses = Parallel(
            n_jobs=n_jobs,
            backend="loky",        # cloudpickle: handles closures
            prefer="processes",    # CPU-bound work
        )(delayed(objective)(s) for s in solutions)

        es.tell(solutions, fitnesses)
        es.disp()
        cost_history.append(es.result.fbest)

    return es.result.xbest, cost_history


# ============================================================================

# OBJECTIVE FUNCTION

# ============================================================================
def robust_objective(create_controller_fn, param_bounds_check,
                     Ts=0.05, action_type="cont"):
    """
    ISE-based objective. No settling_time in the cost.

    Scale intuition (all terms should be same order of magnitude):
        ise_theta  : θ ∈ [0, 0.078] -> mean θ² ~ 0.001 -> ISE ~ 0.001
                     × 500 -> contribution ~ 0-0.5
        ise_vel    : θ̇ ∈ [0, 0.5] near center -> mean θ̇²·gate ~ 0.01
                     × 20  -> contribution ~ 0-0.2
        ise_u      : u ∈ [-1, 1] -> mean u² ~ 0.2
                     × 0.5 -> contribution ~ 0-0.1
        boundary_v : 0-20 hits × 0.1 -> 0-2.0
    """
    s0_list = get_tuning_initial_conditions()
    eval_modes = ["cont"]

    def _eval_one_mode(controller, mode):
        total = 0.0
        for s0 in s0_list:
            time, theta, u = simulate_controller(
                controller, s0, Ts=Ts, sim_T=TUNING_SIM_T, action_type="cont"
            )
            if len(theta) < 5 or np.any(np.isnan(theta)):
                total += 1e3
                continue

            # Finite-differenced velocity - matches hardware exactly
            theta_dot = np.diff(theta) / np.diff(time)

            m = compute_metrics_ise(time, theta, u, theta_dot)

            cost = (
                500.0 * m["ise_theta"]          # dominant: drive to zero
                + 20.0 * m["ise_vel"]           # damp velocity near dip
                + 0.5 * m["ise_u"]             # effort (secondary)
                + 0.1 * m["boundary_violations"]  # rail hits

            )
            total += min(cost, 1e3)
        return total / len(s0_list)

    def objective(x):
        if not param_bounds_check(x):
            return 1e5
        try:
            controller = create_controller_fn(x)
        except Exception:
            return 1e5
        costs = []
        for mode in eval_modes:
            controller.reset()
            costs.append(_eval_one_mode(controller, mode))
        return float(np.mean(costs))
    return objective


# ============================================================================

# PID TUNING

# ============================================================================

def tune_pid(Ts=0.05, action_type="cont", n_jobs=None):
    from balancer.controllers.pid import PIDController

    ctrl_name = "pid"
    save_dir = os.path.join(TUNING_ROOT, ctrl_name)

    def create_pid(x):
        Kp, Ki, Kd = x
        pid = PIDController(Kp=Kp, Ki=Ki, Kd=Kd, Ts=Ts,
                            action_type=action_type)
        return pid

    def bounds_check(x):
        Kp, Ki, Kd = x
        return 0 < Kp <= 25 and 0 <= Ki <= 10 and 0 <= Kd <= 10

    objective = robust_objective(create_pid, bounds_check,
                                 Ts=Ts, action_type=action_type)

    x0 = [19.0, 0.0, 1.3]
    sigma0 = 2.0
    opts = {
        "bounds": [[0.0, 0.0, 0.0], [25.0, 10.0, 10.0]],
        "maxfevals": 1500,
        "popsize": 40,
        "verb_disp": 1,
    }

    print("\n" + "=" * 70)
    print(f"TUNING PID CONTROLLER (plant: {action_type})")
    print("=" * 70)

    best_x, cost_history = _run_cmaes(
        objective, x0, sigma0, opts, ctrl_name, n_jobs=n_jobs)

    param_dict = {
        "Kp": float(best_x[0]),
        "Ki": float(best_x[1]),
        "Kd": float(best_x[2]),
        "wall_margin": 0.10,
        "override_gain": 0.5,
        "Ts": Ts,
        "tuned_on": action_type,
    }
    print(f"\n[OK] Best PID: Kp={best_x[0]:.3f}, Ki={best_x[1]:.3f}, "
          f"Kd={best_x[2]:.3f}")

    _save_params(ctrl_name, param_dict, save_dir)
    _save_convergence(ctrl_name, cost_history, save_dir)

    best_ctrl = create_pid(best_x)
    _save_best_sim(ctrl_name, best_ctrl, Ts=Ts, save_dir=save_dir,
                   action_type=action_type)

    return best_x, param_dict


def tune_smc_full(Ts=0.05, action_type="cont", n_jobs=None):
    """Tune SMC full (with centering + barrier) - lam_x as tuning param."""
    from balancer.controllers.smc import SMCController

    ctrl_name = "smc_full"
    save_dir = os.path.join(TUNING_ROOT, ctrl_name)

    def create_smc(x):
        lam, k, phi, lam_x = x
        return SMCController(lam=lam, k=k, phi=phi,
                             action_type="cont", lam_x=lam_x)

    def bounds_check(x):
        lam, k, phi, lam_x = x
        return (0.1 <= lam <= 25
                and 0.1 <= k <= 25
                and 0.01 <= phi <= 2.0
                and 0.0 <= lam_x <= 2.0)

    objective = robust_objective(create_smc, bounds_check,
                                 Ts=Ts, action_type=action_type)

    x0 = [8.0, 15.0, 0.2, 0.3]
    sigma0 = 0.5
    opts = {
        "bounds": [[0.1, 0.1, 0.01, 0.0], [25, 25, 2.0, 2.0]],
        "maxfevals": 1200,
        "popsize": 36,
        "verb_disp": 1,
    }

    print("\n" + "=" * 70)
    print(f"TUNING SMC_FULL (λ, k, φ, lam_x) - plant: {action_type}")
    print("=" * 70)

    best_x, cost_history = _run_cmaes(
        objective, x0, sigma0, opts, ctrl_name, n_jobs=n_jobs)

    param_dict = {
        "lam": float(best_x[0]),
        "k":   float(best_x[1]),
        "phi": float(best_x[2]),
        "lam_x": float(best_x[3]),
        "Ts":  Ts,
        "variant": "full",
        "tuned_on": action_type,
    }
    print(f"\n[OK] Best SMC_full: λ={best_x[0]:.3f}, k={best_x[1]:.3f}, "
          f"φ={best_x[2]:.4f}, lam_x={best_x[3]:.4f}")

    _save_params(ctrl_name, param_dict, save_dir)
    _save_convergence(ctrl_name, cost_history, save_dir)
    best_ctrl = create_smc(best_x)
    _save_best_sim(ctrl_name, best_ctrl, Ts=Ts, save_dir=save_dir,
                   action_type=action_type)

    return best_x, param_dict


# ============================================================================

# LQR TUNING

# ============================================================================

def tune_lqr(Ts=0.05, tau_v=0.15, action_type="cont", n_jobs=None):
    from balancer.controllers.lqr import LQRController

    ctrl_name = "lqr"
    save_dir = os.path.join(TUNING_ROOT, ctrl_name)

    Ad_v, Bd_v = get_discrete_system_velocity(DEFAULT_PARAMS, Ts, tau_v=tau_v)

    def create_lqr(x):
        q1, q2, q3, q4, r1 = x
        Q = np.diag([q1, q2, q3, q4])
        R = np.array([[r1]])
        P = solve_discrete_are(Ad_v, Bd_v, Q, R)
        K = np.linalg.inv(Bd_v.T @ P @ Bd_v + R) @ (Bd_v.T @ P @ Ad_v)
        return LQRController(K=K.flatten(), action_type="cont")

    def bounds_check(x):
        q1, q2, q3, q4, r1 = x
        if not (all(qi >= 1e-3 for qi in [q1, q2, q3, q4]) and r1 > 1e-5):
            return False
        try:
            Q = np.diag([q1, q2, q3, q4])
            R = np.array([[r1]])
            solve_discrete_are(Ad_v, Bd_v, Q, R)
            return True
        except np.linalg.LinAlgError:
            return False

    objective = robust_objective(create_lqr, bounds_check,
                                 Ts=Ts, action_type=action_type)

    x0 = [0.1, 0.1, 20.0, 0.1, 0.05]
    sigma0 = 0.5
    opts = {
        "bounds": [[1e-3, 1e-3, 1e-3, 1e-3, 1e-4], [10, 5, 100, 5, 1]],
        "maxfevals": 980,
        "popsize": 32,
        "verb_disp": 1,
    }

    print("\n" + "=" * 70)
    print(f"TUNING LQR CONTROLLER (velocity plant, {action_type})")
    print("=" * 70)

    best_x, cost_history = _run_cmaes(
        objective, x0, sigma0, opts, ctrl_name, n_jobs=n_jobs)

    Q_best = np.diag(best_x[:4])
    R_best = np.array([[best_x[4]]])
    P = solve_discrete_are(Ad_v, Bd_v, Q_best, R_best)
    K_best = np.linalg.inv(Bd_v.T @ P @ Bd_v + R_best) @ (Bd_v.T @ P @ Ad_v)
    K_best = K_best.flatten()

    param_dict = {
        "Q_diag": [float(v) for v in best_x[:4]],
        "R": float(best_x[4]),
        "K": [float(v) for v in K_best],
        "Ts": Ts,
        "tau_v": tau_v,
        "plant": "velocity",
        "tuned_on": action_type,
    }
    print(f"\n[OK] Best LQR: Q=diag({param_dict['Q_diag']}), "
          f"R={param_dict['R']:.4f}")
    print(f"   K = {K_best}")

    _save_params(ctrl_name, param_dict, save_dir)
    _save_convergence(ctrl_name, cost_history, save_dir)
    best_ctrl = create_lqr(best_x)
    _save_best_sim(ctrl_name, best_ctrl, Ts=Ts, save_dir=save_dir,
                   action_type=action_type)

    return best_x, param_dict


# ============================================================================

# LQR VARIANT TUNING

# ============================================================================

def tune_lqr_basic(Ts=0.05, tau_v=0.15, action_type="cont", n_jobs=None):
    """LQR variant 1: ball-focused (Q[0,0] <= 0.01)."""
    from balancer.controllers.lqr import LQRController

    ctrl_name = "lqr_basic"
    save_dir = os.path.join(TUNING_ROOT, ctrl_name)

    Ad_v, Bd_v = get_discrete_system_velocity(DEFAULT_PARAMS, Ts, tau_v=tau_v)

    def create_lqr(x):
        q1, q2, q3, q4, r1 = x
        Q = np.diag([q1, q2, q3, q4])
        R = np.array([[r1]])
        P = solve_discrete_are(Ad_v, Bd_v, Q, R)
        K = np.linalg.inv(Bd_v.T @ P @ Bd_v + R) @ (Bd_v.T @ P @ Ad_v)
        return LQRController(K=K.flatten(), action_type="cont")

    def bounds_check(x):
        q1, q2, q3, q4, r1 = x
        if not (all(qi >= 1e-3 for qi in [q1, q2, q3, q4]) and r1 > 1e-5):
            return False
        if q1 > 0.01:  # enforce ball-focused: minimal cart weight
            return False
        try:
            Q = np.diag([q1, q2, q3, q4])
            R = np.array([[r1]])
            solve_discrete_are(Ad_v, Bd_v, Q, R)
            return True
        except np.linalg.LinAlgError:
            return False

    objective = robust_objective(create_lqr, bounds_check,
                                 Ts=Ts, action_type=action_type)

    x0 = [0.001, 0.02, 19.0, 1.8, 0.07]
    sigma0 = 0.3
    opts = {
        "bounds": [[1e-3, 1e-3, 1e-3, 1e-3, 1e-4], [0.01, 5, 100, 5, 1]],
        "maxfevals": 980,
        "popsize": 32,
        "verb_disp": 1,
    }

    print("\n" + "=" * 70)
    print(f"TUNING LQR_BASIC (ball-focused, Q[0,0]<=0.01, {action_type})")
    print("=" * 70)

    best_x, cost_history = _run_cmaes(
        objective, x0, sigma0, opts, ctrl_name, n_jobs=n_jobs)

    Q_best = np.diag(best_x[:4])
    R_best = np.array([[best_x[4]]])
    P = solve_discrete_are(Ad_v, Bd_v, Q_best, R_best)
    K_best = np.linalg.inv(Bd_v.T @ P @ Bd_v + R_best) @ (Bd_v.T @ P @ Ad_v)
    K_best = K_best.flatten()

    param_dict = {
        "Q_diag": [float(v) for v in best_x[:4]],
        "R": float(best_x[4]),
        "K": [float(v) for v in K_best],
        "Ts": Ts,
        "tau_v": tau_v,
        "plant": "velocity",
        "variant": "basic",
        "tuned_on": action_type,
    }
    print(f"\n[OK] Best LQR_basic: Q=diag({param_dict['Q_diag']}), "
          f"R={param_dict['R']:.4f}")
    print(f"   K = {K_best}")

    _save_params(ctrl_name, param_dict, save_dir)
    _save_convergence(ctrl_name, cost_history, save_dir)
    best_ctrl = create_lqr(best_x)
    _save_best_sim(ctrl_name, best_ctrl, Ts=Ts, save_dir=save_dir,
                   action_type=action_type)

    return best_x, param_dict


def tune_lqr_qcenter(Ts=0.05, tau_v=0.15, action_type="cont", n_jobs=None):
    """LQR variant 2: Q-centering (Q[0,0] free, higher BV penalty)."""
    from balancer.controllers.lqr import LQRController

    ctrl_name = "lqr_qcenter"
    save_dir = os.path.join(TUNING_ROOT, ctrl_name)

    Ad_v, Bd_v = get_discrete_system_velocity(DEFAULT_PARAMS, Ts, tau_v=tau_v)

    def create_lqr(x):
        q1, q2, q3, q4, r1 = x
        Q = np.diag([q1, q2, q3, q4])
        R = np.array([[r1]])
        P = solve_discrete_are(Ad_v, Bd_v, Q, R)
        K = np.linalg.inv(Bd_v.T @ P @ Bd_v + R) @ (Bd_v.T @ P @ Ad_v)
        return LQRController(K=K.flatten(), action_type="cont")

    def bounds_check(x):
        q1, q2, q3, q4, r1 = x
        if not (all(qi >= 1e-3 for qi in [q1, q2, q3, q4]) and r1 > 1e-5):
            return False
        try:
            Q = np.diag([q1, q2, q3, q4])
            R = np.array([[r1]])
            solve_discrete_are(Ad_v, Bd_v, Q, R)
            return True
        except np.linalg.LinAlgError:
            return False

    objective = robust_objective(create_lqr, bounds_check,
                                 Ts=Ts, action_type=action_type)

    x0 = [0.05, 0.01, 30.0, 0.5, 0.05]
    sigma0 = 0.3
    opts = {
        "bounds": [[1e-3, 1e-3, 5.0, 1e-3, 0.01], [1.0, 2.0, 50, 5, 0.5]],
        "maxfevals": 1500,
        "popsize": 40,
        "verb_disp": 1,
    }

    print("\n" + "=" * 70)
    print(f"TUNING LQR_QCENTER (free Q[0,0], high BV penalty, {action_type})")
    print("=" * 70)

    best_x, cost_history = _run_cmaes(
        objective, x0, sigma0, opts, ctrl_name, n_jobs=n_jobs)

    Q_best = np.diag(best_x[:4])
    R_best = np.array([[best_x[4]]])
    P = solve_discrete_are(Ad_v, Bd_v, Q_best, R_best)
    K_best = np.linalg.inv(Bd_v.T @ P @ Bd_v + R_best) @ (Bd_v.T @ P @ Ad_v)
    K_best = K_best.flatten()

    param_dict = {
        "Q_diag": [float(v) for v in best_x[:4]],
        "R": float(best_x[4]),
        "K": [float(v) for v in K_best],
        "Ts": Ts,
        "tau_v": tau_v,
        "plant": "velocity",
        "variant": "qcenter",
        "tuned_on": action_type,
    }
    print(f"\n[OK] Best LQR_qcenter: Q=diag({param_dict['Q_diag']}), "
          f"R={param_dict['R']:.4f}")
    print(f"   K = {K_best}")

    _save_params(ctrl_name, param_dict, save_dir)
    _save_convergence(ctrl_name, cost_history, save_dir)
    best_ctrl = create_lqr(best_x)
    _save_best_sim(ctrl_name, best_ctrl, Ts=Ts, save_dir=save_dir,
                   action_type=action_type)

    return best_x, param_dict


# ============================================================================

# MPC TUNING

# ============================================================================

def tune_mpc(Ts=0.05, tau_v=0.15, action_type="cont", n_jobs=None):
    from balancer.controllers.mpc import MPCController

    ctrl_name = "mpc"
    save_dir = os.path.join(TUNING_ROOT, ctrl_name)

    def create_mpc(x):
        q1, q2, q3, q4, r1, N_float = x
        Q = np.diag([q1, q2, q3, q4])
        R = np.array([[r1]])
        N = max(5, int(round(N_float)))
        return MPCController(Q=Q, R=R, Ts=Ts, N=N, tau_v=tau_v,
                             action_type="cont")

    def bounds_check(x):
        q1, q2, q3, q4, r1, N_float = x
        return all(qi >= 0 for qi in [q1, q2, q3, q4]) and r1 > 1e-5 and 5 <= N_float <= 60

    objective = robust_objective(create_mpc, bounds_check,
                                 Ts=Ts, action_type=action_type)

    x0 = [1e-3, 1e-3, 20.0, 1e-3, 0.2, 30]
    sigma0 = 0.5
    opts = {
        "bounds": [[1e-4, 1e-4, 1e-4, 1e-4, 1e-4, 5], [10, 10, 200, 10, 2.0, 60]],
        "maxfevals": 980,
        "popsize": 32,
        "verb_disp": 1,
    }

    print("\n" + "=" * 70)
    print(f"TUNING MPC CONTROLLER (velocity plant, {action_type})")
    print("=" * 70)

    best_x, cost_history = _run_cmaes(
        objective, x0, sigma0, opts, ctrl_name, n_jobs=n_jobs)
    best_N = max(5, int(round(best_x[5])))

    param_dict = {
        "Q_diag": [float(v) for v in best_x[:4]],
        "R": float(best_x[4]),
        "N":      best_N,
        "Ts": Ts,
        "tau_v": tau_v,
        "plant": "velocity",
        "tuned_on": action_type,
    }
    print(f"\n[OK] Best MPC: Q=diag({param_dict['Q_diag']}), "
          f"R={param_dict['R']:.4f}")

    _save_params(ctrl_name, param_dict, save_dir)
    _save_convergence(ctrl_name, cost_history, save_dir)
    best_ctrl = create_mpc(best_x)
    _save_best_sim(ctrl_name, best_ctrl, Ts=Ts, save_dir=save_dir,
                   action_type=action_type)

    return best_x, param_dict


# ============================================================================

# MPC_WO TUNING (no QP cart constraint, wall override)

# ============================================================================

def tune_mpc_wo(Ts=0.05, tau_v=0.15, action_type="cont", n_jobs=None):
    """Tune MPC with wall override and no QP cart constraint.

    Tunes Q, R, N for MPCController(cart_constraint=False, wall_margin=0.12).
    This matches the hardware-best configuration (80% SR).
    """
    from balancer.controllers.mpc import MPCController

    ctrl_name = "mpc_wo"
    save_dir = os.path.join(TUNING_ROOT, ctrl_name)

    def create_mpc_wo(x):
        q1, q2, q3, q4, r1, N_float = x
        Q = np.diag([q1, q2, q3, q4])
        R = np.array([[r1]])
        N = max(5, int(round(N_float)))
        return MPCController(Q=Q, R=R, Ts=Ts, N=N, tau_v=tau_v,
                             action_type="cont",
                             cart_constraint=False,
                             wall_margin=0.12,
                             override_gain=0.5)

    def bounds_check(x):
        q1, q2, q3, q4, r1, N_float = x
        return all(qi >= 0 for qi in [q1, q2, q3, q4]) and r1 > 1e-5 and 5 <= N_float <= 60

    objective = robust_objective(create_mpc_wo, bounds_check,
                                 Ts=Ts, action_type=action_type)

    x0 = [0.001, 0.001, 10.0, 0.001, 0.01, 10]
    sigma0 = 0.5
    opts = {
        "bounds": [[1e-4, 1e-4, 1e-4, 1e-4, 1e-4, 5], [10, 10, 200, 10, 2.0, 60]],
        "maxfevals": 580,
        "popsize": 16,
        "verb_disp": 1,
    }

    print("\n" + "=" * 70)
    print(f"TUNING MPC+WO CONTROLLER (no cart QP constraint, wall override)")
    print("=" * 70)

    best_x, cost_history = _run_cmaes(
        objective, x0, sigma0, opts, ctrl_name, n_jobs=n_jobs)
    best_N = max(5, int(round(best_x[5])))

    param_dict = {
        "Q_diag": [float(v) for v in best_x[:4]],
        "R": float(best_x[4]),
        "N": best_N,
        "Ts": Ts,
        "tau_v": tau_v,
        "cart_constraint": False,
        "wall_margin": 0.12,
        "override_gain": 0.5,
        "plant": "velocity",
        "tuned_on": action_type,
    }
    print(f"\n\u2705 Best MPC+WO: Q=diag({param_dict['Q_diag']}), "
          f"R={param_dict['R']:.4f}, N={best_N}")

    _save_params(ctrl_name, param_dict, save_dir)
    _save_convergence(ctrl_name, cost_history, save_dir)
    best_ctrl = create_mpc_wo(best_x)
    _save_best_sim(ctrl_name, best_ctrl, Ts=Ts, save_dir=save_dir,
                   action_type=action_type)

    return best_x, param_dict


# ============================================================================

# SMC TUNING

# ============================================================================

def tune_smc(Ts=0.05, action_type="cont", n_jobs=None):
    """Tune SMC basic (no centering, no barrier) - ball stabilization only."""
    from balancer.controllers.smc import SMCController

    ctrl_name = "smc"
    save_dir = os.path.join(TUNING_ROOT, ctrl_name)

    def create_smc(x):
        lam, k, phi = x
        return SMCController(lam=lam, k=k, phi=phi,
                             action_type="cont", lam_x=0.0)

    def bounds_check(x):
        lam, k, phi = x
        return (0.1 <= lam <= 25
                and 0.1 <= k <= 25
                and 0.01 <= phi <= 2.0)

    objective = robust_objective(create_smc, bounds_check,
                                 Ts=Ts, action_type=action_type)

    x0 = [8.0, 15.0, 0.2]
    sigma0 = 0.5
    opts = {
        "bounds": [[0.1, 0.1, 0.01], [25, 25, 2.0]],
        "maxfevals": 980,
        "popsize": 32,
        "verb_disp": 1,
    }

    print("\n" + "=" * 70)
    print(f"TUNING SMC CONTROLLER (λ, k, φ) - plant: {action_type}")
    print("=" * 70)

    best_x, cost_history = _run_cmaes(
        objective, x0, sigma0, opts, ctrl_name, n_jobs=n_jobs)

    param_dict = {
        "lam": float(best_x[0]),
        "k":   float(best_x[1]),
        "phi": float(best_x[2]),
        "lam_x": 0.0,
        "Ts":  Ts,
        "variant": "basic",
        "tuned_on": action_type,
    }
    print(f"\n[OK] Best SMC: λ={best_x[0]:.3f}, k={best_x[1]:.3f}, "
          f"φ={best_x[2]:.4f} (lam_x=0, basic)")

    _save_params(ctrl_name, param_dict, save_dir)
    _save_convergence(ctrl_name, cost_history, save_dir)
    best_ctrl = create_smc(best_x)
    _save_best_sim(ctrl_name, best_ctrl, Ts=Ts, save_dir=save_dir,
                   action_type=action_type)

    return best_x, param_dict


# ============================================================================

# SMC_WO TUNING (composite surface + wall override)

# ============================================================================

def tune_smc_wo(Ts=0.05, action_type="cont", n_jobs=None):
    """Tune SMC with wall override: (lam, k, phi, lam_x)."""
    from balancer.controllers.smc import SMCController

    ctrl_name = "smc_wo"
    save_dir = os.path.join(TUNING_ROOT, "cont", ctrl_name)

    def create_smc(x):
        lam, k, phi, lam_x = x
        return SMCController(lam=lam, k=k, phi=phi,
                             action_type="cont", lam_x=lam_x,
                             wall_margin=0.12, override_gain=0.7)

    def bounds_check(x):
        lam, k, phi, lam_x = x
        return (0.5 <= lam <= 10.0
                and 1.0 <= k <= 20.0
                and 0.05 <= phi <= 1.0
                and 0.0 <= lam_x <= 0.1)

    objective = robust_objective(create_smc, bounds_check,
                                 Ts=Ts, action_type=action_type)

    x0 = [3.5, 10.0, 0.25, 0.02]
    sigma0 = 0.5
    opts = {
        "bounds": [[0.5, 1.0, 0.05, 0.0], [10.0, 20.0, 1.0, 0.1]],
        "maxfevals": 980,
        "popsize": 32,
        "verb_disp": 1,
    }

    print("\n" + "=" * 70)
    print(f"TUNING SMC_WO (\u03bb, k, \u03c6, \u03b2) \u2014 plant: {action_type}")
    print("=" * 70)

    best_x, cost_history = _run_cmaes(
        objective, x0, sigma0, opts, ctrl_name, n_jobs=n_jobs)

    param_dict = {
        "lam": float(best_x[0]),
        "k":   float(best_x[1]),
        "phi": float(best_x[2]),
        "lam_x": float(best_x[3]),
        "wall_margin": 0.12,
        "override_gain": 0.7,
        "sigma_dead_zone": 0.0,
        "Ts":  Ts,
        "variant": "basic_wo",
        "tuned_on": action_type,
    }
    print(f"\n\u2705 Best SMC_WO: \u03bb={best_x[0]:.3f}, k={best_x[1]:.3f}, "
          f"\u03c6={best_x[2]:.4f}, \u03b2={best_x[3]:.4f}")

    _save_params(ctrl_name, param_dict, save_dir)
    _save_convergence(ctrl_name, cost_history, save_dir)
    best_ctrl = create_smc(best_x)
    _save_best_sim(ctrl_name, best_ctrl, Ts=Ts, save_dir=save_dir,
                   action_type=action_type)

    return best_x, param_dict


# ============================================================================

# NMPC TUNING

# ============================================================================

def tune_nmpc(Ts=0.05, action_type="cont", n_jobs=None):
    from balancer.controllers.nmpc import NMPCController

    ctrl_name = "nmpc"
    save_dir = os.path.join(TUNING_ROOT, ctrl_name)

    def create_nmpc(x):
        q1, q2, q3, q4, r1, N_float = x
        N = max(5, int(round(N_float)))
        ctrl = NMPCController(action_type="cont", N=N, dt=Ts)
        ctrl = _rebuild_nmpc_solver(ctrl, q1, q2, q3, q4, r1, N, Ts)
        return ctrl

    def bounds_check(x):
        q1, q2, q3, q4, r1, N_float = x
        return (
            all(qi >= 0 for qi in [q1, q2, q3, q4])
            and r1 > 1e-5
            and 5 <= N_float <= 30
        )

    objective = robust_objective(create_nmpc, bounds_check,
                                 Ts=Ts, action_type=action_type)

    x0 = [1e-3, 1e-3, 20.0, 1e-3, 0.2, 25]
    sigma0 = 0.5
    opts = {
        "bounds": [[1e-4, 1e-4, 1e-4, 1e-4, 1e-4, 5], [10, 10, 200, 10, 2.0, 30]],
        "maxfevals": 480,
        "popsize": 16,
        "verb_disp": 1,
    }

    print("\n" + "=" * 70)
    print(f"TUNING NMPC CONTROLLER (Q, R, N) - plant: {action_type}")
    print("=" * 70)

    effective_jobs = min(n_jobs or multiprocessing.cpu_count(), 4)

    best_x, cost_history = _run_cmaes(
        objective, x0, sigma0, opts, ctrl_name, n_jobs=n_jobs)

    best_N = max(5, int(round(best_x[5])))
    param_dict = {
        "Q_diag": [float(v) for v in best_x[:4]],
        "R": float(best_x[4]),
        "N": best_N,
        "Ts": Ts,
        "tuned_on": action_type,
    }
    print(f"\n[OK] Best NMPC: Q=diag({param_dict['Q_diag']}), "
          f"R={param_dict['R']:.4f}, N={best_N}")

    _save_params(ctrl_name, param_dict, save_dir)
    _save_convergence(ctrl_name, cost_history, save_dir)
    best_ctrl = create_nmpc(best_x)
    _save_best_sim(ctrl_name, best_ctrl, Ts=Ts, save_dir=save_dir,
                   action_type=action_type)

    return best_x, param_dict


def _rebuild_nmpc_solver(ctrl, q1, q2, q3, q4, r1, N, dt):
    """Rebuild CasADi NLP with new cost weights.

    Uses the controller's own _discretize (always velocity-input).
    """
    import casadi as ca

    nx, nu = 4, 1
    Q = np.diag([q1, q2, q3, q4])
    R = np.array([[r1]])
    Qf = Q * 2

    x = ca.SX.sym("x", nx)
    u = ca.SX.sym("u", nu)
    F = ctrl._discretize(x, u, dt)

    X = ca.SX.sym("X", nx, N + 1)
    U = ca.SX.sym("U", nu, N)
    P = ca.SX.sym("P", nx + nx)

    cost = 0
    g = []
    g.append(X[:, 0] - P[0:nx])

    for k in range(N):
        x_k = X[:, k]
        u_k = U[:, k]
        x_ref = P[nx:2 * nx]
        cost += ca.mtimes([(x_k - x_ref).T, Q, (x_k - x_ref)]) \
            + ca.mtimes([u_k.T, R, u_k])

        x_next = F(x_k, u_k)
        g.append(X[:, k + 1] - x_next)

    x_N = X[:, N]
    x_ref = P[nx:2 * nx]
    cost += ca.mtimes([(x_N - x_ref).T, Qf, (x_N - x_ref)])

    opt_vars = ca.vertcat(ca.reshape(X, -1, 1), ca.reshape(U, -1, 1))
    g = ca.vertcat(*g)

    nlp = {"f": cost, "x": opt_vars, "p": P, "g": g}
    solver = ca.nlpsol(
        "solver", "ipopt", nlp,
        {"ipopt.print_level": 0, "ipopt.tol": 1e-5,
         "print_time": False, "ipopt.max_iter": 500},
    )

    ctrl.N = min(N, 30)  # cap for real-time feasibility
    ctrl.dt = dt
    ctrl.nx = nx
    ctrl.nu = nu
    ctrl.solver = solver
    ctrl.X = X
    ctrl.U = U
    ctrl.P = P
    ctrl.opt_vars = opt_vars
    ctrl.g = g
    ctrl.Q_weights = [q1, q2, q3, q4]
    ctrl.R_weight = float(r1)
    q_str = "_".join(f"{v:.1f}" for v in [q1, q2, q3, q4])
    ctrl.model_name = f"NMPC_Q{q_str}_R{r1:.2f}_N{ctrl.N}"
    ctrl._prev_sol = None
    return ctrl


# ============================================================================

# COMPARISON

# ============================================================================

def compare_all_controllers(rl_model_paths=None, Ts=0.05,
                            action_type="cont"):
    """
    Simulate all controllers on representative ICs in both modes.

    Produces per-IC plots and summary for the requested action_type,
    plus validation on the other mode.
    """
    from balancer.controllers import (
        PIDController, LQRController, SMCController,
        MPCController, NMPCController,
    )

    save_dir = os.path.join(TUNING_ROOT, "comparison")
    _ensure_dir(save_dir)

    compare_modes = ["cont"]

    print("\n" + "=" * 70)
    print(f"CONTROLLER COMPARISON (modes: {compare_modes})")
    print("=" * 70)

    # --- Build controllers ---
    controllers = {}

    pid_params = _load_tuned_params("pid")
    pid = PIDController(action_type="cont")
    if pid_params:
        pid.Kp = pid_params.get("Kp", pid.Kp)
        pid.Ki = pid_params.get("Ki", pid.Ki)
        pid.Kd = pid_params.get("Kd", pid.Kd)
        controllers["PID (tuned)"] = pid
    else:
        controllers["PID"] = pid

    lqr_params = _load_tuned_params("lqr")
    if lqr_params and "K" in lqr_params:
        lqr = LQRController(K=np.array(lqr_params["K"]),
                            action_type="cont")
        controllers["LQR (tuned)"] = lqr
    else:
        controllers["LQR"] = LQRController(action_type="cont")

    smc_params = _load_tuned_params("smc")
    if smc_params:
        smc = SMCController(lam=smc_params.get("lam", 1.4),
                            k=smc_params.get("k", 3.0),
                            phi=smc_params.get("phi", 0.2),
                            action_type="cont")
        controllers["SMC (tuned)"] = smc
    else:
        controllers["SMC"] = SMCController(action_type="cont")

    mpc_params = _load_tuned_params("mpc")
    if mpc_params:
        Q = np.diag(mpc_params["Q_diag"])
        R = np.array([[mpc_params["R"]]])
        controllers["MPC (tuned)"] = MPCController(
            Q=Q, R=R, Ts=Ts, action_type="cont")
    else:
        controllers["MPC"] = MPCController(Ts=Ts, action_type="cont")

    nmpc_params = _load_tuned_params("nmpc")
    try:
        if nmpc_params:
            nmpc = NMPCController(
                action_type="cont",
                N=nmpc_params.get("N", 30),
                dt=nmpc_params.get("Ts", Ts),
            )
            nmpc = _rebuild_nmpc_solver(
                nmpc, *nmpc_params["Q_diag"],
                nmpc_params["R"], nmpc_params["N"],
                nmpc_params.get("Ts", Ts),
            )
            controllers["NMPC (tuned)"] = nmpc
        else:
            controllers["NMPC"] = NMPCController(
                action_type="cont", dt=Ts)
    except Exception as e:
        print(f"[WARN] Could not create NMPC: {e}")

    if rl_model_paths is not None:
        if isinstance(rl_model_paths, str):
            rl_model_paths = [rl_model_paths]
        for rl_path in rl_model_paths:
            try:
                from balancer.controllers import RLController
                rl = RLController(path_to_model=rl_path,
                                  action_type="cont")
                controllers[f"RL ({rl.model_name})"] = rl
            except Exception as e:
                print(f"[WARN] Could not load RL model {rl_path}: {e}")

    # --- ICs ---
    test_ics = {
        "small_pos":  np.array([0.0, 0.0, 0.01, 0.0]),
        "small_neg":  np.array([0.0, 0.0, -0.01, 0.0]),
        "large_pos":  np.array([0.0, 0.0, 0.074, 0.0]),
        "large_neg":  np.array([0.0, 0.0, -0.074, 0.0]),
        "cart_left":  np.array([-0.7, 0.0, 0.03, 0.0]),
        "cart_right": np.array([0.7, 0.0, -0.03, 0.0]),
        "worst_case": np.array([0.7765, 0.0, -0.075, 0.0]),
    }

    # --- Simulate per mode ---
    summary = {}
    for mode in compare_modes:
        print(f"\n{'-'*40} mode: {mode} {'-'*40}")
        all_metrics = {name: [] for name in controllers}

        for ic_name, s0 in test_ics.items():
            print(f"\n--- IC: {ic_name} = {s0} ---")
            traces = {}
            for ctrl_name, ctrl in controllers.items():
                time_arr, theta_arr, u_arr = simulate_controller(
                    ctrl, s0, Ts=Ts, sim_T=15.0, action_type=mode)
                traces[ctrl_name] = (time_arr, theta_arr, u_arr)
                metrics = compute_metrics_ise(time_arr, theta_arr, u_arr)
                metrics["ic"] = ic_name
                all_metrics[ctrl_name].append(metrics)
                print(f"  {ctrl_name:20s} | settle={metrics['settling_time']:6.2f}s "
                      f"| overshoot={metrics['overshoot']:.4f}")

            ic_dir = os.path.join(save_dir, f"{ic_name}_{mode}")
            plot_comparison(traces, save_dir=ic_dir)

        # Summary for this mode
        print(f"\n{'='*60}")
        print(f"SUMMARY - {mode} mode")
        print(f"{'='*60}")
        header = (f"{'Controller':20s} | {'Settle':>8s} | "
                  f"{'Overshoot':>10s} | {'Energy':>8s} | {'Jerk':>8s}")
        print(header)
        print("-" * len(header))

        mode_summary = {}
        for ctrl_name, metric_list in all_metrics.items():
            avg = {
                "avg_settling_time": float(np.mean(
                    [m["settling_time"] for m in metric_list])),
                "avg_overshoot": float(np.mean(
                    [m["overshoot"] for m in metric_list])),
                "avg_energy": float(np.mean(
                    [m["energy"] for m in metric_list])),
                "avg_jerk": float(np.mean(
                    [m["jerk"] for m in metric_list])),
            }
            mode_summary[ctrl_name] = avg
            print(f"{ctrl_name:20s} | {avg['avg_settling_time']:8.3f} | "
                  f"{avg['avg_overshoot']:10.5f} | "
                  f"{avg['avg_energy']:8.5f} | {avg['avg_jerk']:8.5f}")

        summary[mode] = mode_summary

    # Save combined summary
    summary_path = os.path.join(save_dir, "summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    # Bar chart (use first mode as primary)
    _plot_bar_chart(summary[compare_modes[0]], save_dir)

    print(f"\nResults saved to {save_dir}/")
    return summary


def _load_tuned_params(ctrl_name):
    path = os.path.join(TUNING_ROOT, ctrl_name, "best_params.json")
    if os.path.exists(path):
        with open(path, "r") as f:
            params = json.load(f)
        print(f"   Loaded tuned params for {ctrl_name} from {path}")
        return params
    return None


def _plot_bar_chart(summary, save_dir):
    names = list(summary.keys())
    metrics_keys = ["avg_settling_time", "avg_overshoot",
                    "avg_energy", "avg_jerk"]
    labels = ["Settling Time [s]", "Overshoot [rad]", "Energy", "Jerk"]

    fig, axes = plt.subplots(1, 4, figsize=(18, 5))
    x = np.arange(len(names))
    width = 0.6

    for ax, mkey, label in zip(axes, metrics_keys, labels):
        vals = [summary[n][mkey] for n in names]
        bars = ax.bar(x, vals, width, alpha=0.85)
        ax.set_ylabel(label)
        ax.set_xticks(x)
        ax.set_xticklabels(names, rotation=45, ha="right", fontsize=8)
        ax.grid(True, alpha=0.3, axis="y")
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2.0,
                    bar.get_height(), f"{val:.3f}",
                    ha="center", va="bottom", fontsize=7)

    fig.suptitle("Controller Comparison", fontsize=13)
    plt.tight_layout()
    path = os.path.join(save_dir, "bar_chart.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Bar chart saved to {path}")


# ============================================================================

# MPPI TUNING

# ============================================================================

def tune_mppi(
    Ts=0.05,
    action_type="cont",
    n_jobs=None,
    checkpoint="../models/world_model/v1/best_model.pth",
):
    """Tune MPPI hyperparameters (Q_diag, R, lambda, horizon, n_samples) via CMA-ES.

    Note: MPPI is GPU-accelerated but CMA-ES evaluates each candidate serially
    within the objective (the parallelism is across the population). Each eval
    runs 7 ICs × 30s sim = ~5s per candidate on CPU. With popsize=24 and
    1000 fevals that's ~200 generations.
    """
    import torch
    from balancer.world_model import WorldModelPredictor
    from balancer.controllers.mppi import MPPIController

    ctrl_name = "mppi"
    save_dir = os.path.join(TUNING_ROOT, ctrl_name)

    # Resolve checkpoint path
    ckpt_path = os.path.abspath(checkpoint)
    if not os.path.isfile(ckpt_path):
        # Try relative to evaluation/
        alt = os.path.join(os.path.dirname(__file__), "..", checkpoint)
        if os.path.isfile(alt):
            ckpt_path = os.path.abspath(alt)
        else:
            raise FileNotFoundError(
                f"World model checkpoint not found: {checkpoint}\n"
                f"  Tried: {ckpt_path}\n"
                f"  Also:  {os.path.abspath(alt)}"
            )

    # Use GPU for MPPI - each worker gets its own predictor
    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"  MPPI tuning device: {device}")
    print(f"  Checkpoint: {ckpt_path}")

    # Pre-load predictor (shared across all evaluations within a worker)
    # For parallel CMA-ES each worker loads its own copy.
    def create_mppi(x):
        q1, q2, q3, q4, r, lam, horizon_f, n_samples_f = x
        horizon = max(3, int(round(horizon_f)))
        n_samples = max(50, int(round(n_samples_f / 50) * 50))  # round to 50s

        predictor = WorldModelPredictor(ckpt_path, device=device)
        ctrl = MPPIController(
            predictor=predictor,
            horizon=horizon,
            n_samples=n_samples,
            lambda_=lam,
            device=device,
            Q_diag=[q1, q2, q3, q4],
            R=r,
        )
        return ctrl

    def bounds_check(x):
        q1, q2, q3, q4, r, lam, horizon_f, n_samples_f = x
        if not all(qi >= 0.0 for qi in [q1, q2, q3, q4]):
            return False
        if r < 0.001 or r > 1.0:
            return False
        if lam < 0.01 or lam > 10.0:
            return False
        if horizon_f < 3 or horizon_f > 40:
            return False
        if n_samples_f < 50 or n_samples_f > 2000:
            return False
        return True

    objective = robust_objective(create_mppi, bounds_check,
                                 Ts=Ts, action_type=action_type)

    # x0: [q1, q2, q3, q4, R, lambda, horizon, n_samples]
    x0 = [0.0, 1.0, 45.0, 1.0, 0.01, 1.0, 15.0, 500.0]
    sigma0 = 3.0
    opts = {
        "bounds": [
            [0.0, 0.0, 0.0, 0.0, 0.001, 0.01, 3.0, 50.0],
            [5.0, 10.0, 100.0, 10.0, 1.0, 10.0, 40.0, 2000.0],
        ],
        "maxfevals": 600,
        "popsize": 16,
        "verb_disp": 1,
    }

    # Force single worker - MPPI uses GPU, can't parallelise across processes
    n_jobs = 1

    print("\n" + "=" * 70)
    print(f"TUNING MPPI CONTROLLER (plant: {action_type})")
    print(f"  Parameters: Q_diag(4), R, λ, horizon, n_samples")
    print("=" * 70)

    best_x, cost_history = _run_cmaes(
        objective, x0, sigma0, opts, ctrl_name, n_jobs=n_jobs)

    best_horizon = max(3, int(round(best_x[6])))
    best_n_samples = max(50, int(round(best_x[7] / 50) * 50))

    param_dict = {
        "Q_diag": [float(best_x[0]), float(best_x[1]),
                   float(best_x[2]), float(best_x[3])],
        "R": float(best_x[4]),
        "lambda_": float(best_x[5]),
        "horizon": best_horizon,
        "n_samples": best_n_samples,
        "checkpoint": checkpoint,
        "Ts": Ts,
        "tuned_on": action_type,
    }
    print(f"\n[OK] Best MPPI: Q={param_dict['Q_diag']}, R={param_dict['R']:.4f}, "
          f"λ={param_dict['lambda_']:.3f}, H={best_horizon}, N={best_n_samples}")

    _save_params(ctrl_name, param_dict, save_dir)
    _save_convergence(ctrl_name, cost_history, save_dir)

    best_ctrl = create_mppi(best_x)
    _save_best_sim(ctrl_name, best_ctrl, Ts=Ts, save_dir=save_dir,
                   action_type=action_type)

    return best_x, param_dict


# ============================================================================

# MAIN

# ============================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Tune controllers using CMA-ES",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Tune PID on continuous plant (default, best for CMA-ES)
  python tune_controllers.py --controller pid

  # Compare all controllers on cont mode
  python tune_controllers.py --compare --action-type cont

  # Compare including RL models
  python tune_controllers.py --compare --rl-model models/policy.zip
        """,
    )
    parser.add_argument(
        "--controller",
        choices=["pid", "lqr", "mpc", "mpc_wo", "smc", "smc_wo", "nmpc", "mppi", "all"],
        default="all",
    )
    parser.add_argument("--Ts", type=float, default=0.05)
    parser.add_argument(
        "--action-type",
        choices=["cont"],
        default="cont",
        help="Plant mode for tuning: cont (smooth, default), "
             "discrete (bang-bang), both (averaged cost)",
    )
    parser.add_argument("--compare", action="store_true")
    parser.add_argument("--rl-model", type=str, nargs="*", default=None)
    parser.add_argument("--output-dir", type=str, default="tuning")
    parser.add_argument(
        "--jobs", type=int, default=None,
        help="Parallel workers for CMA-ES population (default: all cores). "
        "Use --jobs 1 to disable parallelism for debugging.",
    )
    parser.add_argument(
        "--mppi-checkpoint", type=str,
        default="models/world_model/v1/best_model.pth",
        help="Path to world model .pth checkpoint for MPPI tuning.",
    )
    args = parser.parse_args()

    TUNING_ROOT = args.output_dir
    at = args.action_type

    if args.compare:
        compare_all_controllers(rl_model_paths=args.rl_model,
                                Ts=args.Ts, action_type=at)
        exit(0)

    results = {}

    if args.controller in ["pid", "all"]:
        results["pid"] = tune_pid(args.Ts, action_type=at)

    if args.controller in ["lqr", "all"]:
        results["lqr"] = tune_lqr(args.Ts, action_type=at)

    if args.controller in ["mpc", "all"]:
        results["mpc"] = tune_mpc(args.Ts, action_type=at)

    if args.controller in ["mpc_wo"]:
        results["mpc_wo"] = tune_mpc_wo(args.Ts, action_type=at, n_jobs=args.jobs)

    if args.controller in ["smc", "all"]:
        results["smc"] = tune_smc(args.Ts, action_type=at)

    if args.controller in ["smc_wo", "all"]:
        results["smc_wo"] = tune_smc_wo(args.Ts, action_type=at, n_jobs=args.jobs)

    if args.controller in ["nmpc", "all"]:
        results["nmpc"] = tune_nmpc(args.Ts, action_type=at)

    if args.controller in ["mppi"]:
        results["mppi"] = tune_mppi(
            args.Ts, action_type=at, n_jobs=args.jobs,
            checkpoint=args.mppi_checkpoint,
        )

    print("\n" + "=" * 70)
    print(f"TUNING COMPLETE (plant: {at}) - results in: {TUNING_ROOT}/")
    print("=" * 70)
    for name, (best_x, param_dict) in results.items():
        print(f"  {name.upper()}: {param_dict}")

    print(f"\nCompare: python tune_controllers.py --compare --action-type both")
