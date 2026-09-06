"""
Shared controller factory - builds classical + RL controllers for benchmarks.

Usage:
    from utils.load_controllers import build_benchmark_controllers

    controllers = build_benchmark_controllers(
        action_type="discrete",
        Ts=0.05,
        tuning_root="tuning/tuning/discrete",
        model_dir="../models",
    )
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple

import numpy as np

from balancer.core.device import resolve_device

# ===========================================================================

#  RL ALGORITHM SETS PER ACTION TYPE

# ===========================================================================

DISCRETE_RL_ALGOS = [
    "ppo", "a2c", "dqn", "qrdqn", "trpo", "recurrent_ppo", "ars",
]

CONTINUOUS_RL_ALGOS = [
    # "ppo_wm" is the PPO-WM policy (ppo_wm_v1.zip): a PPO agent trained in the
    # learned world model and deployed zero-shot. Listed after "ppo" so discovery
    # picks up ppo_wm_v1.zip as its own 13th benchmark controller (RL-PPO_WM),
    # rather than being masked by ppo.zip.
    "ppo", "ppo_wm", "a2c", "sac", "td3", "trpo", "recurrent_ppo", "crossq", "tqc", "ars",
]

# Controllers that match the paper Table 1 benchmark.
# PID and LQR have WO baked into their tuned params (wall_margin/override_gain),
# so they are the WO variants. SMC and MPC have separate non-WO (basic)
# ablation variants (reported in H2), excluded from the main benchmark.
CLASSICAL_CONTROLLERS = ["pid", "lqr", "smc_wo", "mpc_wo", "nmpc"]
MODEL_BASED_CONTROLLERS = ["mppi"]

# ===========================================================================

#  PARAM LOADING

# ===========================================================================

_DEFAULT_SEARCH_ROOTS = [
    "tuning/tuning/discrete",
    "tuning/tuning/cont",
    "tuning/discrete",
    "tuning/cont",
    "tuning",
]


def find_params_file(
    ctrl_name: str,
    tuning_root: Optional[str] = None,
) -> Optional[str]:
    candidates = []
    if tuning_root:
        candidates.append(os.path.join(tuning_root, ctrl_name, "best_params.json"))
    for root in _DEFAULT_SEARCH_ROOTS:
        candidates.append(os.path.join(root, ctrl_name, "best_params.json"))
    for path in candidates:
        if os.path.isfile(path):
            return path
    return None


def load_params(
    ctrl_name: str,
    tuning_root: Optional[str] = None,
) -> Optional[dict]:
    path = find_params_file(ctrl_name, tuning_root)
    if path is None:
        return None
    with open(path, "r") as f:
        params = json.load(f)
    print(f"   {ctrl_name.upper()}: loaded tuned params from {path}")
    return params


# ===========================================================================

#  INDIVIDUAL CONTROLLER BUILDERS

# ===========================================================================

def _build_pid(params: Optional[dict], action_type: str, Ts: float, **kw):
    from balancer.controllers.pid import PIDController
    defaults = {"Kp": 19.782, "Ki": 0.0, "Kd": 1.367, "Ts": Ts,
                "wall_margin": 0.12, "override_gain": 0.5}
    if params:
        for k in ("Kp", "Ki", "Kd", "wall_margin", "override_gain"):
            if k in params:
                defaults[k] = params[k]
        # Note: Ts from params is NOT loaded - experiment Ts always wins
    defaults.update(kw)
    return PIDController(
        Kp=defaults["Kp"], Ki=defaults["Ki"], Kd=defaults["Kd"],
        Ts=defaults["Ts"], action_type=action_type,
        wall_margin=defaults["wall_margin"],
        override_gain=defaults["override_gain"],
    )


def _build_lqr(params: Optional[dict], action_type: str, Ts: float, **kw):
    from balancer.controllers.lqr import LQRController
    defaults = {"tau_v": 0.15, "wall_margin": 0.12, "override_gain": 0.5}
    if params:
        for k_name in ("tau_v", "wall_margin", "override_gain"):
            if k_name in params:
                defaults[k_name] = params[k_name]
    # Explicit kwargs override both the hardcoded default and the params
    # file - same precedence K already gets, extended to WO params so a
    # no-WO ablation (wall_margin=0 in its best_params.json, or passed
    # explicitly) can't be silently overridden by the 0.12/0.5 default.
    for k_name in ("tau_v", "wall_margin", "override_gain"):
        if k_name in kw:
            defaults[k_name] = kw.pop(k_name)

    K = None
    if params and "K" in params:
        K = np.array(params["K"])
    if "K" in kw:
        K = np.array(kw.pop("K"))

    if K is not None:
        return LQRController(K=K, action_type=action_type, Ts=Ts,
                             tau_v=defaults["tau_v"],
                             wall_margin=defaults["wall_margin"],
                             override_gain=defaults["override_gain"], **kw)
    else:
        return LQRController(action_type=action_type, Ts=Ts,
                             tau_v=defaults["tau_v"],
                             wall_margin=defaults["wall_margin"],
                             override_gain=defaults["override_gain"], **kw)


def _build_smc(params: Optional[dict], action_type: str, Ts: float, **kw):
    from balancer.controllers.smc import SMCController
    defaults = {"lam": 3.0, "k": 6.0, "phi": 0.806, "tau_v": 0.15,
                "lam_x": 0.0, "wall_margin": 0.12, "override_gain": 0.5,
                "sigma_dead_zone": 0.0, "k_d": 0.5, "sensor_offset": 0.0}
    if params:
        for k_name in ("lam", "k", "phi", "tau_v", "lam_x",
                       "wall_margin", "override_gain", "sigma_dead_zone",
                       "k_d", "sensor_offset"):
            if k_name in params:
                defaults[k_name] = params[k_name]
    defaults.update(kw)
    return SMCController(lam=defaults["lam"], k=defaults["k"],
                         phi=defaults["phi"], tau_v=defaults["tau_v"],
                         lam_x=defaults["lam_x"],
                         wall_margin=defaults["wall_margin"],
                         override_gain=defaults["override_gain"],
                         sigma_dead_zone=defaults["sigma_dead_zone"],
                         k_d=defaults["k_d"],
                         sensor_offset=defaults["sensor_offset"],
                         action_type=action_type)


def _build_mpc(params: Optional[dict], action_type: str, Ts: float, **kw):
    from balancer.controllers.mpc import MPCController
    defaults = {"Q_diag": [0.001, 0, 40, 0], "R": 0.017, "N": 10, "tau_v": 0.15,
                "wall_margin": 0.12, "override_gain": 0.5}
    if params:
        for k_name in ("Q_diag", "R", "N", "tau_v", "wall_margin", "override_gain"):
            if k_name in params:
                defaults[k_name] = params[k_name]
    defaults.update(kw)
    Q = np.diag(defaults["Q_diag"])
    R = np.array([[defaults["R"]]])
    return MPCController(
        Q=Q, R=R, Ts=Ts, N=defaults["N"],
        tau_v=defaults["tau_v"], action_type=action_type,
        wall_margin=defaults["wall_margin"], override_gain=defaults["override_gain"],
    )


def _build_mpc_wo(params: Optional[dict], action_type: str, Ts: float, **kw):
    from balancer.controllers.mpc import MPCController
    defaults = {"Q_diag": [0.001, 0, 40, 0], "R": 0.017, "N": 10,
                "tau_v": 0.15, "wall_margin": 0.12, "override_gain": 0.5,
                "cart_constraint": False}
    if params:
        for k_name in ("Q_diag", "R", "N", "tau_v", "wall_margin",
                       "override_gain", "cart_constraint"):
            if k_name in params:
                defaults[k_name] = params[k_name]
    defaults.update(kw)
    Q = np.diag(defaults["Q_diag"])
    R = np.array([[defaults["R"]]])
    return MPCController(
        Q=Q, R=R, Ts=Ts, N=defaults["N"],
        tau_v=defaults["tau_v"], action_type=action_type,
        wall_margin=defaults["wall_margin"],
        override_gain=defaults["override_gain"],
        cart_constraint=defaults["cart_constraint"],
    )


def _build_nmpc(params: Optional[dict], action_type: str, Ts: float, **kw):
    from balancer.controllers.nmpc import NMPCController
    defaults = {"Q_diag": [0, 0, 5, 0], "R": 0.05, "N": 10, "wall_margin": 0.0,
                "override_gain": 0.0}
    if params:
        for k_name in ("Q_diag", "R", "N"):
            if k_name in params:
                defaults[k_name] = params[k_name]
    defaults.update(kw)
    wall_coldstart_margin = defaults.pop("wall_coldstart_margin", 0.0)
    ctrl = NMPCController(
        action_type=action_type,
        N=defaults["N"],
        dt=Ts,
        Q=defaults["Q_diag"],
        R=defaults["R"],
        wall_coldstart_margin=wall_coldstart_margin,
    )
    if params or "Q_diag" in kw:
        # Rebuild with tuned weights (updates solver + model_name)
        ctrl = _rebuild_nmpc_solver(
            ctrl, *defaults["Q_diag"], defaults["R"],
            defaults["N"], Ts,
        )
        ctrl.wall_coldstart_margin = wall_coldstart_margin
    return ctrl


def _build_mppi(params: Optional[dict], action_type: str, Ts: float, **kw):
    import torch
    from balancer.controllers.mppi import MPPIController
    from balancer.world_model import WorldModelPredictor

    checkpoint = kw.pop("checkpoint", None)
    if checkpoint is None and params:
        checkpoint = params.get("checkpoint")
    if checkpoint is None:
        raise ValueError(
            "MPPI requires a 'checkpoint' kwarg or params['checkpoint'] "
            "pointing to a trained world model .pth file."
        )

    stats_path = kw.pop("stats_path", None)
    if stats_path is None and params:
        stats_path = params.get("stats_path")

    device = kw.pop("device", None)  # None -> auto-resolve cuda/mps/cpu
    predictor = WorldModelPredictor(checkpoint, stats_path=stats_path, device=device)

    defaults = {
        "horizon": 30,
        "n_samples": 800,
        "lambda_": 0.4,
        "Q_diag": [0.0, 0.1, 30.0, 0.1],
        "R": 0.001,
        "error_correction": False,
        "terminal_cost": False,
        "context_window": 10,
        "noise_type": "uniform",
        "stateful": True,
    }
    if params:
        for k_name in defaults:
            if k_name in params:
                defaults[k_name] = params[k_name]
    defaults.update(kw)

    print(f"  MPPI: Q={defaults['Q_diag']}, R={defaults['R']}, λ={defaults['lambda_']}, "
          f"H={defaults['horizon']}, N={defaults['n_samples']}")

    return MPPIController(
        predictor=predictor,
        horizon=defaults["horizon"],
        n_samples=defaults["n_samples"],
        lambda_=defaults["lambda_"],
        device=device,
        Q_diag=defaults["Q_diag"],
        R=defaults["R"],
        error_correction=defaults["error_correction"],
        terminal_cost=defaults["terminal_cost"],
        noise_type=defaults["noise_type"],
        stateful=defaults["stateful"],
        context_window=defaults["context_window"],
    )


def _build_rl(
    path_to_model: str,
    action_type: str,
    algo: str = None,
    **kw,
):
    from balancer.controllers.rl import RLController
    return RLController(
        path_to_model=path_to_model,
        action_type=action_type,
        algo=algo,
    )


def _build_offline_rl(
    path_to_model: str,
    device: str = "cpu",
    **kw,
):
    from balancer.controllers.offline_rl import OfflineRLController
    return OfflineRLController(
        path_to_model=path_to_model,
        device=device,
    )


# ===========================================================================

#  NMPC SOLVER REBUILD

# ===========================================================================

def _rebuild_nmpc_solver(ctrl, q1, q2, q3, q4, r1, N, dt):
    import casadi as ca
    N = min(N, 30)  # cap for real-time feasibility
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
    g = [X[:, 0] - P[0:nx]]
    for k in range(N):
        x_k, u_k = X[:, k], U[:, k]
        x_ref = P[nx:2 * nx]
        cost += ca.mtimes([(x_k - x_ref).T, Q, (x_k - x_ref)]) \
              + ca.mtimes([u_k.T, R, u_k])

        g.append(X[:, k + 1] - F(x_k, u_k))
    x_ref = P[nx:2 * nx]
    cost += ca.mtimes([(X[:, N] - x_ref).T, Qf, (X[:, N] - x_ref)])
    opt_vars = ca.vertcat(ca.reshape(X, -1, 1), ca.reshape(U, -1, 1))
    g = ca.vertcat(*g)
    solver = ca.nlpsol(
        "solver", "ipopt",
        {"f": cost, "x": opt_vars, "p": P, "g": g},
        {"ipopt.print_level": 0, "ipopt.tol": 1e-5,
         "print_time": False, "ipopt.max_iter": 500},
    )
    ctrl.N, ctrl.dt, ctrl.nx, ctrl.nu = N, dt, nx, nu
    ctrl.solver, ctrl.X, ctrl.U, ctrl.P = solver, X, U, P
    ctrl.opt_vars, ctrl.g = opt_vars, g
    ctrl.Q_weights = [q1, q2, q3, q4]
    ctrl.R_weight = float(r1)
    q_str = "_".join(f"{v:.1f}" for v in [q1, q2, q3, q4])
    ctrl.model_name = f"NMPC_Q{q_str}_R{r1:.2f}_N{N}"
    ctrl._prev_sol = None
    return ctrl


# ===========================================================================

#  CLASSICAL CONTROLLER DISPATCHER

# ===========================================================================

_BUILDERS = {
    "pid":  _build_pid,
    "lqr":  _build_lqr,
    "smc":  _build_smc,
    "smc_wo": _build_smc,
    "mpc":     _build_mpc,
    "mpc_wo":  _build_mpc_wo,
    "nmpc": _build_nmpc,
    "mppi": _build_mppi,
}


def build_controller(
    ctrl_name: str,
    tuning_root: Optional[str] = None,
    action_type: str = "cont",
    Ts: float = 0.05,
    use_tuned: bool = True,
    **overrides,
) -> Any:
    ctrl_name = ctrl_name.lower()
    if ctrl_name not in _BUILDERS:
        raise ValueError(f"Unknown controller: {ctrl_name}")
    params = load_params(ctrl_name, tuning_root) if use_tuned else None
    source = "tuned" if params else "default"
    if not params and use_tuned:
        print(f"  WARNING  {ctrl_name.upper()}: no tuned params found, using defaults")
    ctrl = _BUILDERS[ctrl_name](params, action_type, Ts, **overrides)
    ctrl._param_source = source
    ctrl._tuning_root = tuning_root
    # Canonical model_name so the JSON controller keys match the committed
    # benchmark (paper/data/exp1_sim/*.json). The deployed PID/LQR/MPC variants
    # carry the "_WO" suffix, SMC's deployed variant is keyed plain "SMC", and
    # NMPC is keyed by its solver horizon (NMPC_N10 / N20 / N30). This also gives
    # each a distinct key so it does not collide with a non-WO variant.
    _DEPLOYED_NAME = {
        "pid": "PID_WO", "lqr": "LQR_WO",
        "smc_wo": "SMC", "mpc_wo": "MPC_WO",
    }
    if ctrl_name in _DEPLOYED_NAME:
        ctrl.model_name = _DEPLOYED_NAME[ctrl_name]
    elif ctrl_name == "nmpc":
        ctrl.model_name = f"NMPC_N{getattr(ctrl, 'N', 10)}"
    return ctrl


# ===========================================================================

#  RL MODEL DISCOVERY

# ===========================================================================

def discover_world_models(
    model_dir: str = "models/world_model",
    versions: Optional[List[str]] = None,
) -> List[Tuple[str, str]]:
    """
    Find world model versions for MPPI evaluation.

    Only the deployed version (v12) is returned by default. The sweep_*/v1
    checkpoints are data-scaling ablations (Appendix VI), not the deployed
    controller.

    Expected directory structure:
        {model_dir}/
        +-- v12/                 <- deployed (paper H1, Table 1)
        |   +-- best_model.pth
        |   +-- norm_stats.npz
        +-- sweep_*/             <- ablation (data-scaling sweep)

    Args:
        model_dir:  Root directory containing world model versions.
        versions:  Explicit list of version names to load (default: ["v12"]).

    Returns:
        List of (version_name, checkpoint_path) tuples.
    """
    if not os.path.isdir(model_dir):
        return []

    if versions is None:
        versions = ["v12"]

    found = []
    for entry in sorted(os.listdir(model_dir)):
        if entry not in versions:
            continue
        version_dir = os.path.join(model_dir, entry)
        if not os.path.isdir(version_dir):
            continue
        ckpt = os.path.join(version_dir, "best_model.pth")
        stats = os.path.join(version_dir, "norm_stats.npz")
        if os.path.isfile(ckpt) and os.path.isfile(stats):
            found.append((entry, ckpt))

    return found


def discover_offline_rl_models(
    model_dir: str = "models/offline_rl",
) -> List[Tuple[str, str]]:
    """
    Find all offline RL (d3rlpy) .d3 model files.

    Searches both the top-level directory and one level of subdirectories
    (e.g. iql_best_demos/model_280000.d3).

    Returns:
        List of (algo_name, model_path) tuples.
    """
    if not os.path.isdir(model_dir):
        return []

    found = []
    _offline_algos = ["cql", "iql", "td3bc", "bcq"]

    # Collect .d3 files from top-level and subdirectories
    d3_files = []
    for entry in sorted(os.listdir(model_dir)):
        full = os.path.join(model_dir, entry)
        if os.path.isfile(full) and entry.endswith(".d3"):
            d3_files.append(full)
        elif os.path.isdir(full):
            for f in sorted(os.listdir(full)):
                if f.endswith(".d3"):
                    d3_files.append(os.path.join(full, f))

    for path in d3_files:
        f = os.path.basename(path)
        stem = f.lower().replace(".d3", "")
        algo = "offline_rl"
        for a in _offline_algos:
            if a in stem or a in path.lower():
                algo = a
                break
        found.append((algo, path))
    return found


def discover_rl_models(
    model_dir: str,
    action_type: str,
) -> List[Tuple[str, str]]:
    """
    Find all RL model .zip files for a given action type.

    Expected directory structure:
        {model_dir}/
        +-- discrete/
        |   +-- ppo.zip
        |   +-- dqn.zip
        |   +-- ...
        +-- cont/
            +-- sac.zip
            +-- td3.zip
            +-- ...

    Returns:
        List of (algo_name, model_path) tuples
    """
    subdir = os.path.join(model_dir, action_type)
    if not os.path.isdir(subdir):
        print(f"  WARNING  RL model directory not found: {subdir}")
        return []

    expected = DISCRETE_RL_ALGOS if action_type == "discrete" else CONTINUOUS_RL_ALGOS
    found = []

    for algo in expected:
        # Try common naming conventions
        candidates = [
            os.path.join(subdir, f"{algo}.zip"),
            os.path.join(subdir, algo, "model.zip"),
            os.path.join(subdir, algo, "best_model.zip"),
        ]
        for path in candidates:
            if os.path.isfile(path):
                found.append((algo, path))
                break
        else:
            # Also search for any .zip containing the algo name
            if os.path.isdir(subdir):
                for f in os.listdir(subdir):
                    if f.endswith(".zip") and algo in f.lower():
                        found.append((algo, os.path.join(subdir, f)))
                        break

    return found


# ===========================================================================

#  FULL BENCHMARK BUILDER

# ===========================================================================

def build_benchmark_controllers(
    action_type: str = "cont",
    Ts: float = 0.05,
    tuning_root: Optional[str] = None,
    model_dir: str = "../models",
    classical_names: Optional[List[str]] = None,
    use_tuned: bool = True,
    skip_rl: bool = False,
    rl_model_paths: Optional[List[Tuple[str, str]]] = None,
    skip_mppi: bool = False,
    mppi_checkpoints: Optional[List[str]] = None,
    world_model_dir: str = "models/world_model",
) -> List[Tuple[str, Any]]:
    """
    Build all controllers for a benchmark experiment.

    Args:
        action_type:      "discrete" or "cont"
        Ts:               Sampling period (s)
        tuning_root:      Path to tuned parameter directory
        model_dir:        Root directory for RL model .zip files
        classical_names:  Which classical controllers (default: all)
        use_tuned:        Load tuned params for classical controllers
        skip_rl:          Skip all RL controllers
        rl_model_paths:   Explicit list of (algo_name, path) - overrides discovery
        skip_mppi:        Skip MPPI world model controllers
        mppi_checkpoints: Explicit list of checkpoint paths for MPPI.
                          If None and not skip_mppi, auto-discovers from world_model_dir.
        world_model_dir:  Directory containing world model versions.

    Returns:
        List of (label, controller) tuples
    """
    if classical_names is None:
        classical_names = CLASSICAL_CONTROLLERS

    print(f"\n{'-'*60}")
    print(f"Building benchmark controllers")
    print(f"  action_type={action_type}, Ts={Ts}s ({1/Ts:.0f} Hz)")
    if tuning_root:
        print(f"  tuning_root={tuning_root}")
    print(f"{'-'*60}")

    controllers = []

    # -- Classical controllers -----------------------------------------
    for name in classical_names:
        try:
            ctrl = build_controller(
                name, tuning_root=tuning_root,
                action_type="cont", Ts=Ts,
                use_tuned=use_tuned,
            )
            src = ctrl._param_source
            label = f"{name.upper()} ({'tuned' if src == 'tuned' else 'default'})"
            controllers.append((label, ctrl))
        except Exception as e:
            print(f"  [FAIL] {name.upper()}: {e}")

    # -- RL controllers ------------------------------------------------
    if not skip_rl:
        if rl_model_paths is None:
            rl_model_paths = discover_rl_models(model_dir, action_type)

        for algo_name, model_path in rl_model_paths:
            try:
                ctrl = _build_rl(model_path, action_type, algo=algo_name)
                label = f"RL-{algo_name.upper()}"
                controllers.append((label, ctrl))
                print(f"   {label}: loaded from {model_path}")
            except Exception as e:
                print(f"  [FAIL] RL-{algo_name.upper()}: {e}")

    # -- Offline RL controllers (d3rlpy) ---------------------------------
    if not skip_rl and action_type == "cont":
        offline_models = discover_offline_rl_models(
            os.path.join(model_dir, "offline_rl")
        )
        for algo_name, model_path in offline_models:
            try:
                ctrl = _build_offline_rl(model_path)
                label = f"OffRL-{algo_name.upper()}"
                controllers.append((label, ctrl))
                print(f"   {label}: loaded from {model_path}")
            except Exception as e:
                print(f"  [FAIL] OffRL-{algo_name.upper()}: {e}")

    # -- MPPI world model controllers ----------------------------------
    if not skip_mppi and action_type == "cont":
        if mppi_checkpoints is not None:
            # Explicit checkpoint list
            for ckpt_path in mppi_checkpoints:
                try:
                    version = Path(ckpt_path).parent.name
                    ctrl = build_controller(
                        "mppi", tuning_root=tuning_root,
                        action_type="cont", Ts=Ts,
                        use_tuned=False, checkpoint=ckpt_path,
                    )
                    label = f"MPPI-{version}"
                    controllers.append((label, ctrl))
                    print(f"   {label}: loaded from {ckpt_path}")
                except Exception as e:
                    print(f"  [FAIL] MPPI ({ckpt_path}): {e}")
        else:
            # Auto-discover world model versions
            wm_models = discover_world_models(world_model_dir)
            for version, ckpt_path in wm_models:
                try:
                    ctrl = build_controller(
                        "mppi", tuning_root=tuning_root,
                        action_type="cont", Ts=Ts,
                        use_tuned=True, checkpoint=ckpt_path,
                    )
                    label = f"MPPI-{version}"
                    controllers.append((label, ctrl))
                    print(f"   {label}: loaded from {ckpt_path}")
                except Exception as e:
                    print(f"  [FAIL] MPPI-{version}: {e}")

    print(f"\n  [OK] Built {len(controllers)} controllers: "
          f"{[n for n, _ in controllers]}\n")

    return controllers


# ===========================================================================

#  LEGACY API (backward compatible)

# ===========================================================================

def build_controllers(
    names=None, tuning_root=None, action_type="discrete",
    Ts=0.05, use_tuned=True, rl_model_paths=None, skip_on_error=True,
):
    """Backward-compatible builder. Use build_benchmark_controllers for new code."""
    if names is None:
        names = list(_BUILDERS.keys())

    controllers = []
    for name in names:
        try:
            ctrl = build_controller(name, tuning_root, action_type, Ts, use_tuned)
            label = f"{name.upper()} ({'tuned' if ctrl._param_source == 'tuned' else 'default'})"
            controllers.append((label, ctrl))
        except Exception as e:
            if skip_on_error:
                print(f"  [FAIL] {name.upper()}: {e}")
            else:
                raise

    if rl_model_paths:
        for path in rl_model_paths:
            try:
                ctrl = _build_rl(path, action_type)
                label = f"RL ({Path(path).stem})"
                controllers.append((label, ctrl))
            except Exception as e:
                if skip_on_error:
                    print(f"  [FAIL] RL ({path}): {e}")
                else:
                    raise

    return controllers


def print_loaded_params(tuning_root=None):
    print(f"\n{'='*50}")
    print("AVAILABLE TUNED PARAMETERS")
    print(f"{'='*50}")
    for name in CLASSICAL_CONTROLLERS:
        params = load_params(name, tuning_root)
        if params:
            tuned_on = params.get("tuned_on", "unknown")
            if name == "pid":
                print(f"  PID:  Kp={params['Kp']:.3f}  Ki={params['Ki']:.3f}  "
                      f"Kd={params['Kd']:.3f}  (tuned on: {tuned_on})")
            elif name == "lqr":
                K = params.get("K", [])
                print(f"  LQR:  K={[f'{k:.3f}' for k in K]}  (tuned on: {tuned_on})")
            elif name == "smc":
                print(f"  SMC:  λ={params['lam']:.3f}  k={params['k']:.3f}  "
                      f"(tuned on: {tuned_on})")
            elif name in ("mpc", "nmpc"):
                print(f"  {name.upper()}:  Q={params.get('Q_diag')}  "
                      f"R={params.get('R'):.4f}  N={params.get('N')}  "
                      f"(tuned on: {tuned_on})")
        else:
            print(f"  {name.upper()}:  (no tuned params found)")
