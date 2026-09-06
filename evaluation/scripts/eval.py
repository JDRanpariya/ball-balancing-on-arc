#!/usr/bin/env python3
"""
Hardware-based controller benchmark.

Runs the same three experiment categories as eval_sim.py but on real hardware.
Only one experiment at a time is recommended (manual ball placement between trials).

Usage:
    # Exp 1: Baseline, discrete actions (most common)
    python scripts/eval.py --exp 1 --action-type discrete

    # Exp 1: Baseline, continuous actions
    python scripts/eval.py --exp 1 --action-type cont

    # Exp 2: Noise robustness (discrete)
    python scripts/eval.py --exp 2 --action-type discrete

    # Exp 3: Frequency robustness (discrete)
    python scripts/eval.py --exp 3 --action-type discrete

    # Classical only
    python scripts/eval.py --exp 1 --skip-rl

    # Specific controllers only
    python scripts/eval.py --exp 1 --controllers pid lqr ppo
"""

from utils.load_controllers import (
    build_benchmark_controllers,
    build_controller,
    print_loaded_params,
    discover_rl_models,
    discover_world_models,
    _build_rl,
    DISCRETE_RL_ALGOS,
    CONTINUOUS_RL_ALGOS,
    CLASSICAL_CONTROLLERS,
    MODEL_BASED_CONTROLLERS,
)
from balancer.hardware.constants import SYSTEM
from experiment.runner import run_experiment
from experiment.config import ExperimentConfig
from balancer.hardware import SystemState, SERIAL_CONFIG, start_reader_threads
import signal
import threading
import argparse
import sys
from pathlib import Path

import numpy as np
import serial
import torch
# Default to 4 threads for MPPI CPU inference: benchmark sweet spot on
# this hardware (faster than single-thread, lower jitter than 8/12).
# Override with --mppi-threads if needed.
torch.set_num_threads(4)

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


FAIL_TIME = SYSTEM.FAIL_TIME

print(SYSTEM)

# -- Stop signal ----------------------------------------------------------

stop_event = threading.Event()
signal.signal(signal.SIGINT, lambda s, f: stop_event.set())


# ===========================================================================

#  EXPERIMENT DEFINITIONS

# ===========================================================================

def get_exp1_configs(action_type: str, num_trials: int):
    return [ExperimentConfig(
        name=f"exp1_baseline_{action_type}",
        description=f"Exp1 - Baseline hardware, {action_type} at 20 Hz",
        env_label="nominal_ball",
        Ts=0.05,
        run_time=FAIL_TIME,
        num_trials=num_trials,
        noise_sigma=0.0,
        action_type=action_type,
        results_dir=f"results/real/exp1_{action_type}",
    )]


def get_exp2_configs(action_type: str, num_trials: int,
                     noise_levels=(0.0, 0.10, 0.20, 0.30)):
    exps = []
    for sigma in noise_levels:
        pct = int(sigma * 100)
        exps.append(ExperimentConfig(
            name=f"exp2_noise_{pct}pct_{action_type}",
            description=f"Exp2 - Noise {pct}%, {action_type} at 20 Hz",
            env_label="nominal_ball",
            Ts=0.05,
            run_time=FAIL_TIME,
            num_trials=num_trials,
            noise_sigma=sigma,
            action_type=action_type,
            results_dir=f"results/real/exp2_{action_type}",
        ))
    return exps


def get_exp3_configs(action_type: str, num_trials: int,
                     frequencies=(5.0, 10.0, 20.0, 50.0)):
    exps = []
    for freq in frequencies:
        Ts = round(1.0 / freq, 4)
        exps.append(ExperimentConfig(
            name=f"exp3_freq_{int(freq)}hz_{action_type}",
            description=f"Exp3 - {int(freq)} Hz, {action_type}",
            env_label="nominal_ball",
            Ts=Ts,
            run_time=FAIL_TIME,
            num_trials=num_trials,
            noise_sigma=0.0,
            action_type=action_type,
            results_dir=f"results/real/exp3_{action_type}",
        ))
    return exps


# ===========================================================================

#  TUNING ROOT AUTO-DETECTION

# ===========================================================================

def _find_tuning_root(action_type: str, explicit: str = None) -> str:
    if explicit:
        return explicit
    candidates = [
        f"tuning/tuning/{action_type}",
        f"tuning/{action_type}",
        "tuning/tuning/discrete",
        "tuning/discrete",
    ]
    for c in candidates:
        if Path(c).is_dir():
            return c
    return None


# ===========================================================================

#  MAIN

# ===========================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Hardware benchmark for ball-on-arc controllers",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/eval.py --exp 1 --action-type discrete
  python scripts/eval.py --exp 1 --action-type cont
  python scripts/eval.py --exp 2 --action-type discrete
  python scripts/eval.py --exp 3 --action-type discrete
  python scripts/eval.py --exp 1 --skip-rl
  python scripts/eval.py --exp 1 --controllers pid lqr smc mpc nmpc ppo dqn
  python scripts/eval.py --exp 1 --trials 20  # quick test
        """,
    )
    parser.add_argument(
        "--exp", type=str, required=True,
        choices=["1", "2", "3"],
        help="Which experiment to run.",
    )
    parser.add_argument(
        "--action-type", type=str, default="cont",
        choices=["discrete", "cont"],
        help="Action type (default: discrete).",
    )
    parser.add_argument(
        "--tuning-root", type=str, default=None,
        help="Path to tuned parameter directory.",
    )
    parser.add_argument(
        "--model-dir", type=str, default="models",
        help="Root directory for RL models.",
    )
    parser.add_argument(
        "--skip-rl", action="store_true",
        help="Skip RL controllers.",
    )
    parser.add_argument(
        "--skip-classical", action="store_true",
        help="Skip classical controllers (PID, LQR, SMC, MPC, NMPC).",
    )
    parser.add_argument(
        "--controllers", nargs="+", default=None,
        help="Explicit list of controller names to run "
             "(e.g. pid lqr ppo sac). Overrides defaults.",
    )
    parser.add_argument(
        "--trials", type=int, default=50,
        help="Number of trials per controller (default: 50).",
    )
    parser.add_argument(
        "--noise-levels", type=float, nargs="+",
        default=[0.0, 0.10, 0.20, 0.30],
        help="Noise sigma levels for Exp 2.",
    )
    parser.add_argument(
        "--frequencies", type=float, nargs="+",
        default=[5.0, 10.0, 20.0, 50.0],
        help="Control frequencies for Exp 3.",
    )
    parser.add_argument(
        "--models", type=str, nargs="+", default=None,
        help="Explicit .zip model paths to evaluate (bypasses model discovery).",
    )
    parser.add_argument(
        "--skip-mppi", action="store_true",
        help="Skip MPPI world model controllers.",
    )
    parser.add_argument(
        "--mppi-checkpoints", type=str, nargs="+", default=None,
        help="Explicit .pth checkpoint paths for MPPI evaluation.",
    )
    parser.add_argument(
        "--world-model-dir", type=str, default="models/world_model",
        help="Directory containing world model versions.",
    )
    parser.add_argument(
        "--mppi-only", action="store_true",
        help="Evaluate only MPPI controllers.",
    )
    parser.add_argument(
        "--nmpc-wall-coldstart", type=float, default=0.0, metavar="MARGIN",
        help="NMPC wall cold-start margin in metres (default 0 = disabled). "
             "When |cart_pos| > cart_limit - MARGIN, the IPOPT warm-start is "
             "discarded and a cold solve is used. Typical value: 0.05.",
    )
    parser.add_argument(
        "--mppi-threads", type=int, default=4, metavar="N",
        help="PyTorch intra-op threads for MPPI CPU inference "
             "(default: 4). Use 1 for conservative single-thread timing.",
    )
    args = parser.parse_args()

    # Apply requested PyTorch thread count before building controllers
    torch.set_num_threads(args.mppi_threads)

    atype = args.action_type
    tuning = _find_tuning_root(atype, args.tuning_root)

    # -- Hardware setup ------------------------------------------------
    print("Initializing hardware...")
    ser_ipc = serial.Serial(
        SERIAL_CONFIG.ipc_port, SERIAL_CONFIG.baudrate,
        timeout=SERIAL_CONFIG.ipc_timeout,
    )
    ser_distance = serial.Serial(
        SERIAL_CONFIG.distance_port, SERIAL_CONFIG.baudrate,
        timeout=SERIAL_CONFIG.distance_timeout,
    )
    state = SystemState()
    start_reader_threads(state, ser_ipc, ser_distance, stop_event)
    print("Hardware ready.")

    # -- Show tuned params ---------------------------------------------
    if tuning:
        print_loaded_params(tuning)

    # -- Build controllers ---------------------------------------------
    ctrl_pairs = []

    # Build RL from explicit model paths
    if args.models:
        import glob as _glob
        expanded = []
        for pattern in args.models:
            expanded.extend(_glob.glob(pattern))
        for path in sorted(expanded):
            stem = Path(path).stem
            # d3rlpy offline RL models (.d3 files)
            if path.endswith('.d3'):
                try:
                    from balancer.controllers.offline_rl import OfflineRLController
                    ctrl = OfflineRLController(path, device='cpu')
                    ctrl_pairs.append((f"OfflineRL-{stem}", ctrl))
                    print(f"  Loaded (d3rlpy): {stem} <- {path}")
                except Exception as e:
                    print(f"  SKIP {stem}: {e}")
                continue
            # SB3 models (.zip files)
            algo = 'ppo'
            for a in ['sac', 'td3', 'tqc', 'trpo', 'a2c', 'dqn', 'crossq']:
                if a in stem.lower():
                    algo = a
                    break
            try:
                ctrl = _build_rl(path, atype, algo=algo)
                ctrl.model_name = stem
                ctrl_pairs.append((f"RL-{stem}", ctrl))
                print(f"  Loaded: {stem} ({algo}) <- {path}")
            except Exception as e:
                print(f"  SKIP {stem}: {e}")

    # Build classical controllers (compatible with --models)
    if args.controllers:
        # User specified explicit list: split into classical + RL
        classical_requested = [
            c for c in args.controllers if c.lower() in CLASSICAL_CONTROLLERS
        ]
        model_based_requested = [
            c for c in args.controllers if c.lower() in MODEL_BASED_CONTROLLERS
        ]
        rl_requested = [
            c for c in args.controllers
            if c.lower() not in CLASSICAL_CONTROLLERS
            and c.lower() not in MODEL_BASED_CONTROLLERS
        ]

        # Build classical (append to ctrl_pairs, don't reset)
        if not args.skip_classical:
            for name in classical_requested:
                try:
                    extra = {}
                    if name.lower() == "nmpc" and args.nmpc_wall_coldstart > 0:
                        extra["wall_coldstart_margin"] = args.nmpc_wall_coldstart
                    ctrl = build_controller(
                        name, tuning_root=tuning,
                        action_type=atype, Ts=0.05, use_tuned=True,
                        **extra,
                    )
                    src = ctrl._param_source
                    label = f"{name.upper()} ({'tuned' if src == 'tuned' else 'default'})"
                    ctrl_pairs.append((label, ctrl))
                    if name.lower() == "nmpc" and args.nmpc_wall_coldstart > 0:
                        print(f"  NMPC wall cold-start margin: {args.nmpc_wall_coldstart}m")
                except Exception as e:
                    print(f"{name.upper()}: {e}")

        # Build model-based controllers (MPPI)
        if model_based_requested:
            for name in model_based_requested:
                try:
                    # Auto-discover checkpoint or use explicit
                    ckpt = None
                    if args.mppi_checkpoints:
                        ckpt = args.mppi_checkpoints[0]
                    else:
                        wm_models = discover_world_models(args.world_model_dir)
                        if wm_models:
                            ckpt = wm_models[0][1]  # first discovered
                    if ckpt is None:
                        print(f"  {name.upper()}: No world model checkpoint found")
                        continue
                    ctrl = build_controller(
                        name, tuning_root=tuning,
                        action_type="cont", Ts=0.05,
                        use_tuned=True, checkpoint=ckpt,
                    )
                    label = f"MPPI"
                    ctrl_pairs.append((label, ctrl))
                    print(f"  {label}: loaded from {ckpt}")
                except Exception as e:
                    print(f"  {name.upper()}: {e}")

        # Build RL from requested algo names
        if rl_requested and not args.skip_rl:
            all_rl = discover_rl_models(args.model_dir, atype)
            rl_map = {algo: path for algo, path in all_rl}
            for algo_name in rl_requested:
                algo_lower = algo_name.lower()
                if algo_lower in rl_map:
                    try:
                        ctrl = _build_rl(
                            rl_map[algo_lower], atype, algo=algo_lower)
                        ctrl_pairs.append((f"RL-{algo_lower.upper()}", ctrl))
                        print(f"RL-{algo_lower.upper()}: {rl_map[algo_lower]}")
                    except Exception as e:
                        print(f"  RL-{algo_lower.upper()}: {e}")
                else:
                    print(f"RL model for '{algo_name}' not found in "
                          f"{args.model_dir}/{atype}/")
    if not args.models and not args.controllers:
        # Default: all controllers (no explicit selection)
        skip_rl_flag = args.skip_rl or args.mppi_only
        skip_mppi_flag = args.skip_mppi
        ctrl_pairs = build_benchmark_controllers(
            action_type=atype,
            Ts=0.05,
            tuning_root=tuning,
            model_dir=args.model_dir,
            use_tuned=True,
            skip_rl=skip_rl_flag,
            skip_mppi=skip_mppi_flag,
            mppi_checkpoints=args.mppi_checkpoints,
            world_model_dir=args.world_model_dir,
        )

        if args.skip_classical or args.mppi_only:
            ctrl_pairs = [
                (label, ctrl) for label, ctrl in ctrl_pairs
                if label.startswith("RL-") or label.startswith("MPPI-")
            ]
            if args.mppi_only:
                ctrl_pairs = [
                    (label, ctrl) for label, ctrl in ctrl_pairs
                    if label.startswith("MPPI-")
                ]

    controllers = [ctrl for _, ctrl in ctrl_pairs]

    if not controllers:
        print("No controllers built. Exiting.")
        return

    print(f"\n  Controllers to evaluate ({len(controllers)}):")
    for label, _ in ctrl_pairs:
        print(f"    - {label}")

    # -- Build experiment configs --------------------------------------
    if args.exp == "1":
        experiments = get_exp1_configs(atype, args.trials)
    elif args.exp == "2":
        experiments = get_exp2_configs(
            atype, args.trials, tuple(args.noise_levels))
    elif args.exp == "3":
        experiments = get_exp3_configs(
            atype, args.trials, tuple(args.frequencies))

    # -- Run experiments -----------------------------------------------
    for exp_cfg in experiments:
        if stop_event.is_set():
            print("\n Stop signal received. Exiting.")
            break

        # For Exp 3 (frequency robustness): rebuild classical controllers
        # with the correct Ts so their internal models match the loop rate.
        # RL controllers are unaffected (no internal Ts dependency).
        if args.exp == "3" and exp_cfg.Ts != 0.05:
            exp_controllers = []
            for label, ctrl in ctrl_pairs:
                if hasattr(ctrl, 'Ts') or hasattr(ctrl, 'dt'):
                    # Rebuild classical controller with correct Ts
                    ctrl_name = label.split()[0].lower().rstrip('()')
                    # Map label back to controller name
                    # Check nmpc BEFORE mpc (mpc is substring of nmpc)
                    name_order = ['pid', 'lqr', 'smc', 'nmpc', 'mpc']
                    rebuild_name = None
                    for k in name_order:
                        if k in label.lower():
                            rebuild_name = k
                            break
                    if rebuild_name:
                        try:
                            new_ctrl = build_controller(
                                rebuild_name, tuning_root=tuning,
                                action_type=atype, Ts=exp_cfg.Ts,
                                use_tuned=True,
                            )
                            exp_controllers.append(new_ctrl)
                            print(f"  Rebuilt {label} with Ts={exp_cfg.Ts}")
                        except Exception as e:
                            print(f"  Failed to rebuild {label}: {e}")
                            exp_controllers.append(ctrl)
                    else:
                        exp_controllers.append(ctrl)
                else:
                    exp_controllers.append(ctrl)
        else:
            exp_controllers = controllers

        print(f"\n{'='*70}")
        print(f"  Experiment : {exp_cfg.name}")
        print(f"  Action     : {exp_cfg.action_type}")
        print(f"  Ts         : {exp_cfg.Ts}s ({1/exp_cfg.Ts:.0f} Hz)")
        print(f"  Noise σ    : {exp_cfg.noise_sigma}")
        print(f"  Trials     : {exp_cfg.num_trials}")
        print(f"  Output     : {exp_cfg.output_dir}")
        print(f"{'='*70}")

        run_experiment(
            controllers=exp_controllers,
            ser_ipc=ser_ipc,
            state=state,
            stop_event=stop_event,
            cfg=exp_cfg,
        )

    # -- Cleanup -------------------------------------------------------
    print(f"\n{'='*70}")
    print("All hardware experiments complete.")
    print(f"{'='*70}")
    print("\nResults directories:")
    seen = set()
    for exp in experiments:
        d = exp.results_dir
        if d not in seen:
            print(f"  {d}/")
            seen.add(d)


if __name__ == "__main__":
    main()
