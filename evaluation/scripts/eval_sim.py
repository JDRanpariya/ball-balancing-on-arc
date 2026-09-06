# !/usr/bin/env python3
"""
Full simulation benchmark for all controllers.

Runs three experiment categories:
    Exp 1: Baseline performance (nominal conditions)
    Exp 2: Sensor noise robustness
    Exp 3: Control frequency robustness

Model directory convention:
    models/
    +-- discrete/       # Discrete action models (PPO, A2C, DQN, QRDQN, TRPO, RecurrentPPO, ARS)
    |   +-- ppo.zip
    |   +-- a2c.zip
    |   +-- dqn.zip
    |   +-- qrdqn.zip
    |   +-- trpo.zip
    |   +-- recurrent_ppo.zip
    |   +-- ars.zip
    +-- cont/           # Continuous action models (PPO, A2C, SAC, TD3, TRPO, RecurrentPPO, CrossQ, TQC, ARS)
        +-- ppo.zip
        +-- a2c.zip
        +-- sac.zip
        +-- td3.zip
        +-- trpo.zip
        +-- recurrent_ppo.zip
        +-- crossq.zip
        +-- tqc.zip
        +-- ars.zip

Usage:
    # Run all experiments
    python scripts/eval_sim.py

    # Baseline only
    python scripts/eval_sim.py --exp 1

    # Noise robustness only, discrete
    python scripts/eval_sim.py --exp 2 --action-type discrete

    # Frequency robustness only
    python scripts/eval_sim.py --exp 3

    # Skip RL models (classical only)
    python scripts/eval_sim.py --skip-rl

    # Custom paths
    python scripts/eval_sim.py --model-dir ../models --tuning-root tuning/tuning/discrete
"""

import sys
from pathlib import Path
# Allow running directly from evaluation/scripts/ (imports resolve from evaluation/)
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from utils.load_controllers import (
    build_benchmark_controllers,
    print_loaded_params,
    discover_world_models,
    DISCRETE_RL_ALGOS,
    CONTINUOUS_RL_ALGOS,
    CLASSICAL_CONTROLLERS,
)
from balancer.hardware.constants import SYSTEM
from experiment.runner_sim import run_sim_experiment
from experiment.config import ExperimentConfig
import numpy as np
import sys
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


FAIL_TIME = SYSTEM.FAIL_TIME

# Known RL algo names for path inference
_KNOWN_ALGOS = ["ppo", "sac", "td3", "tqc", "trpo", "a2c", "dqn", "qrdqn",
                "crossq", "ars", "recurrent_ppo"]


def _infer_algo_from_path(path: str) -> str:
    """Infer RL algorithm name from a model .zip filename."""
    stem = Path(path).stem.lower()
    for algo in _KNOWN_ALGOS:
        if algo in stem:
            return algo
    return "ppo"  # fallback


# ===========================================================================

# EXPERIMENT DEFINITIONS

# ===========================================================================

def get_exp1_baseline(action_types=("discrete", "cont")):
    """Experiment 1: Baseline performance - nominal conditions."""
    exps = []
    for atype in action_types:
        exps.append(ExperimentConfig(
            name=f"exp1_baseline_{atype}",
            description=f"Exp1 - Baseline, {atype} actions at 20 Hz, 100 trials",
            Ts=0.05,
            run_time=FAIL_TIME,
            num_trials=50,
            noise_sigma=0.0,
            action_type=atype,
            results_dir=f"results/sim/exp1_{atype}",
        ))
    return exps


def get_exp2_noise(action_types=("discrete", "cont"),
                   noise_levels=(0.03, 0.06, 0.09)):
    """Experiment 2: Sensor noise robustness."""
    exps = []
    for atype in action_types:
        for sigma in noise_levels:
            pct = int(sigma * 100)
            exps.append(ExperimentConfig(
                name=f"exp2_noise_{pct}pct_{atype}",
                description=f"Exp2 - Noise {pct}%, {atype} actions at 20 Hz",
                Ts=0.05,
                run_time=FAIL_TIME,
                num_trials=50,
                noise_sigma=sigma,
                action_type=atype,
                results_dir=f"results/sim/exp2_{atype}",
            ))
    return exps


def get_exp3_frequency(action_types=("discrete", "cont"),
                       frequencies=(5.0, 10.0, 20.0, 50.0)):
    """Experiment 3: Control frequency robustness."""
    exps = []
    for atype in action_types:
        for freq in frequencies:
            Ts = round(1.0 / freq, 4)
            exps.append(ExperimentConfig(
                name=f"exp3_freq_{int(freq)}hz_{atype}",
                description=f"Exp3 - {int(freq)} Hz, {atype} actions",
                Ts=Ts,
                run_time=FAIL_TIME,
                num_trials=50,
                noise_sigma=0.0,
                action_type=atype,
                results_dir=f"results/sim/exp3_{atype}",
            ))
    return exps



# ===========================================================================

#  TUNING ROOT AUTO-DETECTION

# ===========================================================================

def _find_tuning_root(action_type: str, explicit: str = None) -> str:
    """Find the best tuning root for a given action type."""
    if explicit:
        return explicit
    candidates = [
        f"tuning/tuning/{action_type}",
        f"tuning/{action_type}",
        "tuning/tuning/discrete",  # fallback: discrete params often work for both
        "tuning/discrete",
    ]
    for c in candidates:
        if Path(c).is_dir():
            return c
    return None


# ===========================================================================

#  SINGLE EXPERIMENT RUNNER

# ===========================================================================

def run_single_experiment(
    exp_cfg: ExperimentConfig,
    tuning_root: str = None,
    model_dir: str = "../models",
    skip_rl: bool = False,
    rl_only: bool = False,
    classical_names: list = None,
    rl_model_paths: list = None,
    seed: int = 42,
    skip_mppi: bool = False,
    mppi_checkpoints: list = None,
    world_model_dir: str = "models/world_model",
    mppi_only: bool = False,
):
    """Run one experiment configuration with all relevant controllers."""

    tuning = _find_tuning_root(exp_cfg.action_type, tuning_root)

    print(f"\n{'='*70}")
    print(f"  Experiment : {exp_cfg.name}")
    print(f"  Action     : {exp_cfg.action_type}")
    print(f"  Ts         : {exp_cfg.Ts}s ({1/exp_cfg.Ts:.0f} Hz)")
    print(f"  Noise σ    : {exp_cfg.noise_sigma}")
    print(f"  Trials     : {exp_cfg.num_trials}")
    print(f"  Tuning     : {tuning or 'defaults'}")
    print(f"  Output     : {exp_cfg.results_dir}")
    if rl_only:
        print(f"  Mode       : RL-only")
    elif skip_rl:
        print(f"  Mode       : Classical-only")
    print(f"{'='*70}")

    # Determine which classical controllers to include
    use_classical = (not rl_only) and (not mppi_only) and classical_names
    ctrl_classical_names = classical_names if use_classical else ([] if (rl_only or mppi_only) else None)

    ctrl_pairs = build_benchmark_controllers(
        action_type=exp_cfg.action_type,
        Ts=exp_cfg.Ts,
        tuning_root=tuning,
        model_dir=model_dir,
        classical_names=ctrl_classical_names,
        use_tuned=True,
        skip_rl=skip_rl or mppi_only,
        rl_model_paths=rl_model_paths,
        skip_mppi=skip_mppi or rl_only,
        mppi_checkpoints=mppi_checkpoints,
        world_model_dir=world_model_dir,
    )

    controllers = [ctrl for _, ctrl in ctrl_pairs]

    return run_sim_experiment(
        controllers=controllers,
        cfg=exp_cfg,
        seed=seed,
    )


# ===========================================================================

#  MAIN

# ===========================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Full simulation benchmark for ball-on-arc controllers",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run everything (all 3 experiments × both action types)
  python scripts/eval_sim.py

  # Baseline only, both action types
  python scripts/eval_sim.py --exp 1

  # Noise robustness, discrete only
  python scripts/eval_sim.py --exp 2 --action-type discrete

  # Frequency robustness, continuous only
  python scripts/eval_sim.py --exp 3 --action-type cont

  # Classical controllers only (no RL)
  python scripts/eval_sim.py --skip-rl

  # Custom model directory
  python scripts/eval_sim.py --model-dir /path/to/models

  # Fewer trials for quick testing
  python scripts/eval_sim.py --trials 10 --exp 1
        """,
    )
    parser.add_argument(
        "--exp", type=str, default="all",
        choices=["1", "2", "3", "all"],
        help="Which experiment(s) to run (default: all).",
    )
    parser.add_argument(
        "--action-type", type=str, nargs="+",
        default=["discrete", "cont"],
        choices=["discrete", "cont"],
        help="Action type(s) to benchmark (default: both).",
    )
    parser.add_argument(
        "--tuning-root", type=str, default=None,
        help="Path to tuning results dir.",
    )
    parser.add_argument(
        "--model-dir", type=str, default="models",
        help="Root directory for RL model .zip files.",
    )
    parser.add_argument(
        "--skip-rl", action="store_true",
        help="Skip RL controllers (classical only).",
    )
    parser.add_argument(
        "--trials", type=int, default=None,
        help="Override number of trials per experiment.",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed.",
    )
    parser.add_argument(
        "--noise-levels", type=float, nargs="+",
        default=[0.03, 0.06, 0.09],
        help="Noise sigma values for Exp 2 (default: 0.03 0.06 0.09).",
    )
    parser.add_argument(
        "--frequencies", type=float, nargs="+",
        default=[5.0, 10.0, 20.0, 50.0],
        help="Control frequencies for Exp 3 (default: 5 10 20 50).",
    )
    parser.add_argument(
        "--models", type=str, nargs="+", default=None,
        help="Explicit .zip model paths to evaluate (overrides model discovery).",
    )
    parser.add_argument(
        "--rl-only", action="store_true",
        help="Skip classical controllers, evaluate only RL models.",
    )
    parser.add_argument(
        "--classical-only", action="store_true",
        help="Skip RL models, evaluate only classical controllers.",
    )
    parser.add_argument(
        "--classical", type=str, nargs="+", default=None,
        choices=CLASSICAL_CONTROLLERS,
        help="Subset of classical controllers (default: all).",
    )
    parser.add_argument(
        "--checkpoint-sweep", type=str, nargs="+", default=None,
        help="Glob-expanded .zip paths for checkpoint sweep evaluation.",
    )
    parser.add_argument(
        "--paper-export", type=str, default=None,
        help="Copy results JSON to paper/data/<name>/ for reproducibility.",
    )
    parser.add_argument(
        "--motor", type=str, default="first_order",
        choices=["first_order", "s_curve"],
        help="Motor model for simulation (default: first_order).",
    )
    parser.add_argument(
        "--skip-mppi", action="store_true",
        help="Skip MPPI world model controllers.",
    )
    parser.add_argument(
        "--mppi-checkpoints", type=str, nargs="+", default=None,
        help="Explicit .pth checkpoint paths for MPPI evaluation. "
             "If not set, auto-discovers from --world-model-dir.",
    )
    parser.add_argument(
        "--world-model-dir", type=str, default="models/world_model",
        help="Directory containing world model versions (default: models/world_model).",
    )
    parser.add_argument(
        "--mppi-only", action="store_true",
        help="Evaluate only MPPI controllers (skip classical + RL).",
    )
    args = parser.parse_args()

    atypes = tuple(args.action_type)

    # Resolve --models / --checkpoint-sweep into explicit rl_model_paths
    explicit_rl_paths = None
    if args.models:
        # Explicit model paths - infer algo from filename
        import glob
        expanded = []
        for pattern in args.models:
            expanded.extend(glob.glob(pattern))
        explicit_rl_paths = []
        for p in sorted(expanded):
            algo = _infer_algo_from_path(p)
            explicit_rl_paths.append((algo, p))
        print(f"\n--- Explicit models ({len(explicit_rl_paths)}) ---")
        for algo, path in explicit_rl_paths:
            print(f"  {algo.upper():12s} -> {path}")
    elif args.checkpoint_sweep:
        import glob
        expanded = []
        for pattern in args.checkpoint_sweep:
            expanded.extend(glob.glob(pattern))
        explicit_rl_paths = []
        for p in sorted(expanded):
            algo = _infer_algo_from_path(p)
            explicit_rl_paths.append((algo, p))
        print(f"\n--- Checkpoint sweep ({len(explicit_rl_paths)} models) ---")
        for algo, path in explicit_rl_paths:
            print(f"  {algo.upper():12s} -> {path}")

    # Determine skip flags
    skip_rl = args.skip_rl or args.classical_only or args.mppi_only
    rl_only = args.rl_only or (args.models is not None) or (args.checkpoint_sweep is not None)
    mppi_only = args.mppi_only
    skip_mppi = args.skip_mppi
    classical_names = args.classical  # None means all

    # Show available tuned parameters
    if not rl_only:
        for at in atypes:
            tr = _find_tuning_root(at, args.tuning_root)
            if tr:
                print(f"\n--- Tuned params for '{at}' ---")
                print_loaded_params(tr)

    # Show expected RL models (only if not using explicit paths)
    if not skip_rl and explicit_rl_paths is None:
        from utils.load_controllers import discover_rl_models
        for at in atypes:
            models = discover_rl_models(args.model_dir, at)
            if models:
                print(f"\n--- RL models for '{at}' ({len(models)} found) ---")
                for algo, path in models:
                    print(f"  {algo.upper():20s} -> {path}")
            else:
                print(f"\n  WARNING  No RL models found for '{at}' in {args.model_dir}/{at}/")

    # Show world model checkpoints
    if not skip_mppi and "cont" in atypes:
        wm_models = discover_world_models(args.world_model_dir)
        if wm_models:
            print(f"\n--- World models for MPPI ({len(wm_models)} found) ---")
            for version, path in wm_models:
                print(f"  {version:20s} -> {path}")
        elif not args.mppi_checkpoints:
            print(f"\n  WARNING  No world models found in {args.world_model_dir}/")

    # -- Build experiment list -----------------------------------------
    experiments = []

    if args.exp in ("1", "all"):
        experiments.extend(get_exp1_baseline(atypes))

    if args.exp in ("2", "all"):
        experiments.extend(get_exp2_noise(atypes, tuple(args.noise_levels)))

    if args.exp in ("3", "all"):
        experiments.extend(get_exp3_frequency(atypes, tuple(args.frequencies)))

    # -- Override trial count if specified ------------------------------
    if args.trials is not None:
        for exp in experiments:
            exp.num_trials = args.trials

    # -- Set motor model -----------------------------------------------
    for exp in experiments:
        exp.motor_model = args.motor

    # -- Print plan ----------------------------------------------------
    print(f"\n{'='*70}")
    print(f"BENCHMARK PLAN: {len(experiments)} experiment configs")
    print(f"{'='*70}")
    for i, exp in enumerate(experiments):
        print(f"  [{i+1:2d}] {exp.name:40s}  "
              f"Ts={exp.Ts}s  noise={exp.noise_sigma}  "
              f"trials={exp.num_trials}  action={exp.action_type}")

    # -- Expected RL algorithms ----------------------------------------
    print(f"\n{'-'*70}")
    print("EXPECTED RL ALGORITHMS PER ACTION TYPE:")
    print(f"  discrete: {DISCRETE_RL_ALGOS}")
    print(f"  cont:     {CONTINUOUS_RL_ALGOS}")
    print(f"{'-'*70}")

    # -- Run -----------------------------------------------------------
    for i, exp_cfg in enumerate(experiments):
        print(f"\n\n{'#'*70}")
        print(f"  Running [{i+1}/{len(experiments)}]: {exp_cfg.name}")
        print(f"{'#'*70}")

        run_single_experiment(
            exp_cfg=exp_cfg,
            tuning_root=args.tuning_root,
            model_dir=args.model_dir,
            skip_rl=skip_rl,
            rl_only=rl_only,
            classical_names=classical_names,
            rl_model_paths=explicit_rl_paths,
            seed=args.seed,
            skip_mppi=skip_mppi,
            mppi_checkpoints=args.mppi_checkpoints,
            world_model_dir=args.world_model_dir,
            mppi_only=mppi_only,
        )

    # -- Summary -------------------------------------------------------
    print(f"\n\n{'='*70}")
    print(f"[OK] BENCHMARK COMPLETE - {len(experiments)} experiments finished")
    print(f"{'='*70}")
    print("\nResults directories:")
    seen = set()
    for exp in experiments:
        d = exp.results_dir
        if d not in seen:
            print(f"  {d}/")
            seen.add(d)

    print("\nNext steps:")
    print("  # Analyze Exp 1 (baseline):")
    print("  python utils/analyze_exp.py --exp 1 --files results/sim/exp1_discrete/.../data.json")
    print("")
    print("  # Analyze Exp 2 (noise robustness):")
    print("  python utils/analyze_exp.py --exp 2 \\")
    print("    --files results/sim/exp2_discrete/exp2_noise_0pct_*/data.json \\")
    print("           results/sim/exp2_discrete/exp2_noise_10pct_*/data.json \\")
    print("           results/sim/exp2_discrete/exp2_noise_20pct_*/data.json \\")
    print("           results/sim/exp2_discrete/exp2_noise_30pct_*/data.json")
    print("")
    print("  # Analyze Exp 3 (frequency robustness):")
    print("  python utils/analyze_exp.py --exp 3 \\")
    print("    --files results/sim/exp3_discrete/exp3_freq_5hz_*/data.json \\")
    print("           results/sim/exp3_discrete/exp3_freq_10hz_*/data.json \\")
    print("           results/sim/exp3_discrete/exp3_freq_20hz_*/data.json \\")
    print("           results/sim/exp3_discrete/exp3_freq_50hz_*/data.json")

    # -- Paper export --------------------------------------------------
    if args.paper_export:
        import shutil
        paper_dir = Path(__file__).resolve().parent.parent.parent / "paper" / "data" / args.paper_export
        paper_dir.mkdir(parents=True, exist_ok=True)
        for exp in experiments:
            src = Path(exp.results_dir)
            if src.exists():
                for json_file in src.rglob("*.json"):
                    dst = paper_dir / json_file.name
                    shutil.copy2(json_file, dst)
                    print(f"  Exported: {dst}")
        print(f"\n  Paper data -> {paper_dir}")


if __name__ == "__main__":
    main()
