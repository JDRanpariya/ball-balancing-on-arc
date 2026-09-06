#!/usr/bin/env python3
"""
Batch-train continuous-action RL agents for the ball-balancer.

Trains every (algorithm × reward) pair sequentially, then collects
wall-clock time, reward-milestone steps (90 %/95 % in eval & training
rollouts), and copies the best model into evaluation/models/cont/.

Algorithms
    on-policy  (16 envs): PPO, TRPO
    off-policy ( 1 env ): SAC, TD3, CrossQ, TQC

Reward functions
    ball_gaussian_distance   (max ≈ 1.0 /step -> 600 /episode)
    balanced                 (max ≈ 0.6 /step -> 360 /episode)

Usage (from project root):
    python training/scripts/train_all_cont.py
    python training/scripts/train_all_cont.py --timesteps 500000
    python training/scripts/train_all_cont.py --dry-run
    python training/scripts/train_all_cont.py --wandb   # enable W&B

    # Filter to specific algos / rewards:
    python training/scripts/train_all_cont.py --algos ppo sac --rewards balanced

    # Enable domain randomisation:
    python training/scripts/train_all_cont.py --algos ppo --rewards balanced --dr

    # Both base + DR in one batch:
    python training/scripts/train_all_cont.py --algos ppo sac --rewards balanced --both-dr

    # Custom output tag (models saved as <tag>.zip instead of <algo>_cont_<reward>.zip):
    python training/scripts/train_all_cont.py --algos ppo --rewards balanced --dr --tag ppo_dr

Outputs:
    evaluation/models/cont/<algo>_cont_<reward>.zip   (or <tag>.zip if --tag)
    evaluation/models/cont/training_summary.json

NOTE: if your sim.yaml env config does not already contain action_type
      and reward keys, you may need to use the + prefix:
          +env.action_type=cont  +env.reward=balanced
      (Hydra 1.3 requires + when *adding* a key to a config group.)
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

# ===========================================================================
#  PATHS (relative to this script -> project root)
# ===========================================================================
SCRIPT_DIR = Path(__file__).resolve().parent
TRAINING_DIR = SCRIPT_DIR.parent
PROJECT_ROOT = TRAINING_DIR.parent
TRAIN_PY = SCRIPT_DIR / "train.py"
MODELS_OUT = PROJECT_ROOT / "evaluation" / "models" / "cont"

# ===========================================================================
#  WHAT TO TRAIN
# ===========================================================================
ON_POLICY = ["ppo", "trpo"]
OFF_POLICY = ["sac", "td3", "crossq", "tqc"]
ALL_ALGOS = ON_POLICY + OFF_POLICY

REWARDS = ["ball_gaussian_distance", "balanced"]

# Theoretical max episode reward  (max_per_step × 600 steps)
MAX_EP_REWARD: dict[str, float] = {
    "ball_gaussian_distance": 600.0,
    "balanced":               360.0,
}
MILESTONE_PCTS = [90, 95]

# ===========================================================================
#  DEFAULT HYPER-PARAMETERS  (overridable via CLI)
# ===========================================================================
DEFAULTS = dict(
    total_timesteps=1_000_000,
    eval_freq=50_000,
    n_eval_episodes=10,
    checkpoint_freq=100_000,
    max_checkpoints=50,
    seed=1,
)


def _n_envs(algo: str) -> int:
    """On-policy -> 16 parallel envs; off-policy -> 1."""
    return 16 if algo in ON_POLICY else 1


# ===========================================================================
#  HYDRA COMMAND BUILDER
# ===========================================================================
def _build_cmd(algo: str, reward: str, run_dir: Path, cfg: dict,
               *, enable_dr: bool = False,
               enable_blind_zone: bool = False) -> list[str]:
    cmd = [
        sys.executable, str(TRAIN_PY),
        # -- hydra output --
        f"hydra.run.dir={run_dir}",
        # -- config-group switch --
        f"algo={algo}",
        # -- env overrides --
        "env.action_type=cont",
        f"env.reward={reward}",
        # -- training --
        f"n_envs={_n_envs(algo)}",
        f"total_timesteps={cfg['total_timesteps']}",
        f"seed={cfg['seed']}",
        # -- evaluation --
        f"evaluation.eval_freq={cfg['eval_freq']}",
        f"evaluation.n_eval_episodes={cfg['n_eval_episodes']}",
        # -- checkpoints --
        f"checkpoint.save_freq={cfg['checkpoint_freq']}",
        f"checkpoint.max_checkpoints={cfg['max_checkpoints']}",
        # -- misc --
        "save_model=True",
        "capture_video=False",
    ]
    if enable_dr:
        cmd.append("env.fixed_param=False")
    if enable_blind_zone:
        cmd.append("env.sensor_blind_zone=True")
    return cmd


# ===========================================================================
#  POST-RUN: LOCATE MODEL
# ===========================================================================
def _find_model(run_dir: Path) -> Path | None:
    """best_model > final model > latest checkpoint."""
    for candidate in [
        run_dir / "best_model" / "best_model.zip",   # EvalCallback best
        run_dir / "models" / "model.zip",             # final save
    ]:
        if candidate.exists():
            return candidate
    ckpts = sorted(run_dir.rglob("checkpoint_step_*.zip"))
    return ckpts[-1] if ckpts else None


# ===========================================================================
#  POST-RUN: MILESTONE EXTRACTION  (TensorBoard)
# ===========================================================================
def _parse_milestones(run_dir: Path, reward: str) -> dict[str, int]:
    """
    First global step at which eval/train reward crosses 90 %/95 % of
    the theoretical episode maximum.
    """
    max_r = MAX_EP_REWARD.get(reward, 600.0)
    milestones: dict[str, int] = {}

    event_files = list(run_dir.rglob("events.out.tfevents.*"))
    tb_dirs = list({str(ef.parent) for ef in event_files})
    if not tb_dirs:
        return milestones

    try:
        from tensorboard.backend.event_processing.event_accumulator import (
            EventAccumulator,
        )
    except ImportError:
        print("  WARNING  tensorboard not importable - skipping milestones")
        return milestones

    for tb_d in tb_dirs:
        try:
            ea = EventAccumulator(tb_d, size_guidance={"scalars": 0})
            ea.Reload()
        except Exception:
            continue

        for tag, label in [
            ("eval/mean_reward",    "eval"),
            ("rollout/ep_rew_mean", "train"),
        ]:
            if tag not in ea.Tags().get("scalars", []):
                continue
            events = ea.Scalars(tag)
            for pct in MILESTONE_PCTS:
                key = f"{label}_{pct}pct_step"
                if key in milestones:
                    continue
                threshold = max_r * pct / 100.0
                for ev in events:
                    if ev.value >= threshold:
                        milestones[key] = ev.step
                        break
    return milestones


# ===========================================================================
#  POST-RUN: PARSE evaluations.npz  (fallback / complementary)
# ===========================================================================
def _parse_eval_npz(run_dir: Path, reward: str) -> dict[str, int]:
    """
    If EvalCallback was configured with log_path, evaluations.npz
    contains timesteps and results - often more reliable than TB.
    """
    import numpy as np

    max_r = MAX_EP_REWARD.get(reward, 600.0)
    milestones: dict[str, int] = {}

    npz_files = list(run_dir.rglob("evaluations.npz"))
    if not npz_files:
        return milestones

    data = np.load(npz_files[0])
    timesteps = data["timesteps"]          # shape (n_evals,)
    results = data["results"]            # shape (n_evals, n_episodes)
    mean_rewards = results.mean(axis=1)    # per-eval mean

    for pct in MILESTONE_PCTS:
        key = f"eval_{pct}pct_step"
        if key in milestones:
            continue
        threshold = max_r * pct / 100.0
        idxs = np.where(mean_rewards >= threshold)[0]
        if len(idxs) > 0:
            milestones[key] = int(timesteps[idxs[0]])

    return milestones


# ===========================================================================
#  RUN ONE CONFIGURATION
# ===========================================================================
def _run_one(
    idx: int,
    total: int,
    algo: str,
    reward: str,
    batch_dir: Path,
    cfg: dict,
    *,
    dry_run: bool = False,
    enable_wandb: bool = False,
    enable_dr: bool = False,
    enable_blind_zone: bool = False,
    output_tag: str = None,
) -> dict:

    dr_suffix = "_dr" if enable_dr else ""
    bz_suffix = "_bz" if enable_blind_zone else ""
    tag = output_tag or f"{algo}_cont_{reward}{dr_suffix}{bz_suffix}"
    run_dir = batch_dir / tag
    cmd = _build_cmd(algo, reward, run_dir, cfg,
                    enable_dr=enable_dr,
                    enable_blind_zone=enable_blind_zone)

    env_vars = {**os.environ}
    if not enable_wandb:
        env_vars["WANDB_MODE"] = "disabled"

    # -- header --
    print(f"\n{'-'*70}")
    print(f"  [{idx}/{total}]  {tag}")
    print(f"  algo={algo}  reward={reward}  DR={enable_dr}  BZ={enable_blind_zone}  n_envs={_n_envs(algo)}")
    print(f"  -> {run_dir}")
    print(f"{'-'*70}")

    if dry_run:
        print("  CMD:")
        print("    " + " \\\n    ".join(cmd))
        return dict(tag=tag, algo=algo, reward=reward, dry_run=True)

    t0 = time.time()
    proc = subprocess.run(cmd, env=env_vars)
    elapsed = time.time() - t0

    info: dict = dict(
        tag=tag,
        algo=algo,
        reward=reward,
        domain_randomisation=enable_dr,
        blind_zone=enable_blind_zone,
        n_envs=_n_envs(algo),
        run_dir=str(run_dir),
        elapsed_s=round(elapsed, 1),
        elapsed=str(timedelta(seconds=int(elapsed))),
        returncode=proc.returncode,
    )

    if proc.returncode != 0:
        info.update(model_path=None, milestones={})
        print(f"  X  FAILED  (return code {proc.returncode})")
        return info

    # -- copy best model --
    src = _find_model(run_dir)
    if src:
        MODELS_OUT.mkdir(parents=True, exist_ok=True)
        dst = MODELS_OUT / f"{tag}.zip"
        shutil.copy2(src, dst)
        info["model_path"] = str(dst)
        print(f"  OK  Model -> {dst.relative_to(PROJECT_ROOT)}")
    else:
        info["model_path"] = None
        print("  WARNING  No model file found")

    # -- milestones (try npz first, then TB) --
    ms = _parse_eval_npz(run_dir, reward)
    ms.update(_parse_milestones(run_dir, reward))   # TB fills remaining keys
    info["milestones"] = ms
    if ms:
        for k, v in sorted(ms.items()):
            print(f"     {k}: step {v:,}")
    else:
        print("     (no milestones reached)")

    return info


# ===========================================================================
#  ENTRY POINT
# ===========================================================================
def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--timesteps",  type=int,   default=DEFAULTS["total_timesteps"],
                    help="Total env steps per run  (default: 1 000 000)")
    ap.add_argument("--eval-freq",  type=int,   default=DEFAULTS["eval_freq"],
                    help="Evaluate every N steps   (default: 50 000)")
    ap.add_argument("--seed",       type=int,   default=DEFAULTS["seed"])
    ap.add_argument("--wandb",      action="store_true",
                    help="Enable W&B logging (disabled by default)")
    ap.add_argument("--algos",      type=str, nargs="+", default=None,
                    help="Subset of algorithms to train (default: all)")
    ap.add_argument("--rewards",    type=str, nargs="+", default=None,
                    help="Subset of reward functions (default: all)")
    ap.add_argument("--dr",         action="store_true",
                    help="Enable domain randomisation (fixed_param=False)")
    ap.add_argument("--blind-zone", action="store_true",
                    help="Enable sensor blind zone (sensor_blind_zone=True; use with --dr for BZ+DR)")
    ap.add_argument("--both-dr",    action="store_true",
                    help="Train each combo twice: base + DR")
    ap.add_argument("--tag",        type=str, default=None,
                    help="Custom output tag for model .zip (single-run only)")
    ap.add_argument("--dry-run",    action="store_true",
                    help="Print commands without executing")
    args = ap.parse_args()

    cfg = {
        **DEFAULTS,
        "total_timesteps": args.timesteps,
        "eval_freq":       args.eval_freq,
        "seed":            args.seed,
    }

    algos = args.algos or ALL_ALGOS
    rewards = args.rewards or REWARDS

    # Validate
    for a in algos:
        if a not in ALL_ALGOS:
            print(f"ERROR: unknown algo '{a}'. Choose from: {ALL_ALGOS}")
            sys.exit(1)
    for r in rewards:
        if r not in REWARDS:
            print(f"ERROR: unknown reward '{r}'. Choose from: {REWARDS}")
            sys.exit(1)

    # Build run matrix: list of (algo, reward, dr_flag, tag)
    run_matrix = []
    if args.both_dr:
        for reward in rewards:
            for algo in algos:
                run_matrix.append((algo, reward, False, args.blind_zone, None))
                run_matrix.append((algo, reward, True, args.blind_zone, None))
    else:
        for reward in rewards:
            for algo in algos:
                run_matrix.append((algo, reward, args.dr, args.blind_zone, args.tag))

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    batch_dir = TRAINING_DIR / "runs" / f"cont_batch_{stamp}"
    batch_dir.mkdir(parents=True, exist_ok=True)

    total = len(run_matrix)
    dr_str = "both" if args.both_dr else ("ON" if args.dr else "OFF")

    # -- banner --
    print()
    print("+===============================================================+")
    print("|         CONTINUOUS-ACTION  BATCH  TRAINING                   |")
    print("+===============================================================+")
    print(f"|  Algos      : {', '.join(algos):<47}|")
    print(f"|  Rewards    : {', '.join(rewards):<47}|")
    print(f"|  DR         : {dr_str:<47}|")
    print(f"|  Runs       : {total:<47}|")
    print(f"|  Timesteps  : {cfg['total_timesteps']:>12,}{'':<35}|")
    print(f"|  Eval freq  : {cfg['eval_freq']:>12,}{'':<35}|")
    print(f"|  Seed       : {cfg['seed']:<47}|")
    print(f"|  Batch dir  : runs/cont_batch_{stamp:<30}|")
    print(f"|  Models out : evaluation/models/cont/{'':<26}|")
    print(f"|  W&B        : {'enabled' if args.wandb else 'disabled':<47}|")
    print("+===============================================================+")

    results: list[dict] = []
    t_all = time.time()

    for idx, (algo, reward, dr_flag, bz_flag, out_tag) in enumerate(run_matrix, 1):
        info = _run_one(
            idx, total, algo, reward, batch_dir, cfg,
            dry_run=args.dry_run,
            enable_wandb=args.wandb,
            enable_dr=dr_flag,
            enable_blind_zone=bz_flag,
            output_tag=out_tag,
        )
        results.append(info)

    wall = time.time() - t_all

    if args.dry_run:
        print(f"\n  (dry run - nothing was executed)\n")
        return

    # -- persist JSON summary --
    summary = dict(
        timestamp=stamp,
        total_wall_time=str(timedelta(seconds=int(wall))),
        config=cfg,
        milestone_pcts=MILESTONE_PCTS,
        max_episode_reward=MAX_EP_REWARD,
        runs=results,
    )
    MODELS_OUT.mkdir(parents=True, exist_ok=True)
    summary_path = MODELS_OUT / "training_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)

    # -- pretty table --
    W = 140
    print(f"\n{'='*W}")
    print(f"  BATCH COMPLETE - wall time {timedelta(seconds=int(wall))}")
    print(f"{'='*W}")

    hdr = f"{'Algo':<8}{'Reward':<28}{'Envs':>4} {'Time':<13}"
    for pct in MILESTONE_PCTS:
        hdr += f"{'Eval ' + str(pct) + '%':>13}"
    for pct in MILESTONE_PCTS:
        hdr += f"{'Train ' + str(pct) + '%':>13}"
    hdr += "  Model"
    print(hdr)
    print("-" * W)

    for r in results:
        m = r.get("milestones", {})
        row = (
            f"{r['algo']:<8}{r['reward']:<28}{r.get('n_envs',''):>4} "
            f"{r.get('elapsed','-'):<13}"
        )
        for pct in MILESTONE_PCTS:
            v = m.get(f"eval_{pct}pct_step")
            row += f"{(f'{v:,}' if v else '-'):>13}"
        for pct in MILESTONE_PCTS:
            v = m.get(f"train_{pct}pct_step")
            row += f"{(f'{v:,}' if v else '-'):>13}"
        row += f"  {'OK' if r.get('model_path') else 'X'}"
        print(row)

    print(f"\n  Summary -> {summary_path.relative_to(PROJECT_ROOT)}")
    print(f"  Models  -> {MODELS_OUT.relative_to(PROJECT_ROOT)}/")
    print()


if __name__ == "__main__":
    main()
