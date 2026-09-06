#!/usr/bin/env python3
"""
Offline RL training for the ArcBall CONTINUOUS control task.

Supports CQL, IQL, TD3+BC, and BCQ via d3rlpy (Seno & Imai, 2022;
continuous action space).
Trains from arcball_real_demonstrations/arcball_post_recalib_flat.h5 with dense reward recomputation.

Continuous-action specifics:
  1. Continuous action space - uses CQL/IQL/TD3+BC/BCQ for float actions
  2. Dense reward recomputation - balanced_reward + dip_reward, clipped [-1,1]
  3. Proper episode segmentation with timeouts (not false terminals)
  4. Observation & action scalers for stable training
  5. Direct integration with eval pipeline

Quick start
-----------
  # CQL with defaults
  python offline_train_cont.py --data-path ../../data/dataset/arcball_real_demonstrations/arcball_post_recalib_flat.h5

  # IQL on GPU with W&B
  python offline_train_cont.py \\
      --algo iql --n-steps 500000 --gpu \\
      --wandb --wandb-project arcball-offline-cont

  # TD3+BC with custom reward weights
  python offline_train_cont.py \\
      --algo td3bc --dip-weight 0.3 --n-steps 300000

  # Dry-run to verify data loading
  python offline_train_cont.py --dry-run
"""

import argparse
import logging
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import h5py
import numpy as np
import torch
from sklearn.model_selection import train_test_split

from d3rlpy.dataset import MDPDataset
from d3rlpy.constants import ActionSpace
from d3rlpy.algos import (
    BCQ,
    BCQConfig,
    CQL,
    CQLConfig,
    IQL,
    IQLConfig,
    TD3PlusBC,
    TD3PlusBCConfig,
)
from d3rlpy.preprocessing import (
    StandardObservationScaler,
    MinMaxActionScaler,
    StandardRewardScaler,
)

# ---------------------------------------------------------------------------
# Optional W&B
# ---------------------------------------------------------------------------
try:
    import wandb
    _WANDB_AVAILABLE = True
except ImportError:
    _WANDB_AVAILABLE = False

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
def setup_logging(level: str = "INFO") -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s [%(levelname)-8s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        stream=sys.stdout,
    )

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------
def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

# ---------------------------------------------------------------------------
# System constants (hardcoded to keep this script self-contained)
# ---------------------------------------------------------------------------
CART_LIMIT = 0.7765
BALL_LIMIT = 0.0810
POS_LIMITS = [CART_LIMIT, BALL_LIMIT]

# State indices
CART_X = 0
CART_DOT = 1
BALL_X = 2
BALL_DOT = 3

# ---------------------------------------------------------------------------
# Reward functions (self-contained for portability)
# ---------------------------------------------------------------------------

def _balanced_reward(
    state: np.ndarray,
    action: float = 0.0,
    prev_action: float = 0.0,
    # shape params
    w_pos: float = 0.60,
    w_vel: float = 0.25,
    w_effort: float = 0.05,
    w_jerk: float = 0.05,
    w_boundary: float = 0.35,
) -> float:
    """Dense balanced reward - mirrors balancer/envs/rewards.py exactly."""
    theta = state[BALL_X]
    theta_dot = state[BALL_DOT]
    theta_abs = abs(theta)
    ball_limit = BALL_LIMIT

    sigma_outer = ball_limit / 4.0
    sigma_inner = ball_limit / 25.0
    sigma_vel_gate = ball_limit / 10.0

    # 1. Position: Dual Gaussian
    outer = np.exp(-theta**2 / (2 * sigma_outer**2))
    inner = np.exp(-theta**2 / (2 * sigma_inner**2))
    pos_reward = 0.6 * outer + 0.4 * inner

    # 2. Velocity: Normalized and gated
    vel_gate = np.exp(-theta**2 / (2 * sigma_vel_gate**2))
    max_vel = 2.0
    vel_normalized = min(theta_dot**2 / max_vel**2, 1.0)
    vel_pen = vel_gate * vel_normalized

    # 3. Ball boundary penalty
    boundary_ratio = theta_abs / ball_limit
    if boundary_ratio > 0.85:
        boundary_pen = w_boundary * ((boundary_ratio - 0.85) / 0.15)**2
    else:
        boundary_pen = 0.0

    # 3b. Cart wall penalty
    cart_pos = state[CART_X]
    cart_ratio = abs(cart_pos) / CART_LIMIT
    if cart_ratio > 0.85:
        into_wall = (cart_pos > 0 and action > 0) or (cart_pos < 0 and action < 0)
        if into_wall:
            cart_wall_pen = 0.2 * ((cart_ratio - 0.85) / 0.15)**2
        else:
            cart_wall_pen = 0.0
    else:
        cart_wall_pen = 0.0

    # 4. Effort/Jerk
    effort_pen = min(action**2, 1.0)
    jerk_pen = min((action - prev_action)**2, 1.0)

    reward = (
        w_pos * pos_reward
        - w_vel * vel_pen
        - w_effort * effort_pen
        - w_jerk * jerk_pen
        - boundary_pen
        - cart_wall_pen
    )
    return float(reward)


def _dip_reward(state: np.ndarray) -> float:
    """Binary reward: 1.0 if ball within 0.007 rad of center."""
    return 1.0 if abs(state[BALL_X]) <= 0.007 else 0.0


def compute_rewards(
    observations: np.ndarray,
    next_observations: np.ndarray,
    actions: np.ndarray,
    dip_weight: float = 1.0,
    w_pos: float = 0.60,
    w_vel: float = 0.25,
    w_effort: float = 0.05,
    w_jerk: float = 0.05,
    w_boundary: float = 0.35,
) -> np.ndarray:
    """
    Recompute dense rewards for the entire dataset (vectorized).

    reward = balanced_reward(next_state, action, prev_action) + dip_weight * dip_reward(next_state)
    Clipped to [-1, 1].

    Args:
        observations: (N, 4) current states
        next_observations: (N, 4) next states
        actions: (N,) continuous actions
        dip_weight: weight for binary dip reward bonus

    Returns:
        (N,) float32 reward array
    """
    n = len(observations)
    ball_limit = BALL_LIMIT

    # Extract columns from next_observations
    theta = next_observations[:, BALL_X]
    theta_dot = next_observations[:, BALL_DOT]
    cart_pos = next_observations[:, CART_X]
    theta_abs = np.abs(theta)

    # Gaussian shape params
    sigma_outer = ball_limit / 4.0
    sigma_inner = ball_limit / 25.0
    sigma_vel_gate = ball_limit / 10.0

    # 1. Position: Dual Gaussian
    outer = np.exp(-theta**2 / (2 * sigma_outer**2))
    inner = np.exp(-theta**2 / (2 * sigma_inner**2))
    pos_reward = 0.6 * outer + 0.4 * inner

    # 2. Velocity penalty (gated)
    vel_gate = np.exp(-theta**2 / (2 * sigma_vel_gate**2))
    vel_normalized = np.minimum(theta_dot**2 / 4.0, 1.0)  # max_vel=2.0
    vel_pen = vel_gate * vel_normalized

    # 3. Ball boundary penalty
    boundary_ratio = theta_abs / ball_limit
    boundary_pen = np.where(
        boundary_ratio > 0.85,
        w_boundary * ((boundary_ratio - 0.85) / 0.15)**2,
        0.0,
    )

    # 3b. Cart wall penalty
    cart_ratio = np.abs(cart_pos) / CART_LIMIT
    into_wall = ((cart_pos > 0) & (actions > 0)) | ((cart_pos < 0) & (actions < 0))
    cart_wall_pen = np.where(
        (cart_ratio > 0.85) & into_wall,
        0.2 * ((cart_ratio - 0.85) / 0.15)**2,
        0.0,
    )

    # 4. Effort & jerk penalties
    effort_pen = np.minimum(actions**2, 1.0)
    prev_actions = np.concatenate([[0.0], actions[:-1]])
    jerk_pen = np.minimum((actions - prev_actions)**2, 1.0)

    # Balanced reward
    rewards = (
        w_pos * pos_reward
        - w_vel * vel_pen
        - w_effort * effort_pen
        - w_jerk * jerk_pen
        - boundary_pen
        - cart_wall_pen
    )

    # Dip bonus
    dip_bonus = np.where(theta_abs <= 0.007, 1.0, 0.0)
    rewards += dip_weight * dip_bonus

    # Clip to [-1, 1]
    np.clip(rewards, -1.0, 1.0, out=rewards)
    return rewards.astype(np.float32)


# ---------------------------------------------------------------------------
# W&B Logger Adapter (reused from discrete script)
# ---------------------------------------------------------------------------

class WandbAdapter:
    """d3rlpy v2 LoggerAdapter forwarding metrics to W&B."""

    def __init__(self, algo, experiment_name: str, n_steps_per_epoch: int,
                 checkpoint_dir: Path) -> None:
        self._algo = algo
        self._experiment_name = experiment_name
        self._n_steps_per_epoch = n_steps_per_epoch
        self._checkpoint_dir = checkpoint_dir
        self._pending: Dict[str, Any] = {}
        wandb.config.update({
            "algo/class_name": type(algo).__name__,
            "algo/experiment_name": experiment_name,
            "algo/n_steps_per_epoch": n_steps_per_epoch,
        }, allow_val_change=True)

    def write_params(self, params: Dict[str, Any]) -> None:
        wandb.config.update(
            {f"algo/{k}": v for k, v in params.items()},
            allow_val_change=True,
        )

    def before_write_metric(self, epoch: int, step: int) -> None:
        self._pending = {"train/epoch": epoch}

    def write_metric(self, epoch: int, step: int, name: str, value: float) -> None:
        key = name if "/" in name else f"train/{name}"
        self._pending[key] = value

    def after_write_metric(self, epoch: int, step: int) -> None:
        if self._pending:
            wandb.log(self._pending, step=step)
            self._pending = {}

    def watch_model(self, epoch: int, step: int) -> None:
        pass

    def save_model(self, epoch: int, algo) -> None:
        ckpt_path = self._checkpoint_dir / f"checkpoint_epoch{epoch:04d}.d3"
        try:
            algo.save(str(ckpt_path))
            log.info("Checkpoint -> %s", ckpt_path)
        except Exception as exc:
            log.warning("Checkpoint at epoch %d failed: %s", epoch, exc)

    def close(self) -> None:
        wandb.finish()


@dataclass
class WandbAdapterFactory:
    """Factory for d3rlpy fit() logger_adapter kwarg."""
    checkpoint_dir: Path

    def create(self, algo, experiment_name: str, n_steps_per_epoch: int) -> WandbAdapter:
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        return WandbAdapter(algo, experiment_name, n_steps_per_epoch, self.checkpoint_dir)


# ---------------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------------

def load_dataset(
    path: Path,
    episode_length: int,
    test_size: float,
    seed: int,
    dip_weight: float,
    use_stored_reward: bool = False,
    top_pct: float = 1.0,
) -> Tuple[MDPDataset, MDPDataset, Dict[str, Any]]:
    """
    Load HDF5 and return train/val MDPDatasets with recomputed rewards.

    HDF5 layout (arcball_post_recalib_flat.h5):
        dataset[:, 0:4]  = prev state [cart_pos, cart_vel, ball_pos, ball_vel]
        dataset[:, 4]    = action (continuous, [-1, 1])
        dataset[:, 5:9]  = next state [cart_pos, cart_vel, ball_pos, ball_vel]
        dataset[:, 9]    = stored reward
        dataset[:, 10]   = stored dip_reward

    Returns:
        (train_ds, val_ds, stats_dict)
    """
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")

    log.info("Loading dataset from %s", path)
    with h5py.File(path, "r") as f:
        data = f["dataset"][:]

    n_raw = len(data)
    obs = data[:, 0:4].astype(np.float32)
    actions = data[:, 4:5].astype(np.float32)  # (N, 1) for continuous
    next_obs = data[:, 5:9].astype(np.float32)
    stored_reward = data[:, 9].astype(np.float32)
    stored_dip = data[:, 10].astype(np.float32)

    log.info("Raw transitions: %d", n_raw)
    log.info("Action range: [%.4f, %.4f], mean=%.4f, std=%.4f",
             actions.min(), actions.max(), actions.mean(), actions.std())
    log.info("Obs range per dim: min=%s, max=%s", obs.min(0), obs.max(0))

    # -- Reward recomputation --
    if use_stored_reward:
        log.info("Using stored reward (sparse). mean=%.4f", stored_reward.mean())
        rewards = stored_reward
    else:
        log.info("Recomputing dense reward: balanced + %.2f * dip_reward ...", dip_weight)
        rewards = compute_rewards(obs, next_obs, actions[:, 0], dip_weight=dip_weight)
        log.info("Recomputed reward stats: mean=%.4f, std=%.4f, min=%.4f, max=%.4f",
                 rewards.mean(), rewards.std(), rewards.min(), rewards.max())
        # Compare with stored
        dip_frac = stored_dip.mean()
        log.info("Stored dip fraction: %.2f%% (transitions with ball centered)", dip_frac * 100)

    # -- Episode segmentation --
    # Truncate to multiple of episode_length
    remainder = n_raw % episode_length
    if remainder != 0:
        n = n_raw - remainder
        log.warning("Truncating %d trailing transitions -> using %d", remainder, n)
        obs = obs[:n]
        actions = actions[:n]
        next_obs = next_obs[:n]
        rewards = rewards[:n]
    else:
        n = n_raw

    n_episodes = n // episode_length
    log.info("%d transitions | %d episodes (episode_length=%d)", n, n_episodes, episode_length)

    # Use timeouts (not terminals) - these are truncations, not true failures
    # True terminal = ball fell off arc (|ball_pos| > BALL_LIMIT)
    terminals = np.zeros(n, dtype=bool)
    timeouts = np.zeros(n, dtype=bool)

    for ep in range(n_episodes):
        ep_end = (ep + 1) * episode_length - 1
        # Check if ball is out of bounds at episode end
        ball_pos_end = abs(next_obs[ep_end, BALL_X])
        if ball_pos_end > BALL_LIMIT * 0.95:
            terminals[ep_end] = True
        else:
            timeouts[ep_end] = True

    log.info("Episodes: %d terminal, %d timeout",
             terminals.sum(), timeouts.sum())

    # -- Top-percentile episode filtering --
    # Keep only the best episodes by return (sum of rewards)
    if top_pct < 1.0:
        ep_returns = np.array([
            rewards[e * episode_length:(e + 1) * episode_length].sum()
            for e in range(n_episodes)
        ])
        threshold = np.percentile(ep_returns, (1.0 - top_pct) * 100)
        keep_mask = ep_returns >= threshold
        keep_eps = np.where(keep_mask)[0]
        n_keep = len(keep_eps)
        log.info("Top-%.0f%% filtering: keeping %d / %d episodes (return >= %.3f)",
                 top_pct * 100, n_keep, n_episodes, threshold)
        log.info("  Return stats of kept episodes: mean=%.3f, max=%.3f",
                 ep_returns[keep_mask].mean(), ep_returns[keep_mask].max())

        # Rebuild arrays with only kept episodes
        keep_idx = np.concatenate([
            np.arange(e * episode_length, (e + 1) * episode_length) for e in keep_eps
        ])
        obs = obs[keep_idx]
        actions = actions[keep_idx]
        next_obs = next_obs[keep_idx]
        rewards = rewards[keep_idx]

        # Rebuild terminal/timeout arrays
        n = len(keep_idx)
        n_episodes = n_keep
        terminals = np.zeros(n, dtype=bool)
        timeouts = np.zeros(n, dtype=bool)
        for ep in range(n_episodes):
            ep_end = (ep + 1) * episode_length - 1
            ball_pos_end = abs(next_obs[ep_end, BALL_X])
            if ball_pos_end > BALL_LIMIT * 0.95:
                terminals[ep_end] = True
            else:
                timeouts[ep_end] = True
        log.info("  After filtering: %d transitions, %d episodes", n, n_episodes)

    # -- Next-obs consistency check --
    # Within an episode, obs[i+1] should ≈ next_obs[i]
    check_len = min(episode_length - 1, 200)
    max_diff = float(np.abs(next_obs[:check_len] - obs[1:check_len + 1]).max())
    if max_diff > 1e-3:
        log.warning("next_obs consistency: max_diff=%.6f (>1e-3)", max_diff)
    else:
        log.info("next_obs consistency OK (max_diff=%.2e)", max_diff)

    # -- Episode-level train/val split --
    ep_idx = np.arange(n_episodes)
    train_eps, val_eps = train_test_split(
        ep_idx, test_size=test_size, random_state=seed, shuffle=True)
    train_eps, val_eps = np.sort(train_eps), np.sort(val_eps)

    def _ep_to_trans(ep_ids):
        return np.concatenate([
            np.arange(e * episode_length, (e + 1) * episode_length) for e in ep_ids
        ])

    train_idx = _ep_to_trans(train_eps)
    val_idx = _ep_to_trans(val_eps)

    log.info("Split -> train=%d (%d eps) | val=%d (%d eps) (%.0f%% val)",
             len(train_idx), len(train_eps), len(val_idx), len(val_eps), test_size * 100)

    def _make_mdp(idxs):
        return MDPDataset(
            observations=obs[idxs],
            actions=actions[idxs],
            rewards=rewards[idxs],
            terminals=terminals[idxs],
            timeouts=timeouts[idxs],
            action_space=ActionSpace.CONTINUOUS,
        )

    stats = {
        "n_transitions": n,
        "n_episodes": n_episodes,
        "n_train_eps": len(train_eps),
        "n_val_eps": len(val_eps),
        "reward_mean": float(rewards.mean()),
        "reward_std": float(rewards.std()),
        "action_mean": float(actions.mean()),
        "action_std": float(actions.std()),
        "dip_fraction": float(stored_dip.mean()) if not use_stored_reward else None,
    }

    return _make_mdp(train_idx), _make_mdp(val_idx), stats


# ---------------------------------------------------------------------------
# Evaluators
# ---------------------------------------------------------------------------

def build_evaluators(val_ds: MDPDataset) -> Dict:
    try:
        from d3rlpy.metrics import TDErrorEvaluator, AverageValueEstimationEvaluator
        val_episodes = val_ds.episodes
        return {
            "val_td_error": TDErrorEvaluator(episodes=val_episodes),
            "val_avg_q": AverageValueEstimationEvaluator(episodes=val_episodes),
        }
    except Exception as exc:
        log.warning("Could not build evaluators (%s).", exc)
        return {}


# ---------------------------------------------------------------------------
# Algorithm factory
# ---------------------------------------------------------------------------

_ALGO_REGISTRY = {
    "cql": (CQL, CQLConfig),
    "iql": (IQL, IQLConfig),
    "td3bc": (TD3PlusBC, TD3PlusBCConfig),
    "bcq": (BCQ, BCQConfig),
}


def build_algo(args: argparse.Namespace):
    if args.algo not in _ALGO_REGISTRY:
        raise ValueError(f"Unknown algo {args.algo!r}. Available: {list(_ALGO_REGISTRY)}")

    device = "cuda" if args.gpu and torch.cuda.is_available() else "cpu"
    if args.gpu and device == "cpu":
        log.warning("--gpu requested but CUDA unavailable; using CPU.")
    log.info("Compute device: %s", device)

    AlgoClass, ConfigClass = _ALGO_REGISTRY[args.algo]

    # Scalers for stable training
    obs_scaler = StandardObservationScaler() if args.use_scalers else None
    action_scaler = MinMaxActionScaler() if args.use_scalers else None
    reward_scaler = StandardRewardScaler() if args.use_reward_scaler else None

    # Shared config params
    shared = dict(
        batch_size=args.batch_size,
        gamma=args.gamma,
        observation_scaler=obs_scaler,
        action_scaler=action_scaler,
        reward_scaler=reward_scaler,
    )

    # Algorithm-specific params
    if args.algo == "cql":
        shared.update(dict(
            actor_learning_rate=args.actor_lr,
            critic_learning_rate=args.critic_lr,
            alpha_learning_rate=args.alpha_lr,
            n_critics=args.n_critics,
            conservative_weight=args.cql_weight,
            tau=args.tau,
        ))
    elif args.algo == "iql":
        shared.update(dict(
            actor_learning_rate=args.actor_lr,
            critic_learning_rate=args.critic_lr,
            n_critics=args.n_critics,
            expectile=args.iql_expectile,
            weight_temp=args.iql_weight_temp,
            tau=args.tau,
        ))
    elif args.algo == "td3bc":
        shared.update(dict(
            actor_learning_rate=args.actor_lr,
            critic_learning_rate=args.critic_lr,
            n_critics=args.n_critics,
            alpha=args.td3bc_alpha,
            tau=args.tau,
        ))
    elif args.algo == "bcq":
        shared.update(dict(
            actor_learning_rate=args.actor_lr,
            critic_learning_rate=args.critic_lr,
            imitator_learning_rate=args.imitator_lr,
            n_critics=args.n_critics,
            tau=args.tau,
            lam=args.bcq_lam,
            action_flexibility=args.bcq_action_flexibility,
        ))

    config = ConfigClass(**shared)
    algo = AlgoClass(config=config, device=device, enable_ddp=False)
    log.info("Instantiated %s (config: %s)", type(algo).__name__, config)
    return algo


# ---------------------------------------------------------------------------
# Model persistence
# ---------------------------------------------------------------------------

def save_model(algo, out_path: Path) -> Path:
    """Save using d3rlpy's native .save() method."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    algo.save(str(out_path))
    log.info("Model saved -> %s", out_path)
    return out_path


# ---------------------------------------------------------------------------
# W&B helpers
# ---------------------------------------------------------------------------

def init_wandb(args: argparse.Namespace):
    if not args.wandb:
        return None
    if not _WANDB_AVAILABLE:
        raise ImportError("wandb not installed. Run: pip install wandb")
    run = wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity or None,
        name=args.wandb_run_name or None,
        tags=args.wandb_tags or [],
        config=vars(args),
        resume="allow",
        dir=str(Path(args.save_dir)),
    )
    log.info("W&B run -> project=%s name=%s url=%s",
             args.wandb_project, run.name, run.url)
    return run


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train(args: argparse.Namespace) -> None:
    seed_everything(args.seed)

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = save_dir / "checkpoints"

    run = init_wandb(args)

    train_ds, val_ds, stats = load_dataset(
        path=Path(args.data_path),
        episode_length=args.episode_length,
        test_size=args.test_size,
        seed=args.seed,
        dip_weight=args.dip_weight,
        use_stored_reward=args.use_stored_reward,
        top_pct=args.top_pct,
    )

    if run is not None:
        wandb.config.update({"dataset": stats}, allow_val_change=True)

    if args.dry_run:
        log.info("--dry-run: skipping training.")
        if run is not None:
            wandb.finish()
        return

    algo = build_algo(args)
    log.info("Building model from dataset ...")
    algo.build_with_dataset(train_ds)

    evaluators = build_evaluators(val_ds)
    model_path = save_dir / f"cont_{args.algo}_model.d3"
    if args.model_path:
        model_path = Path(args.model_path)

    logger_adapter = WandbAdapterFactory(checkpoint_dir=checkpoint_dir) if run else None

    log.info("Training %s | n_steps=%d batch=%d γ=%.3f actor_lr=%g critic_lr=%g",
             args.algo.upper(), args.n_steps, args.batch_size,
             args.gamma, args.actor_lr, args.critic_lr)

    fit_kwargs: Dict[str, Any] = dict(
        n_steps=args.n_steps,
        n_steps_per_epoch=args.eval_interval,
        evaluators=evaluators or None,
    )
    if logger_adapter is not None:
        fit_kwargs["logger_adapter"] = logger_adapter

    algo.fit(train_ds, **fit_kwargs)

    # Final save
    save_model(algo, model_path)
    log.info("Training complete. Model -> %s", model_path)

    if run is not None:
        artifact = wandb.Artifact(
            name=f"cont_{args.algo}_model", type="model",
            metadata={"n_steps": args.n_steps, "algo": args.algo},
        )
        artifact.add_file(str(model_path))
        run.log_artifact(artifact)
        wandb.finish()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    g = p.add_argument_group("data")
    g.add_argument("--data-path",
                   default="../../data/dataset/arcball_real_demonstrations/arcball_post_recalib_flat.h5",
                   metavar="PATH")
    g.add_argument("--episode-length", type=int, default=600,
                   help="Steps per episode (600 = 30s @ 20Hz, matches BalancerSim-v1)")
    g.add_argument("--test-size", type=float, default=0.05, metavar="FRAC")

    g = p.add_argument_group("reward")
    g.add_argument("--dip-weight", type=float, default=0.2,
                   help="Weight for binary dip reward bonus (default: 0.2)")
    g.add_argument("--use-stored-reward", action="store_true",
                   help="Use stored sparse reward instead of recomputing.")
    g.add_argument("--top-pct", type=float, default=1.0,
                   help="Keep only top N%% of episodes by return (default: 1.0 = no filtering)")

    g = p.add_argument_group("training")
    g.add_argument("--algo", choices=list(_ALGO_REGISTRY), default="cql")
    g.add_argument("--n-steps", type=int, default=500_000)
    g.add_argument("--batch-size", type=int, default=256)
    g.add_argument("--gamma", type=float, default=0.99)
    g.add_argument("--actor-lr", type=float, default=3e-4, metavar="LR")
    g.add_argument("--critic-lr", type=float, default=3e-4, metavar="LR")
    g.add_argument("--alpha-lr", type=float, default=3e-4, metavar="LR",
                   help="CQL alpha auto-tune learning rate")
    g.add_argument("--imitator-lr", type=float, default=1e-3, metavar="LR",
                   help="BCQ imitator learning rate")
    g.add_argument("--n-critics", type=int, default=2)
    g.add_argument("--tau", type=float, default=0.005,
                   help="Soft target update rate")
    g.add_argument("--eval-interval", type=int, default=10_000,
                   help="Steps between evaluations (= epoch size)")

    g = p.add_argument_group("algorithm-specific")
    g.add_argument("--cql-weight", type=float, default=5.0,
                   help="CQL conservative_weight (default: 5.0)")
    g.add_argument("--iql-expectile", type=float, default=0.7,
                   help="IQL expectile τ (default: 0.7)")
    g.add_argument("--iql-weight-temp", type=float, default=3.0,
                   help="IQL advantage weight temperature β (default: 3.0)")
    g.add_argument("--td3bc-alpha", type=float, default=2.5,
                   help="TD3+BC α (BC regularization strength)")
    g.add_argument("--bcq-lam", type=float, default=0.75,
                   help="BCQ λ (perturbation scale)")
    g.add_argument("--bcq-action-flexibility", type=float, default=0.05,
                   help="BCQ action perturbation range")

    g = p.add_argument_group("scalers")
    g.add_argument("--use-scalers", action="store_true", default=True,
                   help="Use observation/action scalers (default: True)")
    g.add_argument("--no-scalers", dest="use_scalers", action="store_false")
    g.add_argument("--use-reward-scaler", action="store_true", default=False,
                   help="Normalize rewards with StandardRewardScaler")

    g = p.add_argument_group("weights & biases")
    g.add_argument("--wandb", action="store_true", default=False)
    g.add_argument("--wandb-project", default="arcball-offline-cont", metavar="PROJECT")
    g.add_argument("--wandb-entity", default=None, metavar="ENTITY")
    g.add_argument("--wandb-run-name", default=None, metavar="NAME")
    g.add_argument("--wandb-tags", nargs="*", default=[], metavar="TAG")

    g = p.add_argument_group("misc")
    g.add_argument("--seed", type=int, default=42)
    g.add_argument("--gpu", action="store_true", default=False)
    g.add_argument("--save-dir", default=".", metavar="DIR")
    g.add_argument("--model-path", default=None, metavar="PATH",
                   help="Explicit output path (default: <save-dir>/cont_<algo>_model.d3)")
    g.add_argument("--log-level", choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                   default="INFO")
    g.add_argument("--dry-run", action="store_true", default=False)

    return p.parse_args(argv)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    args = parse_args()
    setup_logging(args.log_level)
    log.info(
        "Run config:\n%s",
        "\n".join(f"  {k:<30s} = {v}" for k, v in sorted(vars(args).items())),
    )
    train(args)
