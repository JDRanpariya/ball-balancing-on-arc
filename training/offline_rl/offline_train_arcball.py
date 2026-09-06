#!/usr/bin/env python3
"""
Offline RL training for the ArcBall discrete control task.

Supports DiscreteCQL and DiscreteBCQ via d3rlpy (Seno & Imai, 2022;
implementing CQL, Kumar et al. 2020, and BCQ, Fujimoto et al. 2019), with
optional Weights & Biases experiment tracking.

Quick start
-----------
  # CQL with defaults, no W&B
  python offline_train_arcball.py

  # CQL with W&B logging
  python offline_train_arcball.py --wandb --wandb-project arcball-offline

  # BCQ on GPU, 100k steps, full W&B config
  python offline_train_arcball.py \\
      --algo bcq --n-steps 100000 --gpu \\
      --wandb --wandb-project arcball-offline \\
      --wandb-entity my-team --wandb-run-name bcq_baseline \\
      --wandb-tags bcq random_data

  # Explicit model output path
  python offline_train_arcball.py --model-path ./runs/cql_v1.pt

  # Dry-run to verify data loading only
  python offline_train_arcball.py --dry-run
"""

import argparse
import logging
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import h5py
import numpy as np
import torch
from sklearn.model_selection import train_test_split

from d3rlpy.dataset import MDPDataset
from d3rlpy.algos import (
    DiscreteBCQ,
    DiscreteBCQConfig,
    DiscreteCQL,
    DiscreteCQLConfig,
)

# ---------------------------------------------------------------------------

# Optional W&B import

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
    log.debug("Seeded everything with seed=%d", seed)


# ---------------------------------------------------------------------------

# W&B logger adapter for d3rlpy

# ---------------------------------------------------------------------------

class WandbAdapter:
    """
    d3rlpy LoggerAdapter that forwards all training metrics to W&B.

    Full d3rlpy v2 adapter protocol (every method the logger calls)
    ---------------------------------------------------------------
    write_params(params)
    before_write_metric(epoch, step)
    write_metric(epoch, step, name, value)
    after_write_metric(epoch, step)
    watch_model(epoch, step)           <- called after metrics are committed
    save_model(epoch, algo)            <- called by d3rlpy's internal checkpointing
    close()
    """

    def __init__(
        self,
        algo,
        experiment_name: str,
        n_steps_per_epoch: int,
        checkpoint_dir: Path,
    ) -> None:
        self._algo               = algo
        self._experiment_name    = experiment_name
        self._n_steps_per_epoch  = n_steps_per_epoch
        self._checkpoint_dir     = checkpoint_dir
        self._pending: Dict[str, Any] = {}

        wandb.config.update(
            {
                "algo/class_name":        type(algo).__name__,
                "algo/experiment_name":   experiment_name,
                "algo/n_steps_per_epoch": n_steps_per_epoch,
            },
            allow_val_change=True,
        )
        log.info(
            "WandbAdapter ready  experiment=%s  steps_per_epoch=%d",
            experiment_name, n_steps_per_epoch,
        )

    # --- d3rlpy LoggerAdapter protocol ---

    def write_params(self, params: Dict[str, Any]) -> None:
        """Forward algo internals (network sizes etc.) to W&B config."""
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
        """Flush all buffered metrics in ONE wandb.log call to keep x-axes aligned."""
        if self._pending:
            wandb.log(self._pending, step=step)
            self._pending = {}

    def watch_model(self, epoch: int, step: int) -> None:
        """
        Called by d3rlpy once per epoch after metrics are committed.
        Registers gradient histogram logging on the first epoch only.
        """
        if epoch == 1:
            try:
                impl         = getattr(self._algo, "_impl", None)
                torch_module = getattr(impl, "q_function", None) if impl else None
                if torch_module is not None:
                    wandb.watch(
                        torch_module,
                        log="gradients",
                        log_freq=self._n_steps_per_epoch,
                    )
                    log.info("wandb.watch() registered on q_function.")
                else:
                    log.debug("watch_model: q_function not found - skipping wandb.watch().")
            except Exception as exc:
                log.debug("watch_model: wandb.watch() failed (%s) - continuing.", exc)

    def save_model(self, epoch: int, algo) -> None:
        """
        Called by d3rlpy's internal logger after every epoch as a checkpoint.

        Signature is fixed by d3rlpy:  save_model(epoch: int, algo: AlgoBase)

        We delegate to the module-level _persist_model() helper so the
        checkpoint lands in self._checkpoint_dir with an epoch suffix,
        and we upload it as a W&B artifact so every checkpoint is versioned.
        """
        ckpt_path = self._checkpoint_dir / f"checkpoint_epoch{epoch:04d}.pt"
        try:
            _persist_model(algo, ckpt_path)
            log.info("Checkpoint saved -> %s", ckpt_path)
            # Upload checkpoint as a versioned W&B artifact
            if wandb.run is not None:
                artifact = wandb.Artifact(
                    name=f"{self._experiment_name}_checkpoint",
                    type="model-checkpoint",
                    metadata={"epoch": epoch},
                )
                artifact.add_file(str(ckpt_path))
                wandb.log_artifact(artifact)
        except Exception as exc:
            # Checkpoint failure must NEVER kill training
            log.warning("Checkpoint at epoch %d failed: %s", epoch, exc)

    def close(self) -> None:
        wandb.finish()
        log.info("W&B run finished.")


@dataclass
class WandbAdapterFactory:
    """
    Factory consumed by d3rlpy fit() via the ``logger_adapter`` kwarg.

    d3rlpy v2 calls:
        factory.create(algo, experiment_name, n_steps_per_epoch)
    """
    checkpoint_dir: Path

    def create(
        self,
        algo,
        experiment_name: str,
        n_steps_per_epoch: int,
    ) -> WandbAdapter:
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        return WandbAdapter(
            algo=algo,
            experiment_name=experiment_name,
            n_steps_per_epoch=n_steps_per_epoch,
            checkpoint_dir=self.checkpoint_dir,
        )


# ---------------------------------------------------------------------------

# Model persistence  (module-level, used by both WandbAdapter and train())

# ---------------------------------------------------------------------------

def _persist_model(algo, out_path: Path) -> Path:
    """
    Try every known d3rlpy save API in order.  Returns the path saved to.

    d3rlpy v2  ->  algo.save_model(path)   weights-only .pt
    fallback   ->  torch.save(_impl, path) always works if model is built
    """
    save_methods = ["save", "save_model", "save_policy"]
    available    = {m: hasattr(algo, m) for m in save_methods}
    log.info("Save-method availability: %s", available)
    log.info("Saving model -> %s", out_path)

    # Attempt 1 - d3rlpy v2 algo.save()
    if available["save"]:
        try:
            algo.save(str(out_path))
            log.info("Saved via algo.save()")
            return out_path
        except Exception as exc:
            log.warning("algo.save() failed: %s", exc)

    # Attempt 2 - d3rlpy v1 algo.save_model()
    if available["save_model"]:
        try:
            algo.save_model(str(out_path))
            log.info("Saved via algo.save_model()")
            return out_path
        except Exception as exc:
            log.warning("algo.save_model() failed: %s", exc)

    # Attempt 3 - raw torch.save on _impl
    impl = getattr(algo, "_impl", None)
    if impl is not None:
        try:
            torch.save(impl, str(out_path))
            log.info("Saved via torch.save(_impl)")
            return out_path
        except Exception as exc:
            log.warning("torch.save(_impl) failed: %s", exc)

    raise RuntimeError(
        "All save attempts failed.\n"
        f"  algo type      : {type(algo)}\n"
        f"  save-like attrs: {[a for a in dir(algo) if 'save' in a.lower()]}"
    )


def resolve_model_path(args: argparse.Namespace, save_dir: Path) -> Path:
    """Return the final model output path from CLI args."""
    if args.model_path:
        p = Path(args.model_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        return p
    return save_dir / f"discrete_{args.algo}_model.pt"


# ---------------------------------------------------------------------------

# W&B initialisation helpers

# ---------------------------------------------------------------------------

def init_wandb(args: argparse.Namespace) -> Optional[Any]:
    if not args.wandb:
        return None
    if not _WANDB_AVAILABLE:
        raise ImportError("wandb not installed. Run: pip install wandb")

    run = wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity  or None,
        name=args.wandb_run_name  or None,
        tags=args.wandb_tags      or [],
        config=vars(args),
        resume="allow",
        dir=str(Path(args.save_dir)),
    )
    log.info(
        "W&B run initialised -> project=%s  name=%s  url=%s",
        args.wandb_project, run.name, run.url,
    )
    return run


def log_model_artifact(
    run: Any,
    model_path: Path,
    algo_name: str,
    metadata: Optional[Dict[str, Any]] = None,
) -> None:
    if run is None:
        return
    artifact = wandb.Artifact(
        name=f"discrete_{algo_name}_model",
        type="model",
        description=f"Final model - Discrete{algo_name.upper()}",
        metadata=metadata or {},
    )
    target = Path(model_path)
    if target.is_dir():
        artifact.add_dir(str(target))
    else:
        artifact.add_file(str(target))
    run.log_artifact(artifact)
    log.info("Model uploaded as W&B artifact -> %s", artifact.name)


# ---------------------------------------------------------------------------

# Dataset

# ---------------------------------------------------------------------------

def load_dataset(
    path: Path,
    episode_length: int,
    test_size: float,
    seed: int,
    n_actions: int,
) -> Tuple[MDPDataset, MDPDataset]:
    """
    Read the HDF5 file and return train / val MDPDatasets.

    HDF5 column layout
    ------------------
    [:, 0:4]  observations
    [:, 4]    discrete action
    [:, 5:9]  next observations  (consistency check only)
    [:, 9]    reward
    """
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")

    log.info("Loading dataset from %s", path)
    with h5py.File(path, "r") as f:
        data = f["dataset"][:]

    n_raw    = len(data)
    obs      = data[:, 0:4].astype(np.float32)
    acts     = data[:, 4].astype(np.int64)
    next_obs = data[:, 5:9].astype(np.float32)
    rwds     = data[:, 9].astype(np.float32)

    # Handle incomplete trailing episode
    remainder = n_raw % episode_length
    if remainder != 0:
        n = n_raw - remainder
        log.warning(
            "Truncating %d trailing transitions (not divisible by %d) -> using %d.",
            remainder, episode_length, n,
        )
        obs = obs[:n]; acts = acts[:n]; next_obs = next_obs[:n]; rwds = rwds[:n]
    else:
        n = n_raw

    n_episodes = n // episode_length
    log.info("%d transitions  |  %d episodes  (episode_length=%d)", n, n_episodes, episode_length)

    # Terminal flags
    terminals = np.zeros(n, dtype=bool)
    terminals[(episode_length - 1)::episode_length] = True
    assert terminals.sum() == n_episodes

    # next_obs consistency check (within-episode only)
    check_len = min(episode_length - 1, 200)
    max_diff  = float(np.abs(next_obs[:check_len] - obs[1:check_len + 1]).max())
    if max_diff > 1e-4:
        log.warning("next_obs consistency check FAILED (max_diff=%.6f) - check pipeline.", max_diff)
    else:
        log.info("next_obs consistency check passed (max_abs_diff=%.2e).", max_diff)

    # Action range validation
    max_act, min_act = int(acts.max()), int(acts.min())
    if min_act < 0:
        raise ValueError(f"Negative action index {min_act} in data.")
    if max_act >= n_actions:
        raise ValueError(f"Action {max_act} ≥ --n-actions={n_actions}. Pass --n-actions {max_act+1}.")
    log.info("action_size=%d  (range in data: %d … %d)", n_actions, min_act, max_act)

    # Episode-level split
    ep_idx = np.arange(n_episodes)
    train_eps, val_eps = train_test_split(ep_idx, test_size=test_size, random_state=seed, shuffle=True)
    train_eps, val_eps = np.sort(train_eps), np.sort(val_eps)

    def _ep_to_trans(ep_ids):
        return np.concatenate([
            np.arange(e * episode_length, (e + 1) * episode_length) for e in ep_ids
        ])

    train_idx, val_idx = _ep_to_trans(train_eps), _ep_to_trans(val_eps)
    log.info(
        "Split -> train=%d transitions (%d eps) | val=%d transitions (%d eps)  (%.0f%% val)",
        len(train_idx), len(train_eps), len(val_idx), len(val_eps), test_size * 100,
    )

    def _make_mdp(idxs):
        return MDPDataset(
            observations=obs[idxs],
            actions=acts[idxs],
            rewards=rwds[idxs],
            terminals=terminals[idxs],
            action_size=n_actions,
        )

    return _make_mdp(train_idx), _make_mdp(val_idx)


# ---------------------------------------------------------------------------

# Evaluators

# ---------------------------------------------------------------------------

def build_evaluators(val_ds: MDPDataset) -> Dict:
    try:
        from d3rlpy.metrics import TDErrorEvaluator, AverageValueEstimationEvaluator
        val_episodes = val_ds.episodes
        evaluators = {
            "val_td_error": TDErrorEvaluator(episodes=val_episodes),
            "val_avg_q":    AverageValueEstimationEvaluator(episodes=val_episodes),
        }
        log.info("Validation evaluators: %s", list(evaluators))
        return evaluators
    except Exception as exc:
        log.warning("Could not build evaluators (%s). No val metrics.", exc)
        return {}


# ---------------------------------------------------------------------------

# Algorithm factory

# ---------------------------------------------------------------------------

_ALGO_REGISTRY = {
    "cql": (DiscreteCQL, DiscreteCQLConfig),
    "bcq": (DiscreteBCQ, DiscreteBCQConfig),
}


def build_algo(args: argparse.Namespace):
    if args.algo not in _ALGO_REGISTRY:
        raise ValueError(f"Unknown algo {args.algo!r}. Available: {list(_ALGO_REGISTRY)}")

    device = "cuda" if args.gpu and torch.cuda.is_available() else "cpu"
    if args.gpu and device == "cpu":
        log.warning("--gpu requested but CUDA unavailable; using CPU.")
    log.info("Compute device: %s", device)

    AlgoClass, ConfigClass = _ALGO_REGISTRY[args.algo]
    shared = dict(
        batch_size=args.batch_size,
        gamma=args.gamma,
        learning_rate=args.learning_rate,
        n_critics=args.n_critics,
        target_update_interval=args.target_update_interval,
    )
    extra = (
        dict(alpha=args.cql_alpha)                                              if args.algo == "cql" else
        dict(action_flexibility=args.bcq_action_flexibility, beta=args.bcq_beta) if args.algo == "bcq" else
        {}
    )
    config = ConfigClass(**shared, **extra)
    algo   = AlgoClass(config=config, enable_ddp=False, device=device)
    log.info("Instantiated %s", type(algo).__name__)
    return algo


# ---------------------------------------------------------------------------

# Training

# ---------------------------------------------------------------------------

def train(args: argparse.Namespace) -> None:
    seed_everything(args.seed)

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # Checkpoint sub-directory used by WandbAdapter.save_model()
    checkpoint_dir = save_dir / "checkpoints"

    run = init_wandb(args)

    train_ds, val_ds = load_dataset(
        path=Path(args.data_path),
        episode_length=args.episode_length,
        test_size=args.test_size,
        seed=args.seed,
        n_actions=args.n_actions,
    )

    if run is not None:
        wandb.config.update({
            "dataset/n_train_transitions": len(train_ds.episodes) * args.episode_length,
            "dataset/n_val_transitions":   len(val_ds.episodes)   * args.episode_length,
            "dataset/n_train_episodes":    len(train_ds.episodes),
            "dataset/n_val_episodes":      len(val_ds.episodes),
        }, allow_val_change=True)

    if args.dry_run:
        log.info("--dry-run set: skipping training.")
        if run is not None:
            wandb.finish()
        return

    algo = build_algo(args)
    log.info("Building model internals from dataset …")
    algo.build_with_dataset(train_ds)

    evaluators   = build_evaluators(val_ds)
    model_path   = resolve_model_path(args, save_dir)

    # WandbAdapterFactory carries checkpoint_dir so WandbAdapter.save_model()
    # knows where to write per-epoch checkpoints
    logger_adapter = WandbAdapterFactory(checkpoint_dir=checkpoint_dir) if run is not None else None

    log.info(
        "Training %s  |  n_steps=%d  batch=%d  γ=%.3f  lr=%g",
        args.algo.upper(), args.n_steps, args.batch_size, args.gamma, args.learning_rate,
    )
    fit_kwargs: Dict[str, Any] = dict(
        n_steps=args.n_steps,
        n_steps_per_epoch=args.eval_interval,
        evaluators=evaluators or None,
    )
    if logger_adapter is not None:
        fit_kwargs["logger_adapter"] = logger_adapter

    algo.fit(train_ds, **fit_kwargs)

    # Final model save
    try:
        _persist_model(algo, model_path)
    except RuntimeError as exc:
        log.error("Final model save failed:\n%s", exc)
        if run is not None:
            wandb.finish()
        return

    if run is not None:
        log_model_artifact(
            run=run,
            model_path=model_path,
            algo_name=args.algo,
            metadata=dict(
                n_steps=args.n_steps, algo=args.algo,
                n_actions=args.n_actions, gamma=args.gamma,
                batch_size=args.batch_size,
            ),
        )
        if wandb.run is not None:
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
                   default="../data/dataset/archive/dataset_arcball_600k.h5", metavar="PATH")
    g.add_argument("--episode-length", type=int, default=500)
    g.add_argument("--test-size", type=float, default=0.05, metavar="FRAC")
    g.add_argument("--n-actions", type=int, default=8, metavar="N")

    g = p.add_argument_group("training")
    g.add_argument("--algo", choices=list(_ALGO_REGISTRY), default="cql")
    g.add_argument("--n-steps", type=int, default=200_000)
    g.add_argument("--batch-size", type=int, default=256)
    g.add_argument("--gamma", type=float, default=0.99)
    g.add_argument("--learning-rate", type=float, default=6.25e-5, metavar="LR")
    g.add_argument("--n-critics", type=int, default=1)
    g.add_argument("--target-update-interval", type=int, default=8_000)
    g.add_argument("--eval-interval", type=int, default=10_000)

    g = p.add_argument_group("algorithm-specific")
    g.add_argument("--cql-alpha", type=float, default=1.0, metavar="α")
    g.add_argument("--bcq-action-flexibility", type=float, default=0.3, metavar="τ")
    g.add_argument("--bcq-beta", type=float, default=0.5, metavar="β")

    g = p.add_argument_group("weights & biases")
    g.add_argument("--wandb", action="store_true", default=False)
    g.add_argument("--wandb-project", default="arcball-offline-rl", metavar="PROJECT")
    g.add_argument("--wandb-entity", default=None, metavar="ENTITY")
    g.add_argument("--wandb-run-name", default=None, metavar="NAME")
    g.add_argument("--wandb-tags", nargs="*", default=[], metavar="TAG")

    g = p.add_argument_group("misc")
    g.add_argument("--seed", type=int, default=42)
    g.add_argument("--gpu", action="store_true", default=False)
    g.add_argument("--save-dir", default=".", metavar="DIR")
    g.add_argument(
        "--model-path",
        default=None,
        metavar="PATH",
        help=(
            "Explicit path for the final saved model, e.g. ./runs/cql.pt  "
            "Defaults to <save-dir>/discrete_<algo>_model.pt"
        ),
    )
    g.add_argument("--log-level", choices=["DEBUG", "INFO", "WARNING", "ERROR"], default="INFO")
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
