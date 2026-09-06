"""
predictor.py  -  WorldModelPredictor
=====================================
Inference wrapper for the trained LSTM world model.

Handles:
- Loading model checkpoint (.pth)
- Loading normalisation statistics (.npz)
- Normalising raw states and denormalising predicted deltas
- Device placement (CPU / CUDA)
- Single-step and multi-step prediction interfaces

Usage:
    >>> from balancer.world_model import WorldModelPredictor
    >>> predictor = WorldModelPredictor(
    ...     checkpoint_path="evaluation/models/world_model/v1/best_model.pth",
    ...     device="cpu",
    ... )
    >>> # Single-step prediction
    >>> next_state = predictor.predict(state, action)
    >>> # Access underlying model for MPPI rollouts
    >>> delta, hc = predictor.model(x_normalised, hc)
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import torch

from balancer.core.device import resolve_device

from .model import LSTMWorldModel, WorldModelConfig


class WorldModelPredictor:
    """Inference wrapper for a trained LSTMWorldModel.

    Loads a checkpoint and normalisation statistics, then provides methods
    for single-step and batched prediction with automatic normalisation.

    The predictor exposes `.model` for direct access (used by MPPIController
    for batched GPU rollouts), plus convenience methods that handle
    normalisation end-to-end.

    Args:
        checkpoint_path: Path to the `.pth` checkpoint file.
        stats_path: Path to normalisation stats `.npz` file. If None,
            looks for `norm_stats.npz` in the same directory as the checkpoint.
        device: Torch device string ('cpu' or 'cuda').
        config: Optional WorldModelConfig override. If None, attempts to
            load from checkpoint metadata, else uses defaults.

    Attributes:
        model: The underlying LSTMWorldModel (in eval mode).
        device: Torch device.
        s_mean: State normalisation mean (4,).
        s_std: State normalisation std (4,).
        delta_mean: Delta normalisation mean (4,).
        delta_std: Delta normalisation std (4,).
    """

    def __init__(
        self,
        checkpoint_path: str,
        stats_path: Optional[str] = None,
        device: str = "cuda",
        config: Optional[WorldModelConfig] = None,
    ):
        self.device = resolve_device(device)
        checkpoint_path = Path(checkpoint_path)

        # -- Load checkpoint -------------------------------------------
        ckpt = torch.load(
            checkpoint_path, map_location=self.device, weights_only=False)

        # Resolve config
        if config is not None:
            cfg = config
        elif "config" in ckpt:
            cfg = WorldModelConfig(**ckpt["config"])
        else:
            cfg = WorldModelConfig()

        self._cfg = cfg
        self.model = LSTMWorldModel(cfg).to(self.device)

        # Load weights - handle both raw state_dict and wrapped checkpoint
        if "model_state_dict" in ckpt:
            self.model.load_state_dict(ckpt["model_state_dict"])
        else:
            self.model.load_state_dict(ckpt)

        self.model.eval()

        # -- Load normalisation stats ----------------------------------
        if stats_path is None:
            stats_path = checkpoint_path.parent / "norm_stats.npz"
        else:
            stats_path = Path(stats_path)

        if not stats_path.exists():
            raise FileNotFoundError(
                f"Normalisation stats not found at {stats_path}. "
                f"Train the model with stats export enabled."
            )

        stats = np.load(stats_path)
        self.s_mean = torch.tensor(
            stats["s_mean"].flatten(), dtype=torch.float32, device=self.device)
        self.s_std = torch.tensor(
            stats["s_std"].flatten(), dtype=torch.float32, device=self.device)
        self.delta_mean = torch.tensor(
            stats["delta_mean"].flatten(), dtype=torch.float32, device=self.device)
        self.delta_std = torch.tensor(
            stats["delta_std"].flatten(), dtype=torch.float32, device=self.device)

        # Action normalisation (v2+). For v1 models without these, action is passed raw.
        if "a_mean" in stats and "a_std" in stats:
            self.a_mean = torch.tensor(
                stats["a_mean"].flatten(), dtype=torch.float32, device=self.device)
            self.a_std = torch.tensor(
                stats["a_std"].flatten(), dtype=torch.float32, device=self.device)
            self._normalise_action = True
        else:
            self._normalise_action = False

    # -- Normalisation helpers -------------------------------------------------

    def normalise_state(self, state: torch.Tensor) -> torch.Tensor:
        """Normalise raw state tensor using training statistics.

        Args:
            state: (..., 4) raw state tensor.

        Returns:
            Normalised state tensor of same shape.
        """
        return (state - self.s_mean) / self.s_std

    def normalise_action(self, action: torch.Tensor) -> torch.Tensor:
        """Normalise raw action tensor.

        For v2+ models: (action - a_mean) / a_std
        For v1 models: returns action unchanged (no normalisation was used).

        Args:
            action: (..., 1) raw action tensor.

        Returns:
            Normalised action tensor.
        """
        if self._normalise_action:
            return (action - self.a_mean) / self.a_std
        return action

    def denormalise_delta(self, delta_norm: torch.Tensor) -> torch.Tensor:
        """Convert normalised delta prediction back to real units.

        Args:
            delta_norm: (..., 4) normalised delta tensor.

        Returns:
            Delta in real physical units.
        """
        return delta_norm * self.delta_std + self.delta_mean

    # -- Single-step prediction ------------------------------------------------

    @torch.no_grad()
    def predict(self, state: np.ndarray, action: float) -> np.ndarray:
        """Predict next state from raw state and continuous action.

        Handles normalisation internally - pass raw physical-unit state.

        Args:
            state: (4,) numpy array [cart_pos, cart_vel, ball_pos, ball_vel].
            action: Continuous action scalar in [-0.9, 0.9].

        Returns:
            (4,) numpy array - predicted next state in physical units.
        """
        s = torch.tensor(state, dtype=torch.float32,
                         device=self.device).unsqueeze(0)  # (1, 4)
        a = torch.tensor([[action]], dtype=torch.float32,
                         device=self.device)  # (1, 1)

        s_norm = self.normalise_state(s)
        a_norm = self.normalise_action(a)
        x = torch.cat([s_norm, a_norm], dim=-1).unsqueeze(1)  # (1, 1, 5)

        delta_norm, _ = self.model(x)
        delta = self.denormalise_delta(delta_norm[:, 0, :])  # (1, 4)

        next_state = s + delta
        return next_state.squeeze(0).cpu().numpy()

    # -- Batched prediction for MPPI -------------------------------------------

    @torch.no_grad()
    def predict_batch(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        hc: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Batched single-step prediction for MPPI rollouts.

        Handles normalisation internally - pass raw states.

        Args:
            states: (B, 4) raw state tensor on self.device.
            actions: (B, 1) continuous action tensor on self.device.
            hc: Optional LSTM hidden state tuple.

        Returns:
            next_states: (B, 4) predicted next states in physical units.
            (h, c): Updated hidden state.
        """
        s_norm = self.normalise_state(states)
        a_norm = self.normalise_action(actions)
        x = torch.cat([s_norm, a_norm], dim=-1).unsqueeze(1)  # (B, 1, 5)

        delta_norm, hc = self.model(x, hc)
        delta = self.denormalise_delta(delta_norm[:, 0, :])  # (B, 4)

        next_states = states + delta
        return next_states, hc

    def __repr__(self) -> str:
        return (
            f"WorldModelPredictor("
            f"hidden={self._cfg.hidden_dim}, "
            f"layers={self._cfg.num_layers}, "
            f"device={self.device})"
        )
