"""Observation-augmenting wrappers for the Balancer environment."""

from __future__ import annotations

from collections import deque
from typing import Optional

import numpy as np
import gymnasium as gym
from gymnasium.spaces import Box

from balancer.core.params import DR_RANGES

# Observable params only (exclude dip geometry - not useful for policy)
PARAM_RANGES: dict[str, tuple[float, float]] = {
    k: DR_RANGES[k]
    for k in [
        "gravity",
        "arc_radius",
        "friction_coeff_cart",
        "mu_ball_rolling",
        "mu_ball_viscous",
        "tau_v",
    ]
}


class ParameterAugmentedObs(gym.ObservationWrapper):
    """Appends current dynamics parameters to the observation vector.

    Replaces the near-identical ``ParamInjectionWrapper`` and
    ``ParameterConditioningWrapper`` - both were doing the same thing.
    The wrapped environment must expose ``env.current_params`` as a ``dict``
    that is populated during ``reset()``.

    Args:
        env:        Environment to wrap.
        param_keys: Keys to pull from ``current_params``.
                    Defaults to every key defined in :data:`PARAM_RANGES`.

    Raises:
        ValueError: If any key in *param_keys* is absent from :data:`PARAM_RANGES`.

    Example::

        env = BalancerSim()
        env = ParameterAugmentedObs(env)                       # all params
        env = ParameterAugmentedObs(env, ["gravity", "ball_mass"])  # subset
    """

    def __init__(
        self,
        env: gym.Env,
        param_keys: list[str] | None = None,
    ) -> None:
        super().__init__(env)

        self.param_keys: list[str] = param_keys or list(PARAM_RANGES.keys())

        unknown = set(self.param_keys) - PARAM_RANGES.keys()
        if unknown:
            raise ValueError(
                f"Unknown param keys: {unknown}. "
                f"Valid keys: {set(PARAM_RANGES.keys())}"
            )

        param_low  = np.array([PARAM_RANGES[k][0] for k in self.param_keys], dtype=np.float32)
        param_high = np.array([PARAM_RANGES[k][1] for k in self.param_keys], dtype=np.float32)

        base = self.unwrapped.observation_space
        self.observation_space = Box(
            low=np.concatenate([base.low,  param_low]),
            high=np.concatenate([base.high, param_high]),
            dtype=np.float32,
        )

    def observation(self, obs: np.ndarray) -> np.ndarray:
        params = np.array(
            [self.unwrapped.current_params[k] for k in self.param_keys],
            dtype=np.float32,
        )
        return np.concatenate([obs, params])


class HistoryWrapper(gym.Wrapper):
    """Stacks the last *steps* (observation ‖ action) frames into a
    single flat observation.

    Supports both discrete (one-hot encoded) and continuous (raw float)
    action spaces.

    Initial frames are zero-filled so the observation shape is always
    consistent from the very first ``reset()``.

    Args:
        env:   Environment to wrap.
        steps: Number of (obs, action) time-steps to keep. Must be ≥ 2.

    Raises:
        ValueError: If *steps* < 2.

    Note:
        Uses :class:`collections.deque` with ``maxlen=steps`` so that appending
        a new frame automatically evicts the oldest in O(1).
    """

    def __init__(self, env: gym.Env, steps: int) -> None:
        super().__init__(env)
        if steps < 2:
            raise ValueError(f"steps must be ≥ 2, got {steps}.")
        self.steps = steps

        # Determine action dimensionality for frame encoding
        if hasattr(self.action_space, 'n'):
            # Discrete: one-hot encoding
            self._discrete = True
            self._action_dim = self.action_space.n
            act_low = np.zeros(self._action_dim)
            act_high = np.ones(self._action_dim)
        else:
            # Continuous Box: raw action values
            self._discrete = False
            self._action_dim = int(np.prod(self.action_space.shape))
            act_low = np.broadcast_to(self.action_space.low.flatten(),
                                      (self._action_dim,))
            act_high = np.broadcast_to(self.action_space.high.flatten(),
                                       (self._action_dim,))

        # Shape of a single frame: obs concatenated with action representation
        self._step_low = np.concatenate(
            [self.observation_space.low, act_low]
        ).astype(np.float32)
        self._step_high = np.concatenate(
            [self.observation_space.high, act_high]
        ).astype(np.float32)

        self.observation_space = Box(
            low=np.tile(self._step_low, self.steps),
            high=np.tile(self._step_high, self.steps),
            dtype=np.float32,
        )

        self._history: deque[np.ndarray] = self._make_history()

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _make_history(self) -> deque[np.ndarray]:
        """Return a fresh deque of zero frames with maxlen already set."""
        return deque(
            [np.zeros_like(self._step_low) for _ in range(self.steps)],
            maxlen=self.steps,
        )

    def _encode(self, obs: np.ndarray, action) -> np.ndarray:
        """Concatenate observation with action representation."""
        if self._discrete:
            act_vec = np.zeros(self._action_dim, dtype=np.float32)
            act_vec[int(action)] = 1.0
        else:
            act_vec = np.atleast_1d(np.asarray(action, dtype=np.float32)).flatten()
        return np.concatenate([obs, act_vec])

    def _flat_history(self) -> np.ndarray:
        return np.concatenate(list(self._history)).astype(np.float32)

    # ------------------------------------------------------------------
    # Gymnasium API
    # ------------------------------------------------------------------

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ) -> tuple[np.ndarray, dict]:
        obs, info = self.env.reset(seed=seed, options=options)
        self._history = self._make_history()
        null_action = 0 if self._discrete else np.zeros(self._action_dim)
        self._history.append(self._encode(obs, null_action))
        return self._flat_history(), info

    def step(
        self, action
    ) -> tuple[np.ndarray, float, bool, bool, dict]:
        obs, reward, terminated, truncated, info = self.env.step(action)
        self._history.append(self._encode(obs, action))
        return self._flat_history(), reward, terminated, truncated, info
