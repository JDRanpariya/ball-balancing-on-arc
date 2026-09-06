"""Curriculum / reset-schedule wrappers for the Balancer environment."""

from __future__ import annotations

from typing import Optional

import numpy as np
import gymnasium as gym


class DynResetEpisode(gym.Wrapper):
    """Gradually widens the initial-state distribution over training.

    On each ``reset()`` the wrapper samples a state from the current
    observation-space bounds and seeds the underlying environment.
    Intended to start from a narrow safe region and slowly expand as the
    agent improves - external training code should widen the bounds
    progressively by modifying ``observation_space.low / high``.

    Warning:
        Randomly sampled initial states can be physically invalid.
        Validate against hardware safety limits before use on real robots.
    """

    def __init__(self, env: gym.Env) -> None:
        super().__init__(env)

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ) -> tuple[np.ndarray, dict]:
        super().reset(seed=seed, options=options)

        self.env.unwrapped._state = self.observation_space.sample()
        obs = self.env.get_obs()

        if self.render_mode == "human":
            self.render()
        return obs, {}