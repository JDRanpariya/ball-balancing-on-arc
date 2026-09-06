"""Diagnostic / monitoring wrappers for the Balancer environment."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import gymnasium as gym

_log = logging.getLogger(__name__)


class BalancerLogger(gym.Wrapper):
    """Emits a structured log line on every ``step()``.

    Args:
        env:       Environment to wrap.
        use_print: When ``True`` write to *stdout* via ``print()``.
                   When ``False`` (default) emit at ``DEBUG`` level so output
                   can be silenced without touching this code::

                       logging.getLogger("balancer.wrappers.logging").setLevel(logging.WARNING)

    Example::

        env = BalancerSim()
        env = BalancerLogger(env)              # silent unless DEBUG enabled
        env = BalancerLogger(env, use_print=True)  # always visible
    """

    def __init__(self, env: gym.Env, *, use_print: bool = False) -> None:
        super().__init__(env)
        self._use_print = use_print

    # ------------------------------------------------------------------

    def _emit(self, msg: str) -> None:
        if self._use_print:
            print(msg)
        else:
            _log.debug(msg)

    # ------------------------------------------------------------------

    def step(
        self, action: Any
    ) -> tuple[np.ndarray, float, bool, bool, dict]:
        obs, reward, terminated, truncated, info = self.env.step(action)
        self._emit(
            f"[BalancerLogger] "
            f"obs={obs}  reward={reward:.4f}  action={action}  "
            f"terminated={terminated}  truncated={truncated}"
        )
        return obs, reward, terminated, truncated, info