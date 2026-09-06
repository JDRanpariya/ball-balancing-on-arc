"""Controller adapter for d3rlpy offline RL policies (continuous action space).

Loads a policy saved with d3rlpy (Seno & Imai, JMLR 2022) and exposes the
standard BaseController interface.

Usage:
    from balancer.controllers.offline_rl import OfflineRLController

    ctrl = OfflineRLController("path/to/cont_cql_model.d3")
    action = ctrl.step(state, cart_limit)
"""

import numpy as np
from balancer.controllers.base import BaseController


class OfflineRLController(BaseController):
    """
    Controller wrapping a d3rlpy continuous-action policy.

    Args:
        path_to_model: Path to saved d3rlpy model (.d3 file from algo.save()).
        device: Inference device ('cpu' or 'cuda').
        deterministic: Not used (d3rlpy predict is deterministic), kept for API compat.
    """

    def __init__(
        self,
        path_to_model: str,
        device: str = "cpu",
        deterministic: bool = True,
    ):
        self.device = device
        self.deterministic = deterministic
        self.model = self._load_model(path_to_model)

        from pathlib import Path
        self.model_name = Path(path_to_model).stem

    def _load_model(self, path: str):
        """Load a d3rlpy model using load_learnable."""
        import d3rlpy
        model = d3rlpy.load_learnable(path, device=self.device)
        return model

    def step(self, state, cart_limit) -> float:
        """
        Compute one control action.

        Args:
            state: [cart_pos, cart_vel, ball_angle, ball_ang_vel]
            cart_limit: Cart position limit (unused by policy, kept for interface).

        Returns:
            float: Control signal in [-1, 1].
        """
        obs = np.asarray(state, dtype=np.float32).reshape(1, -1)
        action = self.model.predict(obs)
        return float(np.clip(action.flat[0], -1.0, 1.0))

    def reset(self) -> None:
        """No internal state to reset."""
        pass
