"""
Generalized RL controller for Stable-Baselines3 policies.

Loads policies trained with Stable-Baselines3 (Raffin et al., JMLR 2021) and
its SB3-Contrib extensions, exposing the standard BaseController interface.
Supports: PPO, A2C, DQN, SAC, TD3, TRPO, RecurrentPPO, QRDQN, CrossQ, TQC, ARS.
"""

import importlib
import numpy as np
from balancer.controllers.base import BaseController


# ===========================================================================

#  ALGORITHM REGISTRY

# ===========================================================================

ALGO_REGISTRY = {
    # -- Stable-Baselines3 core ----------------------------------------
    "ppo":  ("stable_baselines3",  "PPO"),
    # PPO policy trained inside the learned world model (PPO-WM), deployed
    # zero-shot; a plain SB3 PPO checkpoint at inference time.
    "ppo_wm": ("stable_baselines3", "PPO"),
    "a2c":  ("stable_baselines3",  "A2C"),
    "dqn":  ("stable_baselines3",  "DQN"),
    "sac":  ("stable_baselines3",  "SAC"),
    "td3":  ("stable_baselines3",  "TD3"),
    # -- SB3-Contrib ---------------------------------------------------
    "qrdqn":         ("sb3_contrib", "QRDQN"),
    "trpo":          ("sb3_contrib", "TRPO"),
    "recurrent_ppo": ("sb3_contrib", "RecurrentPPO"),
    "crossq":        ("sb3_contrib", "CrossQ"),
    "tqc":           ("sb3_contrib", "TQC"),
    "ars":           ("sb3_contrib", "ARS"),
}

# Algorithms that maintain recurrent hidden state

_RECURRENT_ALGOS = {"recurrent_ppo"}


def _load_algo_class(algo_name: str):
    """Dynamically import and return the algorithm class."""
    algo_name = algo_name.lower()
    if algo_name not in ALGO_REGISTRY:
        raise ValueError(
            f"Unknown algorithm '{algo_name}'. "
            f"Available: {list(ALGO_REGISTRY.keys())}"
        )
    module_name, class_name = ALGO_REGISTRY[algo_name]
    module = importlib.import_module(module_name)
    return getattr(module, class_name)


def infer_algo_from_path(path: str) -> str:
    """
    Infer algorithm name from model file path.

    Convention:  models/{action_type}/{algo_name}.zip
    Fallback:    'ppo' (backward compatible)
    """
    from pathlib import Path
    stem = Path(path).stem.lower()

    # Direct match against registry
    if stem in ALGO_REGISTRY:
        return stem

    # Check if stem contains an algo name
    for algo in ALGO_REGISTRY:
        if algo in stem:
            return algo

    return "ppo"  # safe default


# ===========================================================================

#  RL CONTROLLER

# ===========================================================================

class RLController(BaseController):
    """
    Universal RL controller for all SB3/SB3-contrib algorithms.

    Args:
        path_to_model:  Path to saved .zip model file.
        action_type:    "discrete" or "cont".
        algo:           Algorithm name (e.g. "ppo", "sac", "dqn").
                        If None, inferred from filename.
        deterministic:  Use deterministic policy at inference.
    """

    def __init__(
        self,
        path_to_model: str,
        action_type: str = "discrete",
        algo: str = None,
        deterministic: bool = True,
    ):
        self.action_type = action_type
        self.deterministic = deterministic

        # Infer algorithm if not specified
        if algo is None:
            algo = infer_algo_from_path(path_to_model)
        self.algo_name = algo.lower()

        # Load model
        self.model = self._load_model(path_to_model)

        # Model name for logging
        from pathlib import Path
        self.model_name = f"{self.algo_name}_{Path(path_to_model).stem}"

        # Recurrent state tracking (for RecurrentPPO)
        self._is_recurrent = self.algo_name in _RECURRENT_ALGOS
        self._lstm_states = None
        self._episode_start = True

    def _load_model(self, path: str):
        """Load model using the appropriate algorithm class."""
        cls = _load_algo_class(self.algo_name)
        try:
            model = cls.load(
                path,
                custom_objects={
                    "clip_range": lambda _: 0.2,
                    "lr_schedule": lambda _: 3e-4,
                },
            )
        except TypeError:
            # Some algorithms don't accept custom_objects
            model = cls.load(path)
        return model

    def step(self, state, cart_limit):
        """
        Compute one control action.

        Returns:
            float - control signal in [-1, 1]
        """
        obs = np.asarray(state, dtype=np.float32)

        # -- Predict action --------------------------------------------
        if self._is_recurrent:
            action, self._lstm_states = self.model.predict(
                obs,
                state=self._lstm_states,
                episode_start=np.array([self._episode_start]),
                deterministic=self.deterministic,
            )
            self._episode_start = False
        else:
            action, _ = self.model.predict(
                obs, deterministic=self.deterministic)

        # -- Map to control signal -------------------------------------
        cart_pos = state[0]

        if self.action_type == "discrete":
            # action is int ∈ {0, 1, 2}
            a = int(action)
            if a == 2:
                u = 1.0
            elif a == 0:
                u = -1.0
            else:
                u = 0.0
        else:
            # action is float array from Box(-1, 1)
            u = float(np.asarray(action).flat[0])

        return u

    def reset(self):
        """Reset internal state (LSTM states for recurrent models)."""
        self._lstm_states = None
        self._episode_start = True
