"""
world_model_env.py - Gymnasium env backed by LSTM world model.

Uses the trained world model as dynamics instead of physics equations.
Supports both single-env (CPU) and vectorized batched (GPU) operation.

The LSTM hidden state is maintained across episode steps for accurate
multi-step predictions.
"""

from __future__ import annotations

from typing import Optional, Tuple

import gymnasium as gym
import numpy as np
import torch

from balancer.core.state import CART_X, CART_DOT, BALL_X, BALL_DOT
from balancer.envs.rewards import REWARDS, REWARDS_NEED_ACTION
from balancer.hardware.constants import SYSTEM
from balancer.world_model import WorldModelPredictor


class BalancerWorldModel(gym.Env):
    """Gymnasium env using the LSTM world model as dynamics.

    The world model was trained on real hardware data and captures
    all real-system effects (motor dynamics, sensor characteristics,
    friction) implicitly.

    Args:
        checkpoint_path: Path to world model .pth checkpoint.
        reward: Reward function name (from REWARDS registry).
        device: 'cpu' or 'cuda'.
        noise_std: Gaussian noise added to predictions for stochasticity.
        max_episode_steps: Maximum steps per episode.
        action_type: 'cont' (continuous) only.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        checkpoint_path: str = "evaluation/models/world_model/v1/best_model.pth",
        reward: str = "balanced",
        device: str = "cpu",
        noise_std: float = 0.001,
        max_episode_steps: int = 600,
        action_type: str = "cont",
        render_mode: str = None,
        **kwargs,  # Accept extra Hydra keys (fixed_param, etc.)
    ):
        super().__init__()
        assert action_type == "cont", "World model env only supports continuous actions"

        self.predictor = WorldModelPredictor(checkpoint_path, device=device)
        self.device = self.predictor.device
        self.reward_name = reward
        self.noise_std = noise_std
        self.max_episode_steps = max_episode_steps

        # Limits
        self.cart_limit = SYSTEM.CART_LIMIT
        self.ball_limit = SYSTEM.BALL_LIMIT
        self.pos_limits = (self.cart_limit, self.ball_limit)

        # Spaces
        obs_high = np.array(
            [self.cart_limit, SYSTEM.MAX_CART_VEL, self.ball_limit, 2.0],
            dtype=np.float32,
        )
        self.observation_space = gym.spaces.Box(
            -obs_high, obs_high, dtype=np.float32)
        self.action_space = gym.spaces.Box(
            low=-1.0, high=1.0, shape=(1,), dtype=np.float32
        )

        # State
        self._state = np.zeros(4, dtype=np.float32)
        self._prev_action = 0.0
        self._step_count = 0
        self._wall_steps = 0
        self._hc: Optional[Tuple[torch.Tensor, torch.Tensor]] = None

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ) -> Tuple[np.ndarray, dict]:
        super().reset(seed=seed)

        # Sample initial condition from realistic distribution
        # Cart: uniform over [-0.7*cart_limit, 0.7*cart_limit] - avoid wall zones
        # so the policy never learns to exploit walls as a stable state
        # Ball: uniform over [-ball_limit*0.9, ball_limit*0.9] (avoid immediate termination)
        # Velocities: small random
        cart_pos = self.np_random.uniform(-self.cart_limit *
                                          0.7, self.cart_limit * 0.7)
        cart_vel = self.np_random.uniform(-0.05, 0.05)
        ball_pos = self.np_random.uniform(-self.ball_limit *
                                          0.9, self.ball_limit * 0.9)
        ball_vel = self.np_random.uniform(-0.02, 0.02)

        self._state = np.array(
            [cart_pos, cart_vel, ball_pos, ball_vel], dtype=np.float32
        )
        self._prev_action = 0.0
        self._step_count = 0
        self._wall_steps = 0
        self._hc = None  # Reset LSTM hidden state

        return self._state.copy(), {}

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, dict]:
        action_float = float(np.clip(action, -1.0, 1.0).flat[0])

        # Predict next state using world model (stateless)
        s_t = torch.tensor(
            self._state, dtype=torch.float32, device=self.device
        ).unsqueeze(0)
        a_t = torch.tensor(
            [[action_float]], dtype=torch.float32, device=self.device
        )

        next_s, self._hc = self.predictor.predict_batch(s_t, a_t, self._hc)
        next_state = next_s.squeeze(0).cpu().numpy()

        # Add small noise for stochasticity
        if self.noise_std > 0:
            next_state += self.np_random.normal(0, self.noise_std, size=4).astype(
                np.float32
            )

        self._state = next_state
        self._step_count += 1

        # Termination
        cart_out = abs(self._state[CART_X]) > self.cart_limit
        ball_out = abs(self._state[BALL_X]) > self.ball_limit
        # Terminate if cart is stuck at wall for >1s (20 steps)
        # Forces policy to learn wall-recovery using WM's real recovery data
        if abs(self._state[CART_X]) / self.cart_limit > 0.92:
            self._wall_steps += 1
        else:
            self._wall_steps = 0
        wall_stuck = self._wall_steps >= 20
        terminated = bool(cart_out or ball_out or wall_stuck)
        truncated = self._step_count >= self.max_episode_steps

        # Reward
        if not terminated:
            reward_fn = REWARDS[self.reward_name]
            if self.reward_name in REWARDS_NEED_ACTION:
                reward = reward_fn(
                    self._state.tolist(),
                    self.pos_limits,
                    action=action_float,
                    prev_action=self._prev_action,
                )
            else:
                reward = reward_fn(self._state.tolist(), self.pos_limits)
        else:
            reward = -1.0

        self._prev_action = action_float

        obs = np.clip(self._state, self.observation_space.low,
                      self.observation_space.high)
        return obs, float(reward), terminated, truncated, {}


class BalancerWorldModelVec(gym.Env):
    """Batched world model env for vectorized PPO training.

    Steps N environments simultaneously in a single GPU kernel call.
    Use instead of SubprocVecEnv for maximum throughput.

    This is a single Gym env that internally manages N parallel episodes.
    Wrap with DummyVecEnv(n=1) for SB3 compatibility.

    Args:
        n_envs: Number of parallel environments.
        checkpoint_path: Path to world model .pth checkpoint.
        reward: Reward function name.
        device: 'cpu' or 'cuda'.
        noise_std: Gaussian noise for stochasticity.
        max_episode_steps: Maximum steps per episode.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        n_envs: int = 16,
        checkpoint_path: str = "evaluation/models/world_model/v1/best_model.pth",
        reward: str = "balanced",
        device: str = "cuda",
        noise_std: float = 0.001,
        max_episode_steps: int = 600,
    ):
        super().__init__()
        self.n_envs = n_envs
        self.predictor = WorldModelPredictor(checkpoint_path, device=device)
        self.device = self.predictor.device
        self.reward_name = reward
        self.noise_std = noise_std
        self.max_episode_steps = max_episode_steps

        # Limits
        self.cart_limit = SYSTEM.CART_LIMIT
        self.ball_limit = SYSTEM.BALL_LIMIT
        self.pos_limits = (self.cart_limit, self.ball_limit)

        # Spaces (single env interface for SB3)
        obs_high = np.array(
            [self.cart_limit, SYSTEM.MAX_CART_VEL, self.ball_limit, 2.0],
            dtype=np.float32,
        )
        self.observation_space = gym.spaces.Box(
            -obs_high, obs_high, dtype=np.float32)
        self.action_space = gym.spaces.Box(
            low=-1.0, high=1.0, shape=(1,), dtype=np.float32
        )

        # Batched state (on device)
        self._states = torch.zeros(n_envs, 4, device=self.device)
        self._prev_actions = torch.zeros(n_envs, device=self.device)
        self._step_counts = torch.zeros(
            n_envs, dtype=torch.int32, device=self.device)
        self._hc: Optional[Tuple[torch.Tensor, torch.Tensor]] = None

    def _sample_ic(self, indices: torch.Tensor) -> None:
        """Sample initial conditions for given env indices."""
        n = indices.shape[0]
        cart_pos = torch.empty(n, device=self.device).uniform_(
            -self.cart_limit, self.cart_limit
        )
        cart_vel = torch.empty(n, device=self.device).uniform_(-0.05, 0.05)
        ball_pos = torch.empty(n, device=self.device).uniform_(
            -self.ball_limit * 0.9, self.ball_limit * 0.9
        )
        ball_vel = torch.empty(n, device=self.device).uniform_(-0.02, 0.02)

        self._states[indices, 0] = cart_pos
        self._states[indices, 1] = cart_vel
        self._states[indices, 2] = ball_pos
        self._states[indices, 3] = ball_vel
        self._step_counts[indices] = 0
        self._prev_actions[indices] = 0.0

        # Reset hidden state for these envs
        if self._hc is not None:
            h, c = self._hc
            h[:, indices, :] = 0.0
            c[:, indices, :] = 0.0

    def reset(
        self, *, seed: Optional[int] = None, options: Optional[dict] = None
    ) -> Tuple[np.ndarray, dict]:
        super().reset(seed=seed)
        all_indices = torch.arange(self.n_envs, device=self.device)
        self._sample_ic(all_indices)
        self._hc = None
        # Return first env's state (for SB3 DummyVecEnv wrapping)
        return self._states[0].cpu().numpy(), {}

    def step_batch(
        self, actions: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Step all N envs simultaneously.

        Args:
            actions: (N, 1) tensor of actions.

        Returns:
            obs: (N, 4) next observations.
            rewards: (N,) reward per env.
            terminated: (N,) bool termination mask.
            truncated: (N,) bool truncation mask.
        """
        actions = actions.clamp(-1.0, 1.0)

        # World model prediction
        next_states, self._hc = self.predictor.predict_batch(
            self._states, actions, self._hc
        )

        # Add noise
        if self.noise_std > 0:
            next_states = next_states + \
                torch.randn_like(next_states) * self.noise_std

        self._states = next_states
        self._step_counts += 1

        # Termination
        cart_out = self._states[:, CART_X].abs() > self.cart_limit
        ball_out = self._states[:, BALL_X].abs() > self.ball_limit
        terminated = cart_out | ball_out
        truncated = self._step_counts >= self.max_episode_steps

        # Compute rewards (on CPU for now - reward functions are numpy)
        states_np = self._states.cpu().numpy()
        actions_np = actions.squeeze(-1).cpu().numpy()
        prev_actions_np = self._prev_actions.cpu().numpy()
        rewards = np.zeros(self.n_envs, dtype=np.float32)

        reward_fn = REWARDS[self.reward_name]
        needs_action = self.reward_name in REWARDS_NEED_ACTION

        for i in range(self.n_envs):
            if terminated[i]:
                rewards[i] = -1.0
            elif needs_action:
                rewards[i] = reward_fn(
                    states_np[i].tolist(),
                    self.pos_limits,
                    action=actions_np[i],
                    prev_action=prev_actions_np[i],
                )
            else:
                rewards[i] = reward_fn(states_np[i].tolist(), self.pos_limits)

        self._prev_actions = actions.squeeze(-1)

        # Auto-reset terminated/truncated envs
        done_mask = terminated | truncated
        if done_mask.any():
            done_indices = done_mask.nonzero(as_tuple=False).squeeze(-1)
            self._sample_ic(done_indices)

        rewards_t = torch.tensor(rewards, device=self.device)
        obs = self._states.clamp(
            torch.tensor(self.observation_space.low, device=self.device),
            torch.tensor(self.observation_space.high, device=self.device),
        )
        return obs, rewards_t, terminated, truncated

    def step(self, action: np.ndarray):
        """Single-env step interface (for SB3 compatibility)."""
        # This shouldn't be used directly in batched mode
        a_t = torch.tensor(
            [[float(action.flat[0])]], dtype=torch.float32, device=self.device
        )
        obs, rew, term, trunc = self.step_batch(a_t)
        return (
            obs[0].cpu().numpy(),
            float(rew[0]),
            bool(term[0]),
            bool(trunc[0]),
            {},
        )
