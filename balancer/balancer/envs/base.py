from typing import Optional
from collections import deque
import gymnasium as gym
import numpy as np
from gymnasium import spaces

from balancer.core.state import CART_X, BALL_X, CART_DOT, BALL_DOT
from balancer.rendering.viewer import viewer2D
from balancer.envs.rewards import REWARDS, REWARDS_NEED_ACTION
from balancer.hardware.constants import SYSTEM


class BalancerBase(gym.Env):
    metadata = {
        "render_modes": ["rgb_array", "human"],
        "render_fps": 50,
    }

    def __init__(
            self,
            reward='regular',
            position_limits=[SYSTEM.CART_LIMIT, SYSTEM.BALL_LIMIT],
            speed_limits=None,
            render_mode="rgb_array",
            use_action_delay=False,
            action_type="discrete"
    ):
        self.viewer = viewer2D(self.metadata['render_fps'], render_mode)
        self._state = None
        self.reward = reward
        self._reward_func = REWARDS[self.reward]
        self.render_mode = render_mode
        self.use_action_delay = use_action_delay
        self.action_delay = 0
        self.action_buffer = deque(
            [1] * max(1, self.action_delay), maxlen=max(1, self.action_delay))
        self.action_type = action_type

        # -- Action tracking for reward functions that need it ---------
        self._prev_action = 0.0
        self._current_action = 0.0
        # Check if reward function accepts action context
        self._reward_needs_action = self.reward in REWARDS_NEED_ACTION

        self.pos_limits = np.array(position_limits, dtype=np.float32)

        if speed_limits is None:
            speed_limits = [np.nan, np.nan]
        speed_limits = np.array(speed_limits, dtype=np.float32)

        self.pos_limits = np.where(
            np.isnan(self.pos_limits), np.inf, self.pos_limits)
        speed_limits = np.where(np.isnan(speed_limits), np.inf, speed_limits)

        self.high = np.array(
            [
                self.pos_limits[0],
                speed_limits[0],
                self.pos_limits[1],
                speed_limits[1],
            ],
            dtype=np.float32,
        )

        self.observation_space = spaces.Box(
            low=-self.high,
            high=self.high,
            dtype=np.float32,
        )

        if self.action_type == "cont":
            # self.action_space = spaces.Box(low=-0.9, high=0.9, shape=(1,), dtype=np.float32)
            self.action_space = spaces.Box(
                low=-1, high=1, shape=(1,), dtype=np.float32)
        else:
            self.action_space = spaces.Discrete(3)

    def _get_action_as_float(self, action):
        """Convert any action format to a float for reward computation."""
        if self.action_type == "cont":
            return float(np.asarray(action).flat[0])
        else:
            # Discrete: map {0: -1, 1: 0, 2: +1} so effort/jerk make sense
            return float(action - 1)

    def _compute_reward(self, action_float):
        """Call reward function with or without action context."""
        if self._reward_needs_action:
            return self._reward_func(
                self._state, self.pos_limits,
                action=action_float,
                prev_action=self._prev_action,
            )
        else:
            return self._reward_func(self._state, self.pos_limits)

    def step(self, action):

        if self.action_type == "cont":
            action = float(np.asarray(action).flat[0])

        if self.use_action_delay and self.action_delay > 0:
            self.action_buffer.append(action)
            delayed_action = self.action_buffer[0]
        else:
            delayed_action = action

        # -- Track actions for reward ----------------------------------
        action_float = self._get_action_as_float(delayed_action)

        # Step the simulation/hardware
        self._update_state(delayed_action)

        terminated = False

        if not terminated:
            reward = self._compute_reward(action_float)
        else:
            reward = 0.0

        # -- Update action history AFTER reward computation ------------
        self._prev_action = action_float

        truncated = False

        obs = np.clip(self._state, self.observation_space.low,
                      self.observation_space.high)

        return np.array(obs, dtype=np.float32), reward, terminated, truncated, {}

    def get_obs(self):
        obs = np.array(
            [
                self._state[CART_X],
                self._state[CART_DOT],
                self._state[BALL_X],
                self._state[BALL_DOT],
            ],
            dtype=np.float32
        )
        return obs

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ):
        super().reset(seed=seed, options=options)
        # -- Reset action tracking -------------------------------------
        self._prev_action = 0.0
        self._current_action = 0.0

        neutral = 1 if self.action_type == "discrete" else 0.0
        self.action_buffer = deque(
            [neutral] * self.action_delay, maxlen=self.action_delay
        )

    def _update_state(self, action):
        raise NotImplementedError

    def render(self):
        return self.viewer.display(self._state)

    def close(self):
        self.viewer.close()
