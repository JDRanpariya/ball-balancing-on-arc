from typing import List, Optional, Union

import os
import json
import numpy as np
import gymnasium as gym

from balancer.hardware.robot import Robot
from balancer.envs.base import BalancerBase
from balancer.core.state import CART_X, CART_DOT, BALL_X, BALL_DOT
from balancer.hardware.constants import SYSTEM

# Software correction for sensor zero offset (true equilibrium ≠ reported zero)
_SENSOR_OFFSET = SYSTEM.SENSOR_OFFSET_RAD


class BalancerReal(BalancerBase):
    def __init__(
        self,
        robot: Optional[Robot] = None,
        reward='regular',
        position_limits=[SYSTEM.CART_LIMIT, SYSTEM.BALL_LIMIT],
        speed_limits=[1, 1],
        render_mode="rgb_array",
        action_type="discrete",
        param_file: str = None,
    ):

        super().__init__(reward, position_limits, speed_limits,
                         render_mode, use_action_delay=False, action_type=action_type)
        if robot is None:
            self.robot = Robot(action_type=action_type)
        else:
            self.robot = robot
            self.robot.action_type = action_type
        self.pos_limits = np.array(position_limits, dtype=np.float32)
        self._state = None
        self.action_type = action_type
        self.current_params = {
            "gravity":             9.8434,
            "arc_radius":          2.1125,
            "friction_coeff_cart": 0.1534,
            "mu_ball_rolling":     0.025,   # split from old friction_coeff_ball
            "mu_ball_viscous":     0.009,
            "tau_v":               0.15,
        }
        if param_file is not None:
            if not os.path.isfile(param_file):
                raise FileNotFoundError(f"Param file not found: {param_file}")
            with open(param_file, 'r') as f:
                param_data = json.load(f)
                if 'best_params' in param_data:
                    self.current_params.update(param_data['best_params'])
                else:
                    self.current_params.update(param_data)

    def _update_state(self, action=None):
        with self.robot._state.lock:
            self._state = [
                self.robot.cart_position,
                self.robot.cart_velocity,
                self.robot.ball_position - _SENSOR_OFFSET,
                self.robot.ball_velocity
            ]

    def step(self, action):
        motor_cmd = action

        # Step the robot (gets state update)
        obs, robot_reward, _, _, _ = self.robot.step(
            motor_cmd, self.action_type)

        # Update internal state from observation
        self._state = obs.tolist()
        # Apply sensor offset correction: true equilibrium is at θ_raw=+0.007
        self._state[BALL_X] -= _SENSOR_OFFSET

        # -- Use base class action tracking + reward -------------------
        action_float = self._get_action_as_float(action)

        # No early termination on real hardware (ball can't fall off - walls)
        # Reward still penalises out-of-bounds states via the reward function
        terminated = False
        truncated = False

        reward = self._compute_reward(action_float)
        # Add hardware IR sensor bonus (dip_reward=1 when ball at center)
        with self.robot._state.lock:
            hw_dip = self.robot._state.dip_reward
        reward += float(hw_dip)  # binary 0/1, full weight
        reward = float(np.clip(reward, -1.0, 1.0))  # clip to [-1, 1]

        # Update action history
        self._prev_action = action_float

        final_obs = np.clip(
            self._state, self.observation_space.low, self.observation_space.high)

        return np.array(final_obs, dtype=np.float32), reward, terminated, truncated, {}

    def reset(
            self,
            seed: Optional[int] = None,
            options: Optional[dict] = None,
    ):
        super().reset(seed=seed, options=options)

        if self._state is None:
            self._state = [0.0, 0.0, 0.0, 0.0]

        obs, _ = self.robot.reset()
        # Apply sensor offset correction
        obs[BALL_X] -= _SENSOR_OFFSET

        # Clip to observation space (sensor can slightly exceed bounds)
        obs = np.clip(obs, self.observation_space.low, self.observation_space.high)
        self._state = obs.tolist()

        print(f"Env Reset..., state is {obs} ")
        return obs, {}

    def close(self):
        self.robot.close()
