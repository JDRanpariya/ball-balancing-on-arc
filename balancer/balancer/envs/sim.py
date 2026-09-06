import torch
from collections import deque
import numpy as np
from typing import Optional

from balancer.envs.base import BalancerBase
from balancer.core.dynamics import NonLinearDynamics
from balancer.core.state import CART_X, CART_DOT, BALL_X, BALL_DOT
from balancer.hardware.constants import SYSTEM
from balancer.core.params import DR_RANGES, DR_MEANS, OBS_NOISE_RANGES, NOMINAL_OBS_NOISE


class BalancerSim(BalancerBase):
    """
    Simulated ball-balancer environment using physics dynamics.

    This environment simulates a ball balancing on a moving arc/cart using
    either analytical nonlinear dynamics or a learned world model.

    Args:
        dyn (NonLinearDynamics, optional): Dynamics model. Defaults to NonLinearDynamics().
        reward (str, optional): Reward function name. Defaults to 'regular'.
        position_limits (list): [cart_limit, ball_limit] in meters.
        speed_limits (list): [cart_speed_limit, ball_speed_limit].
        render_mode (str): "rgb_array" or "human".
        action_type (str): "discrete" or "cont".
        use_world_model (bool): Use learned dynamics instead of physics.
        fixed_param (bool): Use fixed physics parameters.
        device (str): 'cpu' or 'cuda'.
        ball_obs_rate (float): Ball sensor effective update rate in Hz.
            Hardware ToF sensor updates at ~10 Hz while control runs at 20 Hz.
            When < control rate (1/tau), ball obs are held stale between updates.
            Set to None or 0 to disable (perfect observations).
        sensor_blind_zone (bool): Enable VL53L0X minimum-range blind zone.
            When the ball is near the arc edge (|θ| > ~0.064 rad), the ToF
            sensor cannot resolve the distance and the reading saturates.
            The observed angle clamps at the blind zone boundary and velocity
            appears ~0. Threshold is randomised under DR. Default: True.

    Attributes:
        dyn: The dynamics model
        device: Computation device
        use_world_model: Whether using learned model

    Example:
        >>> env = BalancerSim(reward='ball_gaussian_distance', action_type='discrete')
        >>> obs, info = env.reset()
        >>> obs, reward, done, truncated, info = env.step(1)
    """

    def __init__(
            self,
            dyn=None,
            reward='regular',
            position_limits=[SYSTEM.CART_LIMIT, SYSTEM.BALL_LIMIT],
            speed_limits=[1, 1],
            render_mode="rgb_array",
            action_type="discrete",
            fixed_param=True,
            device='cpu',
            tau=0.05,               # default matches hardware 20Hz
            integrator="rk4",       # match tuning
            ball_obs_rate=10.0,     # ToF sensor ~10Hz (measured from hardware)
            sensor_blind_zone=False, # VL53L0X blind zone - enable only with DR
    ):
        super().__init__(reward, position_limits, speed_limits,
                         render_mode, action_type=action_type, use_action_delay=False)

        self.device = device
        self.fixed_param = fixed_param
        self.predictor = None
        self._obs_noise_std = np.zeros(4, dtype=np.float32)
        self.dyn = dyn if dyn is not None else NonLinearDynamics(
            tau=tau,
            kinematics_integrator=integrator,
        )

        # -- Ball sensor staleness (hardware ToF updates at ~12Hz) --
        # Control runs at 1/tau Hz. If ball_obs_rate < control_rate,
        # ball observations are held for multiple steps.
        self._ball_obs_rate = ball_obs_rate
        control_rate = 1.0 / tau
        if ball_obs_rate and ball_obs_rate < control_rate:
            # Mean steps between ball updates
            self._ball_hold_mean = control_rate / ball_obs_rate  # e.g. 20/10 = 2.0
        else:
            self._ball_hold_mean = 1.0  # update every step (no staleness)
        self._ball_obs_countdown = 0  # steps until next ball update
        self._last_ball_obs = np.zeros(2, dtype=np.float32)  # [theta, theta_dot]

        # -- VL53L0X minimum-range blind zone --
        # The VL53L0X cannot measure below ~30-50mm. When the ball is
        # close to a sensor (near arc edge), the reading saturates:
        # the sensor reports a clamped distance as if the ball stopped
        # at the blind zone boundary. On hardware this manifests as the
        # angle reading "sticking" at the BZ boundary.
        #
        # Derived from constants.py:
        #   sensor_center = mean(LEFT_CENTER_MM, RIGHT_CENTER_MM)
        #   θ_blind = (sensor_center - min_range) / 1000 / ARC_RADIUS_M
        #   = (mean(189,158) - 40) / 1000 / 2.101 ≈ 0.064 rad
        #
        # When |θ_true| > θ_blind:
        #   - position obs clamped to ±θ_blind (sign preserved)
        #   - velocity obs goes to ~0 (no apparent motion)
        self._sensor_blind_zone = sensor_blind_zone
        # Nominal blind zone threshold (derived from sensor geometry)
        _sensor_center_mm = (SYSTEM.LEFT_CENTER_MM + SYSTEM.RIGHT_CENTER_MM) / 2.0
        _vl53l0x_min_range_mm = 40.0  # datasheet typical minimum range
        self._blind_zone_theta = (
            (_sensor_center_mm - _vl53l0x_min_range_mm) / 1000.0
            / SYSTEM.ARC_RADIUS_M
        )  # ≈ 0.064 rad

        # -- Eval: pin dynamics to DR-centre + nominal obs noise --
        if self.fixed_param:
            self.dyn.gravity = DR_MEANS["gravity"]
            self.dyn.arc_radius = DR_MEANS["arc_radius"]
            self.dyn.friction_coeff_cart = DR_MEANS["friction_coeff_cart"]
            self.dyn.mu_ball_rolling = DR_MEANS["mu_ball_rolling"]
            self.dyn.mu_ball_viscous = DR_MEANS["mu_ball_viscous"]
            self.dyn.velocity_lag_tau = DR_MEANS["tau_v"]
            self.dyn.dip_depth = DR_MEANS["dip_depth"]
            self.dyn.dip_fwhm = DR_MEANS["dip_fwhm"]
            self._obs_noise_std = NOMINAL_OBS_NOISE.copy()
        else:
            self._obs_noise_std = np.zeros(4, dtype=np.float32)

        # Always set current_params AFTER any overrides
        self.current_params = {
            "gravity":             self.dyn.gravity,
            "arc_radius":          self.dyn.arc_radius,
            "friction_coeff_cart": self.dyn.friction_coeff_cart,
            "mu_ball_rolling":     self.dyn.mu_ball_rolling,
            "mu_ball_viscous":     self.dyn.mu_ball_viscous,
            "tau_v":               self.dyn.velocity_lag_tau,
        }

        if action_type == "cont":
            self.dyn.action_type = "cont"
        else:
            self.dyn.action_type = "discrete"

    def _update_state(self, action):
        cur_state = self._true_state

        if self.action_type == "cont":
            action = float(action)

        x, x_dot, p, p_dot = self.dyn(cur_state, action)

        self._true_state = np.array([x, x_dot, p, p_dot])

        # Obs noise always applied - zeros for legacy, nominal for eval, random for DR
        noisy_state = self._true_state + \
            self.np_random.normal(0, self._obs_noise_std)

        # -- VL53L0X minimum-range blind zone --
        # When ball is near arc edge, sensor saturates: position clamps,
        # velocity appears ~0. Models the real hardware behavior where
        # readings "stick" at ~0.064 rad as ball goes beyond.
        if self._sensor_blind_zone:
            ball_theta = noisy_state[BALL_X]
            if abs(ball_theta) > self._blind_zone_theta:
                # Clamp position to blind zone boundary (sign preserved)
                noisy_state[BALL_X] = np.sign(ball_theta) * self._blind_zone_theta
                # Velocity appears near zero (sensor sees no change)
                # Add small noise so it's not perfectly 0
                noisy_state[BALL_DOT] = self.np_random.normal(0, 0.005)

        # -- Ball sensor staleness --
        # Cart (encoder) always updates at full rate.
        # Ball (ToF) only updates at ~10Hz -> hold previous reading.
        self._ball_obs_countdown -= 1
        if self._ball_obs_countdown <= 0:
            # New ball reading available
            self._last_ball_obs[0] = noisy_state[BALL_X]
            self._last_ball_obs[1] = noisy_state[BALL_DOT]
            # Next update in ~geometric(mean=hold_mean) steps
            # Use geometric draw to model jittery real sensor timing
            if self._ball_hold_mean > 1.0:
                self._ball_obs_countdown = self.np_random.geometric(
                    p=1.0 / self._ball_hold_mean)
            else:
                self._ball_obs_countdown = 1

        # Compose observation: fresh cart + possibly stale ball
        self._state = np.array([
            noisy_state[CART_X],
            noisy_state[CART_DOT],
            self._last_ball_obs[0],
            self._last_ball_obs[1],
        ], dtype=np.float32)

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed, options=options)  # also resets _prev_action

        if not self.fixed_param:
            self.current_params = {
                k: self.np_random.uniform(lo, hi)
                for k, (lo, hi) in DR_RANGES.items()
            }
            self.dyn.gravity = self.current_params["gravity"]
            self.dyn.arc_radius = self.current_params["arc_radius"]
            self.dyn.friction_coeff_cart = self.current_params["friction_coeff_cart"]
            self.dyn.mu_ball_rolling = self.current_params["mu_ball_rolling"]
            self.dyn.mu_ball_viscous = self.current_params["mu_ball_viscous"]
            self.dyn.velocity_lag_tau = self.current_params["tau_v"]
            self.dyn.dip_depth = self.current_params["dip_depth"]
            self.dyn.dip_fwhm = self.current_params["dip_fwhm"]

            # Action delay: disabled - real transport delay is <14ms (<0.3 steps)
            # The jerk-limited profile onset already accounts for this.
            self.action_delay = 0
            self.use_action_delay = False
            self._obs_noise_std = np.array([
                self.np_random.uniform(lo, hi)
                for lo, hi in OBS_NOISE_RANGES
            ], dtype=np.float32)

            # Randomise blind zone threshold (sensor min-range varies with
            # target reflectance, angle of incidence, ambient IR)
            # Nominal: 0.064 rad; range: [0.060, 0.080] rad
            if self._sensor_blind_zone:
                self._blind_zone_theta = self.np_random.uniform(0.060, 0.080)

        cart_margin_factor = 0.9  # up to ±0.70; wall states reached via exploration
        ball_margin_factor = 0.77  # shrinks ball pos init range to 0.6 instead of 0.78

        cart_pos = self.np_random.uniform(low=self.observation_space.low[0] * cart_margin_factor,
                                          high=self.observation_space.high[0] * cart_margin_factor)
        cart_vel = 0.0
        ball_pos = self.np_random.uniform(low=self.observation_space.low[2] * ball_margin_factor,
                                          high=self.observation_space.high[2] * ball_margin_factor)
        ball_vel = 0.0

        self._true_state = np.array([cart_pos, cart_vel, ball_pos, ball_vel])
        # float32 to match _update_state's obs dtype (reset obs was float64)
        self._state = self._true_state.astype(np.float32)  # first obs is clean

        # Reset dynamics-internal motor state so nothing leaks across episodes
        # (command delay buffer, velocity filter, jerk/7-phase accel state)
        self.dyn.reset_cmd_buffer()
        self.dyn.reset_motor()

        # Reset ball sensor staleness
        self._ball_obs_countdown = 1  # first step gets fresh reading
        self._last_ball_obs = np.array([ball_pos, ball_vel], dtype=np.float32)

        obs = self.get_obs()
        obs = np.clip(obs, self.observation_space.low,
                      self.observation_space.high)

        if self.render_mode == "human":
            self.render()

        return obs, {}
