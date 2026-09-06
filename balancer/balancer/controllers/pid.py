"""PID controller for ball-on-arc balancing."""

import numpy as np
from balancer.controllers.base import BaseController


class PIDController(BaseController):
    """
    PI-D controller with wall-proximity override.

    Control law::

        u = Kp·θ + Ki·∫θ dt + Kd·θ̇

    Wall-proximity override (when wall_margin > 0):
        When cart is near a rail limit and the PID command would push
        further into the wall, override with a fixed command away from
        the wall. This handles same-side (cart+ball) configurations
        that are unrecoverable with proportional gains alone.

    Args:
        Kp:              Proportional gain on ball angle.
        Ki:              Integral gain.
        Kd:              Gain on measured ball angular velocity.
        Ts:              Control period [s].
        action_type:     "discrete" or "cont".
        integral_limit:  Symmetric clamp on integrator state.
        wall_margin:     Distance from wall to activate override [m].
                         Set to 0 to disable (vanilla PID).
        override_gain:   Fixed command magnitude during override.
    """

    def __init__(
        self,
        Kp: float = 19.782,
        Ki: float = 0.0,
        Kd: float = 1.367,
        Ts: float = 0.05,
        action_type: str = "cont",
        integral_limit: float = 5.0,
        wall_margin: float = 0.12,
        override_gain: float = 0.5,
    ):
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd
        self.Ts = Ts
        self.action_type = action_type
        self.integral_limit = integral_limit
        self.wall_margin = wall_margin
        self.override_gain = override_gain
        self.model_name = "PID"
        self._integral = 0.0

    def reset(self):
        self._integral = 0.0

    def step(self, state, cart_limit):
        theta = state[2]
        theta_dot = state[3]

        # -- 1. Integrate --
        self._integral += theta * self.Ts
        self._integral = np.clip(
            self._integral, -self.integral_limit, self.integral_limit)

        # -- 2. PI-D control law --
        u = (self.Kp * theta
             + self.Ki * self._integral
             + self.Kd * theta_dot)

        # -- 3. Wall-proximity override --
        # When cart is near a rail limit and PID would push further into
        # the wall, override with motion away from wall.
        if self.wall_margin > 0:
            cart_pos = state[0]
            near_right = cart_pos > (cart_limit - self.wall_margin)
            near_left = cart_pos < -(cart_limit - self.wall_margin)
            if near_right and u > 0:
                u = -self.override_gain
            elif near_left and u < 0:
                u = self.override_gain

        # -- 4. Saturate --
        u_unsat = u
        u = float(np.clip(u, -1, 1))

        # -- 5. Anti-windup --
        if abs(self.Ki) > 1e-12 and abs(u_unsat) > 1:
            self._integral -= theta * self.Ts

        return u
