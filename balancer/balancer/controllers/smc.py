"""
Sliding Mode Controller for velocity-input ball-on-arc (TwinCAT jog mode).

Hardware model:
    Cart velocity tracks v_cmd via first-order lag: ẋ̇ = (v_cmd - ẋ) / tau_v
    Ball responds to resulting cart acceleration (reduced equation).

Ball dynamics decompose as:
    θ̈ = f_vel(θ, θ̇, ẋ)  +  g_vel(θ) · v_cmd

    f_vel = [b2 - friction + R·cos(θ)·ẋ / tau_v] / Ceff   <- drift term
    g_vel = -R·cos(θ) / (Ceff · tau_v)                    <- input gain (< 0)

Sliding surface:  σ = θ̇ + λ·θ

Equivalent control (sets σ̇ = 0 exactly):
    F = f_vel + λ·θ̇
    G = g_vel
    v_eq = -F / G

Full control law:
    v_cmd = -(F + k·tanh(σ/φ)) / G  -  k_d·ẋ

    First term: SMC drives σ->0 (ball stabilisation).
    Second term: explicit cart-velocity damping eliminates open-loop
                 drift after the ball is balanced.  Without k_d the
                 equivalent control's residual bias (model error,
                 friction) integrates into unbounded cart motion.

Deployed parameters (hardware, 92% SR / 50 trials, Supplementary A3):
    λ=3.0, k=6.0, φ=0.806, k_d=0.5
    wall_margin=0.12 m, override_gain=0.5
"""

import numpy as np
from balancer.controllers.base import BaseController
from balancer.core.dynamics import (
    _alpha,
    _Ceff,
    _Ceff_p,
    DEFAULT_PARAMS,
)

from balancer.hardware.constants import SYSTEM


class SMCController(BaseController):
    """
    Sliding Mode Controller for ball-on-arc with velocity-commanded cart.

    Args:
        lam (float):             Sliding surface slope λ.  Default 3.0 (paper A3).
        k (float):               Switching gain.  Default 6.0 (paper A3).
        phi (float):             Boundary layer thickness for sat().  Default 0.806 (paper A3).
        action_type (str):       "discrete" or "cont".
        params (dict):           System parameters.  Uses DEFAULT_PARAMS if None.
        tau_v (float):           Cart velocity lag time constant (s).
                                 Derived from TwinCAT: 900/15000 = 0.06 s.
        max_cart_velocity (float): Max commanded velocity (m/s).  900 mm/s -> 0.9.
    """

    def __init__(
        self,
        lam=3.0,
        k=6.0,
        phi=0.806,
        action_type="discrete",
        params=None,
        tau_v=0.15,
        lam_x=0.0,
        max_cart_velocity=SYSTEM.MAX_CART_VEL,
        wall_margin=0.12,
        override_gain=0.5,
        sigma_dead_zone=0.0,
        k_d=0.5,
        sensor_offset=0.0,
    ):
        self.lam = lam
        self.k = k
        self.phi = phi
        self.action_type = action_type
        self.params = params if params is not None else DEFAULT_PARAMS
        self.tau_v = tau_v
        self.max_cart_velocity = max_cart_velocity
        self.wall_margin = wall_margin
        self.override_gain = override_gain
        self.sigma_dead_zone = sigma_dead_zone
        self.model_name = "SMC"
        self.lam_x = lam_x
        self.k_d = k_d
        self.sensor_offset = sensor_offset
        self.use_predictor = False  # delay compensation (disabled by default)

        # Cache geometry constants from params (recomputed if params changes)
        self._precompute_constants()

        print(
            f"[SMC] Initialized - λ={lam}, k={k}, φ={phi}, "
            f"tau_v={tau_v:.3f}s, v_max={max_cart_velocity:.2f}m/s"
        )

    # -- Internal helpers -----------------------------------------------------

    def _precompute_constants(self):
        """Cache parameters used in every step."""
        p = self.params
        self._R = p["R"]
        self._d = p["d"]
        self._sig = p["sigma"]
        self._k_g = self._R**2 / (2.0 * self._sig**2)  # Gaussian width
        self._m = p["m"]
        self._r = p["r"]
        self._g = p.get("g", 9.81)

    def _sat(self, x):
        """Smooth saturation function (boundary layer)."""
        return float(np.tanh(x / self.phi))

    def _ball_dynamics_split(self, th, thdot, xdot):
        R, d, k_g = self._R, self._d, self._k_g
        I = (2.0 / 5.0) * self._m * self._r**2

        a = _alpha(th, d, k_g)
        ce = _Ceff(th, R, d, k_g, I, self._m, self._r)
        cep = _Ceff_p(th, R, d, k_g)

        b2 = -0.5 * cep * thdot**2 - self._g * (-R * np.sin(th) + a)

        # -- Ball friction (so equivalent control cancels it) --
        mu_rr = 0.025
        beta_v = 0.009
        eps = 0.005
        friction = (mu_rr * self._g * R * np.cos(th) * np.tanh(thdot / eps)
                    + beta_v * thdot)

        # Full f_vel (includes cart velocity effect)
        f_vel = (b2 - friction + R * np.cos(th) * xdot / self.tau_v) / ce
        # f_vel without cart velocity - for v_eq that doesn't maintain ẋ
        f_vel_zero = (b2 - friction) / ce
        g_vel = -R * np.cos(th) / (ce * self.tau_v)

        if abs(g_vel) < 1e-8:
            g_vel = np.copysign(1e-8, g_vel)
        return f_vel, f_vel_zero, g_vel

    # -- BaseController interface ----------------------------------------------

    def reset(self):
        """Reset internal state."""
        self._prev_u = 0.0

    def _predict_state_1step(self, state, prev_u):
        """
        Predict state 1 step ahead to compensate for command delay.
        Uses the first-order cart model and ball dynamics.
        """
        x, xdot, th, thdot = state
        Ts = 0.05  # control period

        # Cart: first-order lag response to previous command
        v_cmd = prev_u * self.max_cart_velocity
        xddot = (v_cmd - xdot) / self.tau_v
        xdot_pred = xdot + xddot * Ts
        x_pred = x + xdot * Ts

        # Ball: use dynamics split
        f_vel, _, g_vel = self._ball_dynamics_split(th, thdot, xdot)
        thddot = f_vel + g_vel * v_cmd
        thdot_pred = thdot + thddot * Ts
        th_pred = th + thdot * Ts

        return np.array([x_pred, xdot_pred, th_pred, thdot_pred])

    def step(self, state, cart_limit):
        cart_pos = state[0]
        xdot = state[1]
        th = state[2] + self.sensor_offset
        thdot = state[3]

        # 0. Optionally predict state 1 step ahead (delay compensation)
        if self.use_predictor:
            pred = self._predict_state_1step(state, self._prev_u)
            th_p = pred[2]
            thdot_p = pred[3]
            xdot_p = pred[1]
        else:
            th_p = th
            thdot_p = thdot
            xdot_p = xdot

        # 1. Ball-only sliding surface: σ = θ̇ + λ·θ
        #    Guarantees θ->0 on the surface.
        sigma = thdot_p + self.lam * th_p

        # 2. Dynamics decomposition (full f_vel includes ẋ coupling)
        f_vel, _, g_vel = self._ball_dynamics_split(th_p, thdot_p, xdot_p)

        # 3. σ̇ = f_vel + g_vel·v_cmd + λ·θ̇ = F + G·v_cmd
        F = f_vel + self.lam * thdot_p
        G = g_vel

        # Prevent division by near-zero G
        if abs(G) < 1e-6:
            G = np.copysign(1e-6, G)

        # 4. Control law: v_cmd = -(F + k·sat(σ)) / G - k_d·ẋ
        #    SMC term drives σ->0 (ball stabilization)
        #    k_d·ẋ damps cart velocity explicitly (stops runaway)
        v_cmd = -(F + self.k * self._sat(sigma)) / G - self.k_d * xdot_p

        # 5. Normalise to [-1, 1]
        u = float(np.clip(v_cmd / self.max_cart_velocity, -1.0, 1.0))

        # 6. Wall-proximity override (safety only - should rarely fire)
        if self.wall_margin > 0:
            near_right = cart_pos > (cart_limit - self.wall_margin)
            near_left = cart_pos < -(cart_limit - self.wall_margin)
            if near_right and u > 0:
                u = -self.override_gain
            elif near_left and u < 0:
                u = self.override_gain

        # 7. Hard clip for numerical safety
        u = float(np.clip(u, -1.0, 1.0))

        # Store for delay compensation
        self._prev_u = u

        return u
