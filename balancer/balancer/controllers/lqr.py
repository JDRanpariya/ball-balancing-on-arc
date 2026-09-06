import numpy as np
from balancer.controllers.base import BaseController
from balancer.core.linear_dynamics import DEFAULT_PARAMS, DEFAULT_TS, get_discrete_system, get_discrete_system_velocity


class LQRController(BaseController):
    def __init__(self, K=None, Q=None, R=None, action_type="discrete",
                 params=None, Ts=None, tau_v=0.15,
                 wall_margin=0.12, override_gain=0.5):
        from scipy.linalg import solve_discrete_are

        self.action_type = action_type
        self.params = params if params is not None else DEFAULT_PARAMS
        self.Ts = Ts if Ts is not None else DEFAULT_TS
        self.tau_v = tau_v
        self.wall_margin = wall_margin
        self.override_gain = override_gain
        self.model_name = "LQR"

        # Always expose both models for reference, but gains use velocity model
        self.Ad,     self.Bd = get_discrete_system(self.params, self.Ts)
        self.Ad_vel, self.Bd_vel = get_discrete_system_velocity(
            self.params, self.Ts, tau_v=self.tau_v
        )

        if K is not None:
            self.K = np.array(K)
            self.model_name = "LQR"
        elif Q is not None and R is not None:
            Q = np.array(Q)
            R = np.array(R).reshape(1, 1)
            P = solve_discrete_are(self.Ad_vel, self.Bd_vel, Q, R)
            self.K = (
                np.linalg.inv(self.Bd_vel.T @ P @ self.Bd_vel + R)
                @ (self.Bd_vel.T @ P @ self.Ad_vel)
            ).flatten()
            q_str = "_".join([f"{q:.0f}" for q in np.diag(Q)])
            self.model_name = f"LQR_Q{q_str}_R{R[0,0]:.2f}_vel"
        else:
            Q = np.diag([0, 0, 20, 0])
            R = np.array([[0.1]])
            P = solve_discrete_are(self.Ad_vel, self.Bd_vel, Q, R)
            self.K = (
                np.linalg.inv(self.Bd_vel.T @ P @ self.Bd_vel + R)
                @ (self.Bd_vel.T @ P @ self.Ad_vel)
            ).flatten()
            self.model_name = "LQR"

        print(
            f"[LQR] Initialized with velocity-input plant (tau_v={tau_v:.3f}s)")
        print(f"[LQR] K = {self.K}")
        print(f"[LQR] Model: {self.model_name}")

    def reset(self):
        """
        Reset controller internal state.

        Note: LQR is a stateless controller (pure state feedback u = -Kx),
        so no internal state needs to be reset. The gain matrix K is constant
        and computed once during initialization.
        """
        pass

    def step(self, state, cart_limit):
        x = np.array(state)
        u = float(-self.K @ x)

        # Wall-proximity override (when enabled)
        if self.wall_margin > 0:
            cart_pos = x[0]
            near_right = cart_pos > (cart_limit - self.wall_margin)
            near_left = cart_pos < -(cart_limit - self.wall_margin)
            if near_right and u > 0:
                u = -self.override_gain
            elif near_left and u < 0:
                u = self.override_gain

        u = np.clip(u, -1, 1)
        return float(u)
