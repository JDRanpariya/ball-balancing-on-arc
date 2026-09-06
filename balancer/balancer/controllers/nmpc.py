import numpy as np
import casadi as ca
from balancer.controllers.base import BaseController


class NMPCController(BaseController):
    def __init__(self, action_type="discrete",
                 dyn="old",
                 N=10,
                 dt=0.05,  # 20 Hz
                 Q=None,
                 R=None,
                 wall_margin=0.0,
                 override_gain=0.0,
                 mc=0.351,
                 mb=0.024,
                 R_arc=2.101,
                 g=9.81,
                 d=0.002,
                 FWHM=0.042,
                 r=0.009,
                 I=None,
                 wall_coldstart_margin=0.0):
        """
        Nonlinear MPC controller.

        Args:
            N:                      Horizon length.
            dt:                     Sampling time (s).
            mc, mb, R_arc, g:       System parameters.
            wall_margin:            Wall-proximity override zone (m). 0 = disabled.
            override_gain:          Override velocity command magnitude when near wall.
            wall_coldstart_margin:  If > 0, force a cold IPOPT start whenever
                                    |cart_pos| > cart_limit - wall_coldstart_margin.
                                    This discards the warm-start when the cart is
                                    near a wall, preventing a corrupted previous
                                    solution from biasing the solver after the cart
                                    stops at the physical limit. Disabled by default
                                    (0.0). Typical value: 0.05 (50 mm).
        """

        self.action_type = action_type
        self.dyn = dyn
        self.wall_margin = wall_margin
        self.override_gain = override_gain
        self.wall_coldstart_margin = wall_coldstart_margin

        # label for experiment logging
        self.model_name = "NMPC"

        # Horizon + timestep
        self.N = N
        self.dt = dt

        # Parameters
        self.mc, self.mb, self.R, self.gravity = mc, mb, R_arc, g
        self.d, self.FWHM, self.r = d, FWHM, r
        self.sigma = FWHM/2.354820045
        self.k = self.R**2 / (2.0 * self.sigma**2)

        # default inertia for a solid sphere if not provided:
        if I is None:
            self.I = (2.0/5.0) * self.mb * self.r**2
        else:
            self.I = I

        # State (x1, x2, x3, x4), Input (u)
        nx, nu = 4, 1
        x = ca.SX.sym("x", nx)
        u = ca.SX.sym("u", nu)

        # System dynamics
        F = self._discretize(x, u, dt)

        # Decision variables
        X = ca.SX.sym("X", nx, N+1)   # states
        U = ca.SX.sym("U", nu, N)     # controls

        # Parameters (initial state + reference)
        P = ca.SX.sym("P", nx + nx)   # [x0, x_ref]

        # ---- Cost Weights ----
        if Q is None:
            Q_mat = np.diag([0, 0, 5, 0])
        else:
            Q_mat = np.diag(Q) if np.ndim(Q) == 1 else np.array(Q)
        if R is None:
            R_mat = np.array([[0.05]])
        else:
            R_mat = np.atleast_2d(R)

        self.Q_weights = np.diag(Q_mat)
        self.R_weight = float(R_mat[0, 0])

        Qf = Q_mat * 2     # terminal cost

        # ---- Cost and Constraints ----
        cost = 0
        g = []

        # Initial condition constraint
        g.append(X[:, 0] - P[0:nx])

        # Build horizon
        for k in range(N):
            x_k = X[:, k]
            u_k = U[:, k]
            x_ref = P[nx:2*nx]

            # Stage cost: (x - x_ref)^T Q (x - x_ref) + u^T R u
            cost += ca.mtimes([(x_k - x_ref).T, Q_mat, (x_k - x_ref)]) \
                + ca.mtimes([u_k.T, R_mat, u_k])

            # Dynamics constraint
            x_next = F(x_k, u_k)
            g.append(X[:, k+1] - x_next)

        # Terminal cost
        x_N = X[:, N]
        x_ref = P[nx:2*nx]
        cost += ca.mtimes([(x_N - x_ref).T, Qf, (x_N - x_ref)])

        # Pack into NLP
        opt_vars = ca.vertcat(ca.reshape(X, -1, 1), ca.reshape(U, -1, 1))
        g = ca.vertcat(*g)

        nlp = {"f": cost, "x": opt_vars, "p": P, "g": g}
        self.solver = ca.nlpsol('solver', 'ipopt', nlp, {
            'ipopt.print_level': 0,
            'ipopt.tol': 1e-5,
            'print_time': False,
            'ipopt.max_iter': 500,
        })

        # Store sizes
        self.nx, self.nu = nx, nu
        self.X, self.U, self.P, self.opt_vars, self.g = X, U, P, opt_vars, g

        # Warm-start storage
        self._prev_sol = None
        self._coldstart_count = 0

    def reset(self):
        self._prev_sol = None

    def _dynamics_force(self, x, u):  # New dynamic with dip at center
        """ Continuous nonlinear dynamics (dip model) using CasADi expressions.
            x = [x, xdot, theta, thetadot], u is scalar force.
        """
        x1, x2, th, thdot = x[0], x[1], x[2], x[3]
        mc, mb, R, gravity = self.mc, self.mb, self.R, self.gravity
        d, k = self.d, self.k
        r, I = self.r, self.I

        # alpha and derivative
        alpha = 2.0 * d * k * th * ca.exp(-k * th**2)
        alpha_p = 2.0 * d * k * ca.exp(-k * th**2) * (1.0 - 2.0 * k * th**2)

        # C and C_eff and derivative (rotation term constant in theta)
        C = R**2 - 2.0 * R * alpha * ca.sin(th) + alpha**2
        Ceff = C + (2/5) * (R**2)
        Ceff_p = -2.0 * R * (alpha_p * ca.sin(th) + alpha *
                             ca.cos(th)) + 2.0 * alpha * alpha_p

        # right-hand side vector components
        b1 = u + mb * R * ca.sin(th) * thdot**2
        b2 = -0.5 * Ceff_p * thdot**2 - gravity * (-R * ca.sin(th) + alpha)

        Delta = (mc + mb) * Ceff - mb * R**2 * ca.cos(th)**2
        Delta = ca.fmax(Delta, 1e-9)   # safeguard (keeps solver stable)

        xdd = (Ceff * b1 - mb * R * ca.cos(th) * b2) / Delta
        thdd = (-R * ca.cos(th) * b1 + (mc + mb) * b2) / Delta

        return ca.vertcat(x2, xdd, thdot, thdd)

    def _dynamics_velocity(self, x, v_cmd):
        x1, x2, th, thdot = x[0], x[1], x[2], x[3]
        mc, mb, R, gravity = self.mc, self.mb, self.R, self.gravity
        d, k = self.d, self.k
        r, I = self.r, self.I
        tau_v = 0.15

        alpha = 2.0 * d * k * th * ca.exp(-k * th**2)
        alpha_p = 2.0 * d * k * ca.exp(-k * th**2) * (1.0 - 2.0 * k * th**2)

        C = R**2 - 2.0 * R * alpha * ca.sin(th) + alpha**2
        Ceff = C + (I / mb) * (R / r)**2
        Ceff_p = -2.0 * R * (alpha_p * ca.sin(th) + alpha *
                             ca.cos(th)) + 2.0 * alpha * alpha_p

        xddot = (v_cmd - x2) / tau_v
        b2 = -0.5 * Ceff_p * thdot**2 - gravity * (-R * ca.sin(th) + alpha)

        # -- Ball friction (includes g·R·cos(θ)) --
        mu_rr = 0.025
        beta_v = 0.009
        eps = 0.005
        Q_rr = mu_rr * gravity * R * ca.cos(th) * ca.tanh(thdot / eps)
        Q_visc = beta_v * thdot

        thdd = (-R * ca.cos(th) * xddot + b2 - (Q_rr + Q_visc)) / Ceff

        return ca.vertcat(x2, xddot, thdot, thdd)

    def _discretize(self, x, u, dt):
        fn = self._dynamics_velocity
        k1 = fn(x, u)
        k2 = fn(x + dt/2 * k1, u)
        k3 = fn(x + dt/2 * k2, u)
        k4 = fn(x + dt * k3, u)
        return ca.Function("F", [x, u], [x + dt/6 * (k1 + 2*k2 + 2*k3 + k4)])

    def step(self, state, cart_limit):
        """ Solve MPC and return first control input """
        x0 = state.copy()
        x_ref = np.array([0, 0, 0, 0])
        p = np.concatenate((x0, x_ref)).astype(float)

        nx, nu, N = self.nx, self.nu, self.N
        nX = nx*(N+1)
        nU = nu*N

        # Bounds for states (no restriction here -> -inf, inf)
        lbx = -np.inf * np.ones(nX + nU)
        ubx = np.inf * np.ones(nX + nU)

        # --- Actuator constraints (on U only) ---
        umin, umax = -0.9, 0.9
        lbx[nX:] = umin
        ubx[nX:] = umax

        # --- Cart position: hard constraint over full horizon ---
        for k in range(N + 1):
            cart_idx = k * nx + 0          # state[0] = cart position
            lbx[cart_idx] = -cart_limit
            ubx[cart_idx] = cart_limit

        # Equality constraints (system dynamics)
        lbg = np.zeros(self.g.size()[0])
        ubg = np.zeros(self.g.size()[0])

        # -- Wall-proximity cold-start -----------------------------
        if self.wall_coldstart_margin > 0 and abs(x0[0]) > (cart_limit - self.wall_coldstart_margin):
            self._prev_sol = None
            self._coldstart_count += 1

        # -- Initial guess -----------------------------------------
        if self._prev_sol is None:
            # Cold start: tile actual state, zero inputs
            X_guess = np.tile(x0, N + 1)
            U_guess = np.zeros(nu * N)
            z0 = np.concatenate([X_guess, U_guess])
        else:
            # Warm start: shift previous solution left by one step
            X_prev = self._prev_sol[:nX].reshape(N + 1, nx)
            U_prev = self._prev_sol[nX:].reshape(N, nu)

            X_warm = np.vstack([X_prev[1:], X_prev[-1:]])
            X_warm[0] = x0  # override with actual measurement

            U_warm = np.vstack([U_prev[1:], U_prev[-1:]])

            z0 = np.concatenate([X_warm.flatten(), U_warm.flatten()])

        # Solve NLP
        sol = self.solver(
            x0=ca.DM(z0),
            lbx=ca.DM(lbx),
            ubx=ca.DM(ubx),
            lbg=ca.DM(lbg),
            ubg=ca.DM(ubg),
            p=ca.DM(p)
        )

        sol_x = sol["x"].full().flatten()
        self._prev_sol = sol_x  # store for next warm-start

        U_opt = sol_x[nX:].reshape(self.nu, self.N)
        u = U_opt[:, 0]

        # --- post-processing ---
        u = float(np.asarray(u).flat[0])

        # Wall-proximity override (when enabled)
        if self.wall_margin > 0:
            cart_pos = x0[0]
            near_right = cart_pos > (cart_limit - self.wall_margin)
            near_left = cart_pos < -(cart_limit - self.wall_margin)
            if near_right and u > 0:
                u = -self.override_gain
            elif near_left and u < 0:
                u = self.override_gain

        return float(np.clip(u, -1.0, 1.0))
