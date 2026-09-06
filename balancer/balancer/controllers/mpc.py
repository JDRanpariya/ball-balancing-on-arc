import osqp
import numpy as np
import scipy.sparse as sp
from balancer.controllers.base import BaseController
from balancer.core.linear_dynamics import DEFAULT_PARAMS, DEFAULT_TS,  get_discrete_system_velocity, get_discrete_system


class MPCController(BaseController):
    def __init__(self, Q=None, R=None, params=None, Ts=None, N=10,
                 umin=-1, umax=1, action_type="discrete", tau_v=0.15,
                 wall_margin=0.12, override_gain=0.5,
                 cart_constraint=True):
        """
        Linear Model Predictive Controller with input constraints.

        Solves a finite-horizon optimal control problem at each timestep using the
        linearized system dynamics. Formulates MPC as a Quadratic Program (QP) and
        solves efficiently using OSQP (Operator Splitting QP) solver.

        Cost function: J = Σ(||x_k - x_ref||²_Q + ||u_k||²_R) + ||x_N||²_Qf
        Subject to: x_{k+1} = Ad*x_k + Bd*u_k, umin ≤ u_k ≤ umax

        Args:
            Q (np.ndarray, optional): State cost matrix (4x4). Default diag(0.001,0,40,0).
            R (np.ndarray, optional): Input cost matrix (1x1). Default [[0.017]].
            params (dict, optional): System parameters. Uses DEFAULT_PARAMS if None.
            Ts (float, optional): Sampling period. Uses DEFAULT_TS if None.
            N (int): Prediction horizon length. Default 10.
            umin (float): Minimum control input. Default -1.0.
            umax (float): Maximum control input. Default 1.0.
            action_type (str): "discrete" or "cont". Default "discrete".

        Attributes:
            A_d, B_d (np.ndarray): Discrete-time system matrices
            Q, R, Qf (np.ndarray): Cost matrices (Qf = 2*Q for terminal cost)
            N (int): Prediction horizon
            prob (osqp.OSQP): Configured QP solver
            model_name (str): Identifier (e.g., "MPC_Q0_0_10_0_R0.20_N50")

        Example:
            >>> # Standard MPC with 50-step horizon
            >>> controller = MPCController(N=50, action_type="discrete")

            >>> # Aggressive control (high Q, low R)
            >>> Q = np.diag([0, 1, 40, 1])
            >>> R = np.array([[0.05]])
            >>> controller = MPCController(Q=Q, R=R, N=30)

            >>> # Control loop
            >>> for t in range(1000):
            ...     action = controller.step(state, cart_limit=2.4)
            ...     state, reward, done, _, _ = env.step(action)

        Performance:
            Typical solve time: 1-5ms on modern CPU (N=50)
            Suitable for real-time control at 50Hz

        Note:
            Uses warm-starting for faster convergence across timesteps.
            For nonlinear systems, use NMPC instead.

        References:
            Stellato, B., et al. (2020). "OSQP: An operator splitting solver for
            quadratic programs." Mathematical Programming Computation, 12(4), 637-672.
        """
        self.params = params if params is not None else DEFAULT_PARAMS
        self.Ts = Ts if Ts is not None else DEFAULT_TS
        self.N = N
        self.umin = umin
        self.umax = umax
        self.action_type = action_type
        self.tau_v = tau_v
        self.wall_margin = wall_margin
        self.override_gain = override_gain
        self.cart_constraint = cart_constraint
        self.model_name = "MPC"

        # Velocity-input discrete model - matches hardware plant
        self.A_d, self.B_d = get_discrete_system_velocity(
            self.params, self.Ts, tau_v=self.tau_v
        )

        # Q[0,0]=0 is fine - cart limits are enforced as hard constraints
        self.Q = Q if Q is not None else np.diag([0.001, 0.0, 40.0, 0.0])
        self.R = R if R is not None else np.array([[0.017]])
        self.Qf = self.Q * 2

        q_str = "_".join([f"{q:.0f}" for q in np.diag(self.Q)])
        self.model_name = f"MPC_Q{q_str}_R{self.R[0,0]:.2f}_N{N}_vel"

        (self.A_bar, self.B_bar,
         self.Q_block, self.R_block,
         self.A_cart, self.B_cart) = self.build_mpc_matrices()

        self.prob = self.setup_osqp()
        self.last_u = np.zeros(self.N)

        print(f"[MPC] velocity-input plant  tau_v={tau_v:.3f}s")
        print(f"[MPC] cart position enforced as hard QP constraint over horizon")
        print(f"[MPC] {self.model_name}")

    def build_mpc_matrices(self):
        n = self.A_d.shape[0]
        m = self.B_d.shape[1]
        N = self.N

        # Build A_bar and B_bar
        A_bar = np.zeros((N*n, n))
        B_bar = np.zeros((N*n, N*m))
        for i in range(N):
            A_bar[i*n:(i+1)*n, :] = np.linalg.matrix_power(self.A_d, i+1)
            for j in range(i+1):
                B_bar[i*n:(i+1)*n, j*m:(j+1) *
                      m] = np.linalg.matrix_power(self.A_d, i-j) @ self.B_d

        # Block diagonal Q_bar and R_bar
        Q_block = sp.block_diag([self.Q]*(N-1) + [self.Qf])
        R_block = sp.block_diag([self.R]*N)

        # -- Cart position rows in the predicted state vector -----------------
        # Predicted states: X = A_bar @ x0 + B_bar @ U
        # Cart position is state index 0, so rows 0, n, 2n, ... in X
        cart_rows = np.arange(0, N * n, n)           # shape (N,)
        A_cart = A_bar[cart_rows, :]              # (N, n)  - free response
        B_cart = B_bar[cart_rows, :]              # (N, N)  - input response

        return A_bar, B_bar, Q_block, R_block, A_cart, B_cart

    def setup_osqp(self):
        N = self.N
        m = self.B_d.shape[1]
        n_u = N * m

        # Hessian
        # np.errstate suppresses divide-by-zero warnings when Q has zero diagonal
        # entries (e.g. Q[1,1]=0 for xdot in paper A3 tuning); OSQP handles the
        # resulting sparse H correctly.
        with np.errstate(divide='ignore', invalid='ignore', over='ignore'):
            H = self.B_bar.T @ self.Q_block @ self.B_bar + self.R_block
        H = np.nan_to_num(H, nan=0.0, posinf=0.0, neginf=0.0)
        H = (H + H.T) / 2.0
        H_sp = sp.csc_matrix(H)

        # Constraint matrix:
        #   rows 0..N-1   : input constraints   I @ U in [umin, umax]
        #   rows N..2N-1  : cart  constraints   B_cart @ U in [lb_cart, ub_cart]
        #                   bounds updated every step because they depend on x0
        G_input = sp.eye(n_u, format='csc')
        if self.cart_constraint:
            G_cart = sp.csc_matrix(self.B_cart)         # (N, n_u)
            G = sp.vstack([G_input, G_cart], format='csc')
            lb = np.concatenate([np.full(n_u, self.umin), np.full(N, -1e9)])
            ub = np.concatenate([np.full(n_u, self.umax), np.full(N,  1e9)])
        else:
            G = G_input
            lb = np.full(n_u, self.umin)
            ub = np.full(n_u, self.umax)

        prob = osqp.OSQP()
        prob.setup(
            P=H_sp, q=np.zeros(n_u), A=G, l=lb, u=ub,
            verbose=False,
            warm_starting=True,
            eps_abs=1e-4, eps_rel=1e-4,   # slightly relaxed for speed
            max_iter=2000,
        )
        return prob

    def reset(self):
        self.last_u = np.zeros(self.N)

    def step(self, state, cart_limit):
        x = np.asarray(state, dtype=float)
        N = self.N
        n = self.A_d.shape[0]

        # -- Linear cost term ------------------------------------------------
        x_ref = np.zeros(n)
        x0_col = x.reshape(n, 1)
        X_ref = np.kron(np.ones(N), x_ref).reshape(N * n, 1)
        diff = self.A_bar @ x0_col - X_ref
        f = (self.B_bar.T @ self.Q_block @ diff).flatten()

        # -- Cart constraint bounds (state-dependent, updated every step) ----
        #
        # Predicted cart positions: x_cart_k = A_cart[k] @ x0 + B_cart[k] @ U
        # Constraint: -cart_limit ≤ x_cart_k ≤ cart_limit
        # -> -cart_limit - A_cart @ x0 ≤ B_cart @ U ≤ cart_limit - A_cart @ x0
        #
        n_u = N * self.B_d.shape[1]
        if self.cart_constraint:
            # This is what makes MPC plan ahead:
            #   "If I go right now I'll hit the wall in k steps - so I plan to
            #    turn around before that, even if theta increases temporarily."
            # (N,) free-response cart positions
            free_cart = self.A_cart @ x
            lb_cart = -cart_limit - free_cart
            ub_cart = cart_limit - free_cart
            lb = np.concatenate([np.full(n_u, self.umin), lb_cart])
            ub = np.concatenate([np.full(n_u, self.umax), ub_cart])
        else:
            lb = np.full(n_u, self.umin)
            ub = np.full(n_u, self.umax)

        self.prob.update(q=f, l=lb, u=ub)

        # -- Warm-start -------------------------------------------------------
        u_guess = np.roll(self.last_u, -1)
        u_guess[-1] = self.last_u[-1]
        self.prob.warm_start(x=u_guess)

        # -- Solve ------------------------------------------------------------
        res = self.prob.solve()

        if res.info.status in ("solved", "solved_inaccurate"):
            u_seq = res.x
        else:
            # Solver failed (infeasible) - fall back to zero
            u_seq = np.zeros(n_u)

        u = float(u_seq[0])

        # Store shifted sequence for next warm-start
        self.last_u[:-1] = u_seq[1:N]
        self.last_u[-1] = u_seq[N - 1]

        # -- Wall-proximity override (when enabled) ------------------------
        # Uses the ACTUAL cart_limit (physical wall), independent of QP constraint
        if self.wall_margin > 0:
            cart_pos = x[0]
            wall_lim = cart_limit if not hasattr(self, 'wo_cart_limit') else self.wo_cart_limit
            near_right = cart_pos > (wall_lim - self.wall_margin)
            near_left = cart_pos < -(wall_lim - self.wall_margin)
            if near_right and u > 0:
                u = -self.override_gain
            elif near_left and u < 0:
                u = self.override_gain

        # Just clip for safety against solver numerical error
        u = float(np.clip(u, -1.0, 1.0))

        return u