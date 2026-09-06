"""
Nonlinear dynamics of ball-on-arc system.

Usage:
    from dynamics import dynamics_first_order, DEFAULT_PARAMS
    
    state_dot = dynamics_first_order(state, input, DEFAULT_PARAMS)
    
    # Velocity-input (jog mode) dynamics:
    state_dot = dynamics_velocity_input(state, v_cmd, DEFAULT_PARAMS)
"""

import numpy as np
from balancer.core.params import DEFAULT_PARAMS
from balancer.hardware.constants import SYSTEM


# ============================================================================

# GEOMETRY HELPERS

# ============================================================================

def _alpha(theta, d, k):
    """Surface gradient from Gaussian dip: a(θ) = 2dk·θ·exp(-kθ²)"""
    return 2.0 * d * k * theta * np.exp(-k * theta**2)


def _alpha_p(theta, d, k):
    """Derivative of surface gradient: a'(θ) = 2dk·exp(-kθ²)·(1 - 2kθ²)"""
    return 2.0 * d * k * np.exp(-k * theta**2) * (1.0 - 2.0 * k * theta**2)


def _C(theta, R, d, k):
    """Geometric coefficient C(θ) from arc curvature and dip"""
    a = _alpha(theta, d, k)
    return R**2 - 2.0 * R * a * np.sin(theta) + a**2


def _Ceff(theta, R, d, k, I, m, r):
    """Effective moment coefficient: C_eff(θ) = C(θ) + (I/m)·(R/r)²"""
    return _C(theta, R, d, k) + (I/m) * (R/r)**2


def _Ceff_p(theta, R, d, k):
    """Derivative of effective moment coefficient: C'_eff(θ)"""
    a = _alpha(theta, d, k)
    ap = _alpha_p(theta, d, k)
    return -2.0 * R * (ap * np.sin(theta) + a * np.cos(theta)) + 2.0 * a * ap


# ============================================================================

# FORCE-INPUT DYNAMICS  (original - used by LQR/MPC/NMPC/SMC tuning)

# ============================================================================

def dynamics_first_order(s, u, params):
    """
    Full nonlinear dynamics with force input: ṡ = f(s, u)

    State: s = [x, ẋ, θ, θ̇] where
        x    : cart position (m)
        ẋ    : cart velocity (m/s)
        θ    : ball angle from vertical (rad)
        θ̇    : ball angular velocity (rad/s)

    Input: u = horizontal force on cart (N)

    Returns: ṡ = [ẋ, ẍ, θ̇, θ̈]

    Derived from Euler-Lagrange equations.
    """
    M, m = params["M"], params["m"]
    R, r = params["R"], params["r"]
    I = (2.0 / 5.0) * m * r**2
    d, sigma = params["d"], params["sigma"]
    g = params.get("g", 9.81)

    x, xdot, th, thdot = s
    k = R**2 / (2.0 * sigma**2)

    a = _alpha(th, d, k)
    ce = _Ceff(th, R, d, k, I, m, r)
    cep = _Ceff_p(th, R, d, k)

    b1 = u + m * R * np.sin(th) * thdot**2
    b2 = -0.5 * cep * thdot**2 - g * (-R * np.sin(th) + a)

    Delta = (M + m) * ce - m * R**2 * np.cos(th)**2
    if abs(Delta) < 1e-10:
        raise ZeroDivisionError("Mass matrix nearly singular.")

    xdd = (ce * b1 - m * R * np.cos(th) * b2) / Delta
    thdd = (-R * np.cos(th) * b1 + (M + m) * b2) / Delta

    return np.array([xdot, xdd, thdot, thdd])


# ============================================================================

# VELOCITY-INPUT DYNAMICS  (jog mode - matches TwinCAT McJogging behaviour)

# ============================================================================

def dynamics_velocity_input(s, v_cmd, params):
    """
    Velocity-input dynamics with physically-correct friction.

    Cart:  ẍ = (v_cmd - ẋ) / τ_v          (servo handles rail friction)
    Ball:  θ̈ from reduced equation + generalised friction Q_θ/m
    """
    R, r = params["R"], params["r"]
    m = params["m"]
    I = (2.0 / 5.0) * m * r**2
    d, sigma = params["d"], params["sigma"]
    g = params.get("g", 9.81)
    tau_v = params.get("tau_v", 0.15)

    # -- Friction parameters --
    # μ_rr: dimensionless rolling-friction coefficient  (steel on PLA ≈ 0.01-0.05)
    mu_rr = params.get("mu_ball_rolling", 0.025)
    # β_v:  viscous damping coefficient [m²/s]  (= b_viscous / m)
    beta_v = params.get("mu_ball_viscous", 0.009)

    x, xdot, th, thdot = s
    k = R**2 / (2.0 * sigma**2)

    # -- Cart: pure first-order lag (NO extra friction) --
    xddot = (v_cmd - xdot) / tau_v

    # -- Ball geometry --
    a = _alpha(th, d, k)
    ce = _Ceff(th, R, d, k, I, m, r)
    cep = _Ceff_p(th, R, d, k)

    # -- Conservative forces --
    b2 = -0.5 * cep * thdot**2 - g * (-R * np.sin(th) + a)

    # -- Generalised friction / m  [m²/s²] --
    #
    # Rolling resistance:  Q_rr/m = μ_rr · g · R · cos(θ) · sign(θ̇)
    #   tanh(θ̇/ε) smooths the sign function for numerical stability
    eps_roll = 0.005                # smoothing width [rad/s]
    Q_rr = mu_rr * g * R * np.cos(th) * np.tanh(thdot / eps_roll)

    # Viscous:  Q_visc/m = β_v · θ̇
    Q_visc = beta_v * thdot

    # -- Ball angular acceleration --
    thdd = (-R * np.cos(th) * xddot + b2 - (Q_rr + Q_visc)) / ce

    return np.array([xdot, xddot, thdot, thdd])


# ============================================================================

# NonLinearDynamics CLASS

# ============================================================================

class NonLinearDynamics:
    """
    Ball-on-arc dynamics with selectable motor model.

    Cart and ball are integrated SEPARATELY (one-way coupling).
    Cart acceleration is tracked internally, never exposed to RL.

    motor_model:
        "first_order" : ẍ = (v_cmd - ẋ)/τ_v
        "s_curve"     : jerk-limited, TwinCAT MC_MoveVelocity
        "7phase"      : TwinCAT 7-phase feedforward profile generator
                        (MC_MoveVelocity with Accel/Decel/Jerk limits)
    """

    def __init__(
        self,
        gravity=9.81,
        cart_mass=0.351,
        ball_mass=0.024,
        arc_radius=2.101,
        ball_radius=0.009,
        force_mag=2,
        tau=0.05,
        kinematics_integrator="rk4",
        action_type="discrete",
        max_cart_velocity=0.9,
        friction_coeff_cart=0.08,
        visc_friction_cart=0,
        velocity_lag_tau=0.15,
        motor_model="first_order",
        jerk_limit=50.0,
        accel_limit=15.0,
        decel_limit=15.0,
        cart_limit=0.7765,
        ball_limit=0.078,       # ±0.08 rad ≈ ±168mm arc length
        ball_restitution=0.3,  # steel on PLA, estimate
        command_delay_steps=1,
        velocity_filter_tau=0.01,
        apply_observation_filter=True,
        enable_cart_friction=False
    ):
        self.gravity = gravity
        self.cart_mass = cart_mass
        self.ball_mass = ball_mass
        self.arc_radius = arc_radius
        self.ball_radius = ball_radius
        self.force_mag = force_mag
        self.tau = tau
        self.kinematics_integrator = kinematics_integrator
        self.action_type = action_type
        self.max_cart_velocity = max_cart_velocity
        self.velocity_lag_tau = velocity_lag_tau
        self.motor_model = motor_model

        self.jerk_limit = jerk_limit
        self.accel_limit = accel_limit
        self.decel_limit = decel_limit

        self.cart_limit = cart_limit
        self.ball_limit = ball_limit
        self.ball_restitution = ball_restitution

        self.enable_cart_friction = enable_cart_friction
        # Internal motor state - NEVER seen by RL
        self._cart_accel = 0.0

        # 7-phase feedforward state
        self._p7_braking = False
        self._p7_vcmd_prev = None

        # Friction / geometry
        # Coulomb friction (m/s² equivalent)
        self.friction_coeff_cart = friction_coeff_cart
        self.viscous_friction_cart = visc_friction_cart  # optional viscous term (1/s)
        self.mu_ball_rolling = 0.025
        self.mu_ball_viscous = 0.009
        self.dip_depth = 0.002
        self.dip_fwhm = 0.020  # actual FWHM ~20mm (measured on hardware)

        # -- Command transport delay --
        # Buffer holds RESOLVED v_cmd values (not raw actions), so 0.0
        # is a true neutral/stop command for both "discrete" and "cont".
        self.command_delay_steps = command_delay_steps
        self._cmd_buffer = [0.0] * command_delay_steps

        # vel obs filter in twincat encoder, 0.01
        self.apply_observation_filter = apply_observation_filter
        self.velocity_filter_tau = velocity_filter_tau
        self._filtered_xdot = None

        # Substeps
        if motor_model == "7phase":
            self._physics_dt = min(tau, 0.002)
        elif motor_model == "s_curve":
            self._physics_dt = min(tau, 0.005)
        else:
            self._physics_dt = min(tau, 0.01)
        self._n_sub = max(1, round(tau / self._physics_dt))
        self._actual_dt = tau / self._n_sub

    # --- Properties (same as your original) ---------------

    @property
    def dip_params(self):
        return {
            "M": self.cart_mass, "m": self.ball_mass,
            "R": self.arc_radius, "r": self.ball_radius,
            "d": self.dip_depth,
            "sigma": self.dip_fwhm / 2.354820045,
            "g": self.gravity,
            "tau_v": self.velocity_lag_tau,
            "friction_coeff_cart": self.friction_coeff_cart,
            "mu_ball_rolling": self.mu_ball_rolling,
            "mu_ball_viscous": self.mu_ball_viscous,
        }

    def reset_motor(self, accel=0.0):
        self._cart_accel = float(accel)
        self._filtered_xdot = None
        self._p7_braking = False
        self._p7_vcmd_prev = None

    @property
    def cart_accel(self):
        return self._cart_accel

    def reset_cmd_buffer(self, prev_cmds=None):
        """Initialize delay buffer. Call before each rollout/episode.

        `prev_cmds`, if given, are raw actions (same as passed to __call__);
        they are resolved to v_cmd here since the buffer stores v_cmd.
        """
        if self.command_delay_steps == 0:
            self._cmd_buffer = []
            return
        if prev_cmds is not None:
            # Take last N entries from provided history, resolved to v_cmd
            buf = [self._resolve_vcmd(c) for c in prev_cmds]
            self._cmd_buffer = buf[-self.command_delay_steps:]
            # Pad front with neutral v_cmd if not enough history
            while len(self._cmd_buffer) < self.command_delay_steps:
                self._cmd_buffer.insert(0, 0.0)
        else:
            self._cmd_buffer = [0.0] * self.command_delay_steps

    # --- CART INTEGRATION (independent of ball) -----------

    # def _cart_substep_first_order(self, x, xdot, v_cmd, dt):
    #    """First-order lag: ẍ = (v_cmd - ẋ)/τ_v"""
    #    xddot = (v_cmd - xdot) / self.velocity_lag_tau
    #    xdot_new = xdot + dt * xddot
    #    x_new = x + dt * xdot_new  # semi-implicit for stability
    #    return x_new, xdot_new, xddot

    def _cart_substep_first_order(self, x, xdot, v_cmd, dt):
        tau = self.velocity_lag_tau
        decay = np.exp(-dt / tau)

        # Coulomb friction (acceleration units, opposes motion)
        if self.enable_cart_friction:
            friction = self.friction_coeff_cart * \
                np.sign(xdot) if abs(xdot) > 1e-6 else 0.0
            friction += self.viscous_friction_cart * xdot
        else:
            friction = 0.0

        # Effective steady-state velocity considering friction
        v_eff = v_cmd - friction * tau

        # Exact integration of first-order ODE with constant v_eff over dt
        xdot_new = v_eff + (xdot - v_eff) * decay
        x_new = x + v_eff * dt + (xdot - v_eff) * tau * (1.0 - decay)

        # Effective acceleration during the substep (for ball dynamics)
        xddot = (xdot_new - xdot) / dt

        return x_new, xdot_new, xddot

    def _cart_substep_scurve(self, x, xdot, v_cmd, dt):
        # Compute friction (same as first-order)
        if self.enable_cart_friction:
            friction = self.friction_coeff_cart * \
                np.sign(xdot) if abs(xdot) > 1e-6 else 0.0
            friction += self.viscous_friction_cart * xdot
        else:
            friction = 0.0

        vel_error = v_cmd - xdot

        # Desired acceleration from velocity error (before friction)
        a_desired = vel_error / self.velocity_lag_tau
        if vel_error >= 0:
            a_desired = min(a_desired, self.accel_limit)
        else:
            a_desired = max(a_desired, -self.decel_limit)

        # Apply friction to desired acceleration
        a_desired -= friction

        # Jerk-limited acceleration change
        a_error = a_desired - self._cart_accel
        jerk = np.clip(a_error / dt, -self.jerk_limit, self.jerk_limit)

        # Exact integration (constant jerk within substep)
        a_old = self._cart_accel
        a_new = a_old + jerk * dt
        a_new = np.clip(a_new, -self.decel_limit, self.accel_limit)

        # Cart acceleration for ball dynamics = average over substep
        xddot = a_old + 0.5 * jerk * dt

        # Position/velocity (exact for constant jerk)
        x_new = x + xdot * dt + 0.5 * a_old * dt**2 + (1.0/6.0) * jerk * dt**3
        xdot_new = xdot + a_old * dt + 0.5 * jerk * dt**2

        # Clamp velocity to motor limits
        xdot_new = np.clip(xdot_new, -self.max_cart_velocity,
                           self.max_cart_velocity)

        self._cart_accel = a_new
        return x_new, xdot_new, xddot

    def _cart_substep_7phase(self, x, xdot, v_cmd, dt):
        """
        TwinCAT 7-phase feedforward velocity profile.

        Matches MC_MoveVelocity + MC_Aborting with
        SetpointGeneratorType = '7 Phases (optimized)'.

        For velocity steps where peak accel < accel_limit (our case),
        the profile is triangular in acceleration:
          Phase 1: jerk = +j  (accel ramps up)
          Phase 2: jerk = -j  (accel ramps back to 0 at target v)

        Jerk selection via switching parabola in (a, Δv) plane:
            v_brake = a·|a| / (2j)
        Phase-commit flag prevents chatter at the boundary.
        Exact sub-substep splitting at the boundary eliminates
        discretisation lag.

        tau_v is UNUSED - the profile IS the dynamics.
        """
        j = self.jerk_limit
        a = self._cart_accel
        v_err = v_cmd - xdot

        # -- Reset phase on command change (MC_Aborting) --
        if (self._p7_vcmd_prev is None
                or abs(v_cmd - self._p7_vcmd_prev) > 0.01):
            self._p7_braking = False
            self._p7_vcmd_prev = v_cmd

        # -- Snap to target --
        if abs(v_err) < 0.002 and abs(a) < 0.2:
            xddot = (v_cmd - xdot) / dt if dt > 1e-12 else 0.0
            x_new = x + 0.5 * (xdot + v_cmd) * dt
            self._cart_accel = 0.0
            self._p7_braking = False
            return x_new, v_cmd, xddot

        sgn = np.sign(v_err) if abs(v_err) > 1e-6 else 0.0
        jerk_val = 0.0
        do_split = False
        t_sw = dt

        if sgn == 0 and abs(a) > 0.01:
            # At target velocity but residual accel -> damp it
            jerk_val = -np.sign(a) * j

        elif abs(a) > 0.01 and sgn != 0 and np.sign(a) != sgn:
            # Accel opposes target -> correct it first
            jerk_val = sgn * j

        elif self._p7_braking:
            # Committed to braking after switching curve
            if abs(a) > 0.02:
                jerk_val = -np.sign(a) * j
            else:
                # a ≈ 0 -> braking done; new mini-profile if needed
                self._p7_braking = False
                jerk_val = sgn * j if abs(v_err) > 0.002 else 0.0

        else:
            # Check switching curve
            v_brake = 0.5 * a * abs(a) / j if abs(a) > 1e-8 else 0.0
            v_remaining = v_err - v_brake

            if sgn != 0 and np.sign(v_remaining) != sgn:
                # Past switching curve -> start braking
                jerk_val = -np.sign(a) * j if abs(a) > 0.01 else 0.0
                self._p7_braking = True
            else:
                # Below switching curve -> accelerate
                jerk_val = sgn * j

                # Exact switching time within this substep
                disc = 2.0 * a * a + 4.0 * j * abs(v_err)
                if disc >= 0.0:
                    t_sw_cand = (
                        -2.0 * a * sgn + np.sqrt(disc)
                    ) / (2.0 * j)
                    if 1e-9 < t_sw_cand < dt - 1e-9:
                        do_split = True
                        t_sw = t_sw_cand

        # -- Sub-substep split at exact switching boundary --
        if do_split:
            j1 = sgn * j
            j2 = -sgn * j
            t1 = t_sw
            t2 = dt - t_sw

            a1 = a + j1 * t1
            v1 = xdot + a * t1 + 0.5 * j1 * t1 * t1
            x1 = (x + xdot * t1 + 0.5 * a * t1 * t1
                  + (1.0 / 6.0) * j1 * t1**3)

            a_new = a1 + j2 * t2
            v_new = v1 + a1 * t2 + 0.5 * j2 * t2 * t2
            x_new = (x1 + v1 * t2 + 0.5 * a1 * t2 * t2
                     + (1.0 / 6.0) * j2 * t2**3)

            a_new = np.clip(a_new, -self.decel_limit, self.accel_limit)
            v_new = np.clip(v_new,
                            -self.max_cart_velocity,
                            self.max_cart_velocity)

            self._cart_accel = float(a_new)
            self._p7_braking = True
            xddot = (v_new - xdot) / dt
            return x_new, v_new, xddot

        # -- Standard single-jerk integration --
        a_new = a + jerk_val * dt
        a_new = np.clip(a_new, -self.decel_limit, self.accel_limit)
        actual_jerk = (a_new - a) / dt if dt > 1e-12 else 0.0

        if self.enable_cart_friction:
            fric = (self.friction_coeff_cart * np.sign(xdot)
                    if abs(xdot) > 1e-6 else 0.0)
            fric += self.viscous_friction_cart * xdot
        else:
            fric = 0.0

        a_eff = a - fric
        xddot = a_eff + 0.5 * actual_jerk * dt
        v_new = xdot + a_eff * dt + 0.5 * actual_jerk * dt**2
        x_new = (x + xdot * dt + 0.5 * a_eff * dt**2
                 + (1.0 / 6.0) * actual_jerk * dt**3)

        v_new = np.clip(v_new,
                        -self.max_cart_velocity,
                        self.max_cart_velocity)
        self._cart_accel = float(a_new)
        return x_new, v_new, xddot

    # --- BALL INTEGRATION (uses cart ẍ as known input) ----

    def _ball_derivatives(self, th, thdot, xddot):
        """
        Ball angular acceleration from reduced equation.

        θ̈ = (-R·cos(θ)·ẍ + b2 - friction) / C_eff(θ)

        Cart acceleration ẍ is a KNOWN INPUT, not a coupled unknown.
        """
        p = self.dip_params
        R, r = p["R"], p["r"]
        m = p["m"]
        I = (2.0 / 5.0) * m * r**2
        d, sigma = p["d"], p["sigma"]
        g = p["g"]
        k = R**2 / (2.0 * sigma**2)

        a = _alpha(th, d, k)
        ce = _Ceff(th, R, d, k, I, m, r)
        cep = _Ceff_p(th, R, d, k)

        b2 = -0.5 * cep * thdot**2 - g * (-R * np.sin(th) + a)

        # Ball friction (with g·R·cos(θ) factor)
        eps_roll = 0.005
        Q_rr = p["mu_ball_rolling"] * g * R * \
            np.cos(th) * np.tanh(thdot / eps_roll)
        Q_visc = p["mu_ball_viscous"] * thdot

        thdd = (-R * np.cos(th) * xddot + b2 - (Q_rr + Q_visc)) / ce
        return thdd

    def _ball_substep_rk4(self, th, thdot, xddot, dt):
        """RK4 on ball [θ, θ̇] with ẍ as constant known input."""
        def f(th_, thdot_):
            thdd = self._ball_derivatives(th_, thdot_, xddot)
            return np.array([thdot_, thdd])

        s = np.array([th, thdot])
        k1 = f(s[0], s[1])
        k2 = f(s[0] + 0.5*dt*k1[0], s[1] + 0.5*dt*k1[1])
        k3 = f(s[0] + 0.5*dt*k2[0], s[1] + 0.5*dt*k2[1])
        k4 = f(s[0] + dt*k3[0], s[1] + dt*k3[1])
        s_new = s + (dt/6.0) * (k1 + 2*k2 + 2*k3 + k4)
        return s_new[0], s_new[1]

    # --- MAIN STEP ----------------------------------------

    def _resolve_vcmd(self, action):
        if self.action_type == "discrete":
            return {0: -self.max_cart_velocity,
                    1: 0.0,
                    2: +self.max_cart_velocity}[int(action)]
        elif self.action_type == "cont":
            return float(np.clip(action,
                                 -self.max_cart_velocity,
                                 self.max_cart_velocity))
        else:
            raise ValueError(f"Unknown action_type: {self.action_type}")

    def __call__(self, state, action):
        """
        Advance by one control step with physical limit handling.
        """
        # -- Apply command delay --
        # Buffer holds resolved v_cmd (not raw actions) so the neutral
        # 0.0 fill value is a true stop for both discrete and cont actions.
        if self.command_delay_steps > 0:
            v_cmd = self._cmd_buffer.pop(0)
            self._cmd_buffer.append(self._resolve_vcmd(action))
        else:
            v_cmd = self._resolve_vcmd(action)

        x, xdot, th, thdot = state
        dt = self._actual_dt

        for _ in range(self._n_sub):

            # ================================================
            # CART SOFT LIMITS (TwinCAT command saturation)
            # ================================================
            # TwinCAT refuses velocity commands that would push
            # the cart further into the limit. The drive still
            # decelerates normally - it just ignores the command.

            effective_v_cmd = v_cmd
            if x >= self.cart_limit and v_cmd > 0:
                effective_v_cmd = 0.0    # refuse positive command
            elif x <= -self.cart_limit and v_cmd < 0:
                effective_v_cmd = 0.0    # refuse negative command

            # -- Integrate cart --
            if self.motor_model == "7phase":
                x, xdot, xddot = self._cart_substep_7phase(
                    x, xdot, effective_v_cmd, dt)
            elif self.motor_model == "s_curve":
                x, xdot, xddot = self._cart_substep_scurve(
                    x, xdot, effective_v_cmd, dt)
            else:
                x, xdot, xddot = self._cart_substep_first_order(
                    x, xdot, effective_v_cmd, dt)

            # Hard clamp cart position (in case deceleration overshoots)
            if abs(x) > self.cart_limit:
                x = np.clip(x, -self.cart_limit, self.cart_limit)
                if (x >= self.cart_limit and xdot > 0) or \
                   (x <= -self.cart_limit and xdot < 0):
                    xdot = 0.0    # cart stops at wall
                    xddot = 0.0   # no acceleration into wall
                    self._cart_accel = 0.0
                    self._p7_braking = False

            # -- Integrate ball --
            th, thdot = self._ball_substep_rk4(th, thdot, xddot, dt)

            # ================================================
            # BALL WALL COLLISION (physical walls on cart)
            # ================================================
            # Steel ball hitting PLA wall: partially inelastic
            # Coefficient of restitution for steel on PLA ≈ 0.2-0.4

            if abs(th) > self.ball_limit:
                th = np.clip(th, -self.ball_limit, self.ball_limit)
                # Reverse velocity with energy loss
                if np.sign(th) * thdot > 0:      # <- moving into wall
                    thdot = -self.ball_restitution * thdot

        true_state = np.array([x, xdot, th, thdot])

        # -- Apply velocity observation filter (matches NC PT1) --
        if self.apply_observation_filter and self.velocity_filter_tau > 0:
            if self._filtered_xdot is None:
                self._filtered_xdot = xdot  # initialize on first call
            alpha = self.tau / (self.velocity_filter_tau + self.tau)
            self._filtered_xdot = ((1.0 - alpha) * self._filtered_xdot
                                   + alpha * xdot)
            # Return filtered velocity (what the sensor would report)
            return np.array([x, self._filtered_xdot, th, thdot])

        return true_state
