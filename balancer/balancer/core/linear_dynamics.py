"""
Linearized system matrices for control design.

This provides Ad, Bd matrices for discrete-time control (LQR, MPC).

Usage:
    from linear_system import get_discrete_system
    
    Ad, Bd = get_discrete_system()  # Uses default params and Ts=0.05s
    
    # Use in LQR
    P = solve_discrete_are(Ad, Bd, Q, R)
    K = inv(Bd.T @ P @ Bd + R) @ (Bd.T @ P @ Ad)
    
    # Use in MPC
    x_next = Ad @ x + Bd @ u
"""

import numpy as np
from scipy.linalg import expm
from balancer.core.params import DEFAULT_PARAMS

DEFAULT_TS = 0.05  # 20 Hz sampling (50 ms)

# ============================================================================

# ANALYTICAL LINEARIZATION AT θ=0

# ============================================================================

def get_continuous_system(params=None):
    """
    Analytical linearization at upright equilibrium (θ = 0).
    
    Derived from Euler-Lagrange equations using small-angle approximation:
        sin(θ) ≈ θ, cos(θ) ≈ 1, θ̇² ≈ 0
    
    Returns: (A, B) for continuous-time LTI system ṡ = As + Bu
             where s = [x, ẋ, θ, θ̇]ᵀ, u = force on cart
    
    Note: This is a CLOSED-FORM solution (no numerical approximation).
          Verification against nonlinear dynamics is done in test_system.py.
    """
    if params is None:
        params = DEFAULT_PARAMS
    
    M, m = params["M"], params["m"]
    R = params["R"]
    d, sigma = params["d"], params["sigma"]
    g = params.get("g", 9.81)

    # Derived constants (see paper Section III-E)
    k = R**2 / (2.0 * sigma**2)          # Gaussian width parameter
    J = (7/5) * R**2                      # Effective inertia (solid sphere)
    Delta0 = (M + m) * J - m * R**2      # Denominator term
    gamma = g * (R - 2.0 * d * k)        # Linearized gravity effect

    # State-space matrices
    A = np.array([
        [0, 1, 0, 0],
        [0, 0, -(m * R * gamma) / Delta0, 0],
        [0, 0, 0, 1],
        [0, 0, (M + m) * gamma / Delta0, 0],
    ], dtype=float)

    B = np.array([
        [0.0],
        [J / Delta0],
        [0.0],
        [-R / Delta0]
    ], dtype=float)
    
    return A, B

def get_continuous_system_velocity(params=None, tau_v=0.15):
    """
    Linearization with cart velocity as input.
    
    Cart dynamics: ẍ = (v_cmd - ẋ) / tau_v
    Ball dynamics: same reduced equation
    
    State: [x, ẋ, θ, θ̇]
    Input: v_cmd (m/s)
    """
    if params is None:
        params = DEFAULT_PARAMS

    M, m  = params["M"], params["m"]
    R     = params["R"]
    d, sigma = params["d"], params["sigma"]
    g     = params.get("g", 9.81)

    k     = R**2 / (2.0 * sigma**2)
    J     = (7/5) * R**2
    gamma = g * (R - 2.0 * d * k)

    # Cart row: ẍ = -ẋ/tau_v + v_cmd/tau_v
    # Ball row: θ̈ = (M+m)*gamma/Ceff * θ - R/Ceff * ẍ_cart ??
    # At θ=0: Ceff = J + (I/m)(R/r)^2 ≈ J for solid sphere on arc
    Ceff0 = J  # linearized at θ=0

    A = np.array([
        [0,          1,             0,           0],
        [0,  -1/tau_v,             0,           0],
        [0,          0,             0,           1],
        [0,  R/(Ceff0*tau_v),  gamma/Ceff0,     0],  # fixed
        ], dtype=float)

    B = np.array([
        [0.0          ],
        [1.0 / tau_v  ],   # v_cmd enters cart acceleration
        [0.0          ],
        [-R/(Ceff0*tau_v)],  # and couples into ball
        ], dtype=float)

    return A, B

# ============================================================================

# DISCRETIZATION (ZOH - Exact for Linear Systems)

# ============================================================================

def get_discrete_system(params=None, Ts=None):
    """
    Zero-order hold (ZOH) discretization of linearized system.
    
    Computes exact discrete-time equivalent:
        x[k+1] = Ad·x[k] + Bd·u[k]
    
    where:
        Ad = exp(A·Ts)
        Bd = ∫₀^Ts exp(A·τ)dτ · B
    
    Args:
        params: System parameters (uses DEFAULT_PARAMS if None)
        Ts: Sampling period in seconds (uses DEFAULT_TS = 0.05s if None)
    
    Returns: (Ad, Bd) discrete-time state-space matrices
    
    Note: ZOH is the standard method for sampled-data control systems.
          It preserves stability and controllability properties.
    """
    if params is None:
        params = DEFAULT_PARAMS
    if Ts is None:
        Ts = DEFAULT_TS
    
    # Get continuous-time system
    A, B = get_continuous_system(params)
    
    # ZOH discretization using matrix exponential
    # Method: Augment [A B; 0 0] and exponentiate
    n = A.shape[0]
    m = B.shape[1]
    M = np.zeros((n + m, n + m))
    M[:n, :n] = A * Ts
    M[:n, n:] = B * Ts
    
    Md = expm(M)
    Ad = Md[:n, :n]
    Bd = Md[:n, n:]
    
    return Ad, Bd

def get_discrete_system_velocity(params=None, Ts=None, tau_v=0.15):
    """
    ZOH discretization of the velocity-input linearized system.

    Uses get_continuous_system_velocity() as the continuous model, then
    applies the same matrix-exponential ZOH method as get_discrete_system().

    Args:
        params:  System parameters (DEFAULT_PARAMS if None)
        Ts:      Sampling period in seconds (DEFAULT_TS if None)
        tau_v:   Cart velocity lag time constant (s), default 0.15

    Returns:
        (Ad, Bd) discrete-time matrices for s[k+1] = Ad s[k] + Bd v_cmd[k]
    """
    if params is None:
        params = DEFAULT_PARAMS
    if Ts is None:
        Ts = DEFAULT_TS

    A, B = get_continuous_system_velocity(params, tau_v=tau_v)

    n = A.shape[0]
    m = B.shape[1]
    M = np.zeros((n + m, n + m))
    M[:n, :n] = A * Ts
    M[:n, n:] = B * Ts

    Md = expm(M)
    Ad = Md[:n, :n]
    Bd = Md[:n, n:]

    return Ad, Bd

# ============================================================================

# CONVENIENCE FUNCTIONS

# ============================================================================

def get_system_info(params=None, Ts=None, tau_v=0.15):
    """
    Get continuous and discrete system matrices for both input types.

    Returns dict with keys:
        'A', 'B'           - force-input continuous (for reference)
        'Ad', 'Bd'         - force-input discrete   (for reference)
        'A_vel', 'B_vel'   - velocity-input continuous
        'Ad_vel', 'Bd_vel' - velocity-input discrete  <- use for LQR/MPC gains
        'params', 'Ts', 'tau_v'
    """
    if params is None:
        params = DEFAULT_PARAMS
    if Ts is None:
        Ts = DEFAULT_TS

    A, B         = get_continuous_system(params)
    Ad, Bd       = get_discrete_system(params, Ts)
    A_vel, B_vel = get_continuous_system_velocity(params, tau_v)
    Ad_vel, Bd_vel = get_discrete_system_velocity(params, Ts, tau_v)

    return {
        'A': A, 'B': B,
        'Ad': Ad, 'Bd': Bd,
        'A_vel': A_vel, 'B_vel': B_vel,
        'Ad_vel': Ad_vel, 'Bd_vel': Bd_vel,
        'params': params, 'Ts': Ts, 'tau_v': tau_v,
    }