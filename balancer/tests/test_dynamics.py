"""
System verification and analysis tools.

This file verifies that:
1. Analytical linearization matches numerical linearization
2. System is controllable
3. Sampling rate is appropriate
4. Discretization methods are compared

Run: python test_system.py
"""

import numpy as np
from numpy.linalg import matrix_rank, cond, norm

# Project imports

from balancer.core.dynamics import dynamics_first_order, DEFAULT_PARAMS as PARAMS_DYNAMICS
from balancer.core.linear_dynamics import (
    get_continuous_system, 
    get_discrete_system,
    DEFAULT_PARAMS as PARAMS_LINEAR,
    DEFAULT_TS
)

# ============================================================================

# VERIFICATION TOOLS

# ============================================================================

def linearize_fd(s, u, params, eps=1e-7):
    """
    Finite-difference (numerical) linearization of nonlinear dynamics.
    
    Computes Jacobian: ṡ ≈ A·s + B·u around operating point (s, u)
    
    This is used ONLY for verification, not for control design.
    Accuracy: O(ε²) with central differences.
    """
    n = len(s)
    
    # Jacobian w.r.t. state
    A = np.zeros((n, n))
    for i in range(n):
        d = np.zeros(n)
        d[i] = eps
        fp = dynamics_first_order(s + d, u, params)
        fm = dynamics_first_order(s - d, u, params)
        A[:, i] = (fp - fm) / (2.0 * eps)

    # Jacobian w.r.t. input
    B = ((dynamics_first_order(s, u + eps, params) -
          dynamics_first_order(s, u - eps, params)) / (2.0 * eps)).reshape(-1, 1)
    
    return A, B

def controllability(A, B):
    """
    Check controllability using Kalman rank condition.
    
    Returns: (rank, condition_number)
        rank = n means fully controllable
        condition number indicates numerical sensitivity
    """
    n = A.shape[0]
    W = B
    for i in range(1, n):
        W = np.hstack([W, np.linalg.matrix_power(A, i) @ B])
    return matrix_rank(W), cond(W)

# ============================================================================

# MAIN VERIFICATION

# ============================================================================

# ============================================================================
# PYTEST REGRESSION TESTS   (run: pytest balancer/tests)
# ============================================================================

def test_parameter_consistency():
    """Dynamics and linear-dynamics modules share identical default params."""
    assert PARAMS_DYNAMICS == PARAMS_LINEAR


def test_analytical_matches_numerical_linearization():
    """Analytical linearization at the origin matches finite-difference."""
    params = PARAMS_DYNAMICS
    A_a, B_a = get_continuous_system(params)
    s0 = np.array([0.0, 0.0, 0.0, 0.0])
    A_n, B_n = linearize_fd(s0, 0.0, params, eps=1e-7)
    assert np.max(np.abs(A_a - A_n)) < 1e-5
    assert np.max(np.abs(B_a - B_n)) < 1e-5


def test_system_controllable():
    """Continuous and discrete systems are fully controllable (rank 4)."""
    params = PARAMS_DYNAMICS
    A_a, B_a = get_continuous_system(params)
    rank_c, _ = controllability(A_a, B_a)
    assert rank_c == 4
    Ad, Bd = get_discrete_system(params, DEFAULT_TS)
    rank_d, _ = controllability(Ad, Bd)
    assert rank_d == 4


if __name__ == "__main__":
    print("\n" + "="*70)
    print("BALL-ON-ARC SYSTEM: VERIFICATION & ANALYSIS")
    print("="*70)
    
    # Check parameter consistency
    assert PARAMS_DYNAMICS == PARAMS_LINEAR, "Parameter mismatch between files!"
    params = PARAMS_DYNAMICS
    
    # =========================================================================
    # TEST 1: Analytical vs Numerical Linearization
    # =========================================================================
    print("\n[TEST 1] Analytical vs Numerical Linearization at θ=0")
    print("-" * 70)
    
    # Analytical (from mathematical derivation)
    A_analytical, B_analytical = get_continuous_system(params)
    
    # Numerical (from nonlinear dynamics via finite-difference)
    s0 = np.array([0.0, 0.0, 0.0, 0.0])
    u0 = 0.0
    A_numerical, B_numerical = linearize_fd(s0, u0, params, eps=1e-7)
    
    # Compare
    A_err = np.max(np.abs(A_analytical - A_numerical))
    B_err = np.max(np.abs(B_analytical - B_numerical))
    
    print(f"Max error in A: {A_err:.2e}")
    print(f"Max error in B: {B_err:.2e}")
    
    if A_err < 1e-5 and B_err < 1e-5:
        print("[OK] PASS: Analytical linearization is correct!")
    else:
        print("[FAIL] FAIL: Analytical linearization does not match nonlinear!")
        print("\nA_analytical:")
        print(A_analytical)
        print("\nA_numerical:")
        print(A_numerical)
    
    # =========================================================================
    # TEST 2: Stability Analysis
    # =========================================================================
    print("\n[TEST 2] Stability Analysis")
    print("-" * 70)
    
    # Check stability regime
    k = params["R"]**2 / (2.0 * params["sigma"]**2)
    gamma = params["g"] * (params["R"] - 2.0 * params["d"] * k)
    
    print(f"Stability parameter γ = g(R - 2dk) = {gamma:.3f} m/s²")
    print(f"  R = {params['R']:.3f} m")
    print(f"  2dk = {2*params['d']*k:.3f} m")
    
    if gamma > 0.1:
        print("-> System is UNSTABLE (inverted pendulum-like)")
    elif gamma < -0.1:
        print("-> System is LOCALLY STABLE (dip dominates)")
    else:
        print("-> System is MARGINALLY STABLE")
    
    # Eigenvalues
    eigvals = np.linalg.eigvals(A_analytical)
    print(f"\nContinuous eigenvalues: {eigvals}")
    print(f"Max Re(λ): {np.max(np.real(eigvals)):.6f}")
    
    # =========================================================================
    # TEST 3: Controllability
    # =========================================================================
    print("\n[TEST 3] Controllability")
    print("-" * 70)
    
    rank_c, cond_c = controllability(A_analytical, B_analytical)
    print(f"Controllability matrix:")
    print(f"  Rank: {rank_c}/4 {'[OK]' if rank_c == 4 else '[FAIL]'}")
    print(f"  Condition number: {cond_c:.2e}")
    print(f"  ||B||: {norm(B_analytical):.6f}")
    
    if rank_c < 4:
        print("[FAIL] FAIL: System is not fully controllable!")
    else:
        print("[OK] PASS: System is fully controllable")
    
    # =========================================================================
    # TEST 4: Discrete System
    # =========================================================================
    print("\n[TEST 4] Discrete-Time System (Ts = {:.3f} s)".format(DEFAULT_TS))
    print("-" * 70)
    
    Ad, Bd = get_discrete_system(params, DEFAULT_TS)
    
    print("Ad =")
    print(Ad)
    print("\nBd =")
    print(Bd.flatten())
    
    # Discrete eigenvalues
    eigvals_d = np.linalg.eigvals(Ad)
    print(f"\nDiscrete eigenvalues: {eigvals_d}")
    print(f"Max |λ|: {np.max(np.abs(eigvals_d)):.4f}")
    
    # Discrete controllability
    rank_d, cond_d = controllability(Ad, Bd)
    print(f"\nControllability:")
    print(f"  Rank: {rank_d}/4 {'[OK]' if rank_d == 4 else '[FAIL]'}")
    print(f"  Condition number: {cond_d:.2e}")
    
    # =========================================================================
    # TEST 5: Sampling Rate Analysis
    # =========================================================================
    print("\n[TEST 5] Sampling Rate Adequacy")
    print("-" * 70)
    
    # Natural frequency from imaginary eigenvalues
    omega_n = np.max(np.abs(np.imag(eigvals)))
    f_n = omega_n / (2*np.pi)
    period = 1/f_n if f_n > 0 else np.inf
    
    print(f"Natural frequency: ω_n = {omega_n:.3f} rad/s ({f_n:.3f} Hz)")
    print(f"Period: T = {period:.3f} s")
    print(f"Sampling rate: f_s = {1/DEFAULT_TS:.0f} Hz")
    
    samples_per_period = period / DEFAULT_TS if period < np.inf else np.inf
    omega_n_Ts = omega_n * DEFAULT_TS
    
    print(f"\nSamples per period: {samples_per_period:.1f}")
    print(f"ω_n·Ts = {omega_n_Ts:.3f} rad/sample")
    
    # Guidelines
    print("\nGuidelines:")
    if samples_per_period > 20:
        print("  [OK] >20 samples/period: Continuous design acceptable")
    elif samples_per_period > 10:
        print("  [OK] >10 samples/period: Discrete design preferred")
    elif samples_per_period > 5:
        print("  WARNING  5-10 samples/period: Discrete design required")
    else:
        print("  [FAIL] <5 samples/period: Increase sampling rate!")
    
    if omega_n_Ts < 0.5:
        print("  [OK] ω_n·Ts < 0.5: Continuous approximation valid")
    else:
        print("  [FAIL] ω_n·Ts ≥ 0.5: Must use discrete design")
    
    # =========================================================================
    # TEST 6: ZOH vs Euler Comparison
    # =========================================================================
    print("\n[TEST 6] Discretization Method Comparison")
    print("-" * 70)
    
    # Euler discretization (for comparison)
    Ad_euler = np.eye(4) + A_analytical * DEFAULT_TS
    Bd_euler = B_analytical * DEFAULT_TS
    
    Ad_err = np.max(np.abs(Ad - Ad_euler))
    Bd_err = np.max(np.abs(Bd - Bd_euler))
    
    print(f"ZOH vs Euler:")
    print(f"  ||Ad - Ad_euler||∞ = {Ad_err:.2e}")
    print(f"  ||Bd - Bd_euler||∞ = {Bd_err:.2e}")
    print(f"  Relative error: {Ad_err/np.max(np.abs(Ad))*100:.2f}%")
    
    if Ad_err < 0.01:
        print("  WARNING  Small difference - Euler might be acceptable")
    else:
        print("  [OK] Significant difference - ZOH is essential")
    
    # =========================================================================
    # SUMMARY
    # =========================================================================
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    
    all_pass = (
        A_err < 1e-5 and B_err < 1e-5 and  # Analytical correct
        rank_c == 4 and rank_d == 4 and     # Controllable
        samples_per_period > 5               # Adequate sampling
    )
    
    if all_pass:
        print("[OK] ALL TESTS PASSED")
        print("\nRecommendations:")
        print("  - Use analytical linearization (verified correct)")
        print("  - Use discrete-time design (ZOH at 20 Hz)")
        print("  - System is fully controllable")
    else:
        print("[FAIL] SOME TESTS FAILED - Review results above")
    
    print("="*70 + "\n")