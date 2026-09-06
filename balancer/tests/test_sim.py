"""Simulator smoke test: the cart-velocity axis limit is respected.

Run as a pytest test (`pytest balancer/tests`) or directly
(`python balancer/tests/test_sim.py`) for the step-by-step printout.
"""
from balancer.core.dynamics import NonLinearDynamics
import numpy as np

VMAX = 0.9  # m/s cart-velocity axis limit


def test_cart_velocity_respects_axis_limit():
    """Commanding 'right' (action=2) drives the cart velocity toward, but never
    past, the 0.9 m/s axis limit."""
    dyn = NonLinearDynamics(kinematics_integrator="rk4", tau=0.05)
    s = np.array([0.0, 0.0, 0.0, 0.0])
    for _ in range(5):
        s = dyn(s, action=2)
    cart_vel = s[1]
    assert 0.0 < abs(cart_vel) <= VMAX + 1e-6


if __name__ == "__main__":
    dyn = NonLinearDynamics(kinematics_integrator="rk4", tau=0.05)
    s = np.array([0.0, 0.0, 0.0, 0.0])
    for i in range(5):
        s = dyn(s, action=2)
        print(f"step {i+1}: cart_vel = {s[1]:.4f} m/s")  # should plateau at 0.9
