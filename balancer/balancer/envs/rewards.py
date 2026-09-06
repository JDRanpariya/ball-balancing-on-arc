import numpy as np
from balancer.core.state import BALL_X, BALL_DOT, CART_X, CART_DOT

# =============================================================================

#  SIMPLE REWARDS (no action context needed)

# =============================================================================


def regular(state, pos_limits):
    return 1.0


def dip_reward(state, pos_limits):
    distance = np.abs(state[BALL_X])
    if distance <= 0.007:
        return 1
    return 0


def dip_reward_neg(state, pos_limits):
    distance = np.abs(state[BALL_X])
    if distance <= 0.007:
        return 1
    return -1


def ball_gaussian_distance(state, pos_limits, spread_factor=16, vel_scale=0.2):
    distance = np.abs(state[BALL_X])
    velocity = np.abs(state[BALL_DOT])
    distance = min(distance, pos_limits[1])
    sigma = pos_limits[1] / spread_factor
    pos_reward = np.exp(-(distance ** 2) / (2 * sigma ** 2))
    vel_penalty = np.exp(-velocity / vel_scale)
    reward = pos_reward * vel_penalty
    reward = np.clip(reward, 0.0, 1.0)
    return float(reward)


def gaussian_wall(state, pos_limits, action=0.0, prev_action=0.0,
                  spread_factor=16, vel_scale=0.2, w_wall=0.35):
    """Gaussian distance reward + cart wall penalty (same as balanced)."""
    # Base gaussian reward
    base = ball_gaussian_distance(state, pos_limits, spread_factor, vel_scale)

    # Cart wall penalty (mirrors balanced_reward logic)
    cart_pos = state[CART_X]
    cart_limit = pos_limits[0]
    cart_ratio = abs(cart_pos) / cart_limit
    if cart_ratio > 0.85:
        into_wall = (cart_pos > 0 and action > 0) or (
            cart_pos < 0 and action < 0)
        if into_wall:
            wall_pen = w_wall * ((cart_ratio - 0.85) / 0.15) ** 2
        else:
            wall_pen = 0.0
    else:
        wall_pen = 0.0

    # Ball boundary penalty
    ball_ratio = abs(state[BALL_X]) / pos_limits[1]
    if ball_ratio > 0.85:
        ball_pen = w_wall * ((ball_ratio - 0.85) / 0.15) ** 2
    else:
        ball_pen = 0.0

    return float(base - wall_pen - ball_pen)


# =============================================================================

#  BALANCED REWARD (needs action context - called from base.py with action args)

# =============================================================================

def balanced_reward(state, pos_limits, action=0.0, prev_action=0.0,
                    # -- shape (tuned for 2.5cm dip) --
                    sigma_outer=None,
                    sigma_inner=None,
                    sigma_vel_gate=None,
                    # -- weights --
                    w_pos=0.60,
                    w_vel=0.25,
                    w_effort=0.05,
                    w_jerk=0.05,
                    w_boundary=0.35):

    theta = state[BALL_X]
    theta_dot = state[BALL_DOT]
    theta_abs = abs(theta)
    ball_limit = pos_limits[1]

    # Better defaults for precision balancing
    if sigma_outer is None:
        sigma_outer = ball_limit / 4.0      # 0.019 rad - broad guidance
    if sigma_inner is None:
        sigma_inner = ball_limit / 25.0     # 0.003 rad - MUCH sharper
    if sigma_vel_gate is None:
        sigma_vel_gate = ball_limit / 10.0  # 0.0077 rad - tighter gate

    # -- 1. Position: Dual Gaussian with actual separation --
    outer = np.exp(-theta**2 / (2 * sigma_outer**2))
    inner = np.exp(-theta**2 / (2 * sigma_inner**2))
    pos_reward = 0.6 * outer + 0.4 * inner  # More weight on precision

    # -- 2. Velocity: Normalized and gated --
    vel_gate = np.exp(-theta**2 / (2 * sigma_vel_gate**2))
    max_vel = 2.0  # Expected max velocity (rad/s)
    vel_normalized = min(theta_dot**2 / max_vel**2, 1.0)  # Clamp to [0,1]
    vel_pen = vel_gate * vel_normalized

    # -- 3. Ball boundary: Soft ramp instead of hard step --
    boundary_ratio = theta_abs / ball_limit
    if boundary_ratio > 0.85:
        boundary_pen = w_boundary * ((boundary_ratio - 0.85) / 0.15)**2
    else:
        boundary_pen = 0.0

    # -- 3b. Cart wall penalty: penalize pushing into wall -
    # When cart is near wall AND action pushes further into it,
    # apply penalty. This teaches the policy to avoid wall-stuck states.
    cart_pos = state[CART_X]
    cart_limit = pos_limits[0]
    cart_ratio = abs(cart_pos) / cart_limit
    if cart_ratio > 0.85:
        # Penalty for being near wall AND pushing into it
        into_wall = (cart_pos > 0 and action > 0) or (
            cart_pos < 0 and action < 0)
        if into_wall:
            cart_wall_pen = 0.2 * ((cart_ratio - 0.85) / 0.15)**2
        else:
            cart_wall_pen = 0.0
    else:
        cart_wall_pen = 0.0

    # -- 4. Effort/Jerk: Normalized --
    effort_pen = min(action**2, 1.0)
    jerk_pen = min((action - prev_action)**2, 1.0)

    reward = (
        w_pos * pos_reward
        - w_vel * vel_pen
        - w_effort * effort_pen
        - w_jerk * jerk_pen
        - boundary_pen
        - cart_wall_pen
    )

    return float(np.clip(reward, -1.0, 1.0))


# =============================================================================

#  REGISTRY

#

#  Rewards that need action context MUST have signature:

#      fn(state, pos_limits, action=float, prev_action=float)

#  and their key must be listed in base.py._reward_needs_action set.

#

#  Simple rewards have signature:

#      fn(state, pos_limits)

# =============================================================================
REWARDS = {
    # -- Simple (no action context) ------------------------------------
    "regular":                regular,
    "dip_reward":             dip_reward,
    "dip_reward_neg":         dip_reward_neg,
    "ball_gaussian_distance": ball_gaussian_distance,
    "gaussian_wall": gaussian_wall,

    # -- Balanced (needs action context) -------------------------------
    # Default: matches classical tuning objective exactly
    "balanced": balanced_reward,

    # Ablations: verify each component matters
    "balanced_no_effort": lambda s, p, action=0.0, prev_action=0.0:
        balanced_reward(s, p, action, prev_action,
                        w_effort=0.0, w_jerk=0.0),

    "balanced_no_settle": lambda s, p, action=0.0, prev_action=0.0:
        balanced_reward(s, p, action, prev_action),

    # Settling band width variants
    "balanced_narrow": lambda s, p, action=0.0, prev_action=0.0:
        balanced_reward(s, p, action, prev_action,
                        sigma_inner=0.003),  # ±0.5cm - tight

    "balanced_wide": lambda s, p, action=0.0, prev_action=0.0:
        balanced_reward(s, p, action, prev_action,
                        sigma_inner=0.012),   # ±2cm - full dip width
}

# -- Keys that require action context (used by base.py) ----------------

REWARDS_NEED_ACTION = {
    "balanced", "balanced_no_effort", "balanced_no_settle",
    "balanced_narrow", "balanced_wide", "gaussian_wall",
}
