"""
Initial condition design for hardware experiments.

Principles:
  1. Fair: no IC should be physically impossible for ANY controller
  2. Discriminating: ICs span easy->hard to reveal performance differences
  3. Balanced: equal left/right to cancel asymmetries
  4. Reproducible: cart position is controlled by automation;
     ball position has natural variance from manual placement

The key insight from simulation: the failure mode is SAME-SIDE
(cart at wall + ball on same side). This is unrecoverable for PID
without wall override, but it also stresses MPC/LQR/RL differently.

We want ICs that:
  - Test regulation (ball near center, cart mid-track)
  - Test recovery (ball displaced, cart has room)
  - Test constraint handling (cart near limit, ball opposite = feasible)
  - Test worst-case (cart near limit, ball same side = hardest)

For fair comparison, the proportion of "worst-case" ICs should be
small enough that a controller can still demonstrate capability on
easier ICs, while large enough to reveal failure modes.
"""

import numpy as np
from balancer.hardware.constants import SYSTEM

CART_LIMIT = SYSTEM.CART_LIMIT  # 0.7765 m
BALL_LIMIT = SYSTEM.BALL_LIMIT  # 0.081 rad


def get_hardware_trial_configs(num_trials: int = 50) -> list:
    """
    Generate trial start configurations for hardware experiments.

    Returns list of dicts with:
        - start_position: cart target in meters (automated via move_cart)
        - expected_theta_sign: which side ball should be displaced to
          (positive=right, negative=left, 0=natural/center)
        - difficulty: 'easy', 'medium', 'hard', 'extreme'
        - description: human-readable label

    Distribution for 50 trials:
        - 10 easy (20%): cart center, ball ~center - tests regulation
        - 16 medium (32%): cart ±0.4m, ball natural - tests tracking
        - 14 hard (28%): cart ±0.65m, ball displaced - tests recovery
        - 10 extreme (20%): cart at limit, ball either side - tests limits

    Trials are interleaved by difficulty and balanced left/right.
    """
    configs = []

    # -- Easy: cart near center, ball at natural resting position --
    # Tests: basic regulation, dip dynamics
    # Cart at ±0.1m gives plenty of room in both directions
    n_easy = max(2, num_trials // 5)
    for i in range(n_easy):
        sign = 1 if i % 2 == 0 else -1
        configs.append({
            "start_position": sign * 0.10,
            "expected_theta_sign": 0,  # ball at natural rest
            "difficulty": "easy",
            "description": f"center_{'right' if sign > 0 else 'left'}",
        })

    # -- Medium: cart offset, ball at natural --
    # Tests: recovery with asymmetric rail budget
    # Cart at ±0.4m means ~0.37m room on one side, ~0.77m on other
    n_medium = max(4, int(num_trials * 0.32))
    for i in range(n_medium):
        sign = 1 if i % 2 == 0 else -1
        configs.append({
            "start_position": sign * 0.40,
            "expected_theta_sign": 0,
            "difficulty": "medium",
            "description": f"offset_{'right' if sign > 0 else 'left'}",
        })

    # -- Hard: cart well offset, ball has natural displacement --
    # Tests: recovery under asymmetric constraints
    # Cart at ±0.65m gives only 0.13m on one side
    # Ball will naturally roll to ~0.02-0.04 rad (random side)
    n_hard = max(4, int(num_trials * 0.28))
    for i in range(n_hard):
        sign = 1 if i % 2 == 0 else -1
        configs.append({
            "start_position": sign * 0.65,
            "expected_theta_sign": 0,  # natural - could be same or opposite
            "difficulty": "hard",
            "description": f"near_wall_{'right' if sign > 0 else 'left'}",
        })

    # -- Extreme: cart at limit --
    # Split 50/50: half where ball is on OPPOSITE side (feasible for all),
    # half where ball is on SAME side (tests wall handling)
    n_extreme = num_trials - n_easy - n_medium - n_hard
    n_opposite = n_extreme // 2
    n_same = n_extreme - n_opposite

    # Opposite side (feasible for all controllers)
    for i in range(n_opposite):
        cart_sign = 1 if i % 2 == 0 else -1
        configs.append({
            "start_position": cart_sign * CART_LIMIT,
            "expected_theta_sign": -cart_sign,  # ball opposite to cart
            "difficulty": "extreme_opposite",
            "description": f"wall_{'right' if cart_sign > 0 else 'left'}_ball_opposite",
        })

    # Same side (hardest - reveals structural limitations)
    for i in range(n_same):
        cart_sign = 1 if i % 2 == 0 else -1
        configs.append({
            "start_position": cart_sign * CART_LIMIT,
            "expected_theta_sign": cart_sign,  # ball same side as cart
            "difficulty": "extreme_same",
            "description": f"wall_{'right' if cart_sign > 0 else 'left'}_ball_same",
        })

    # Shuffle deterministically for reproducibility but interleave difficulty
    rng = np.random.default_rng(seed=42)
    rng.shuffle(configs)

    return configs[:num_trials]


def get_stratified_trial_configs(num_trials: int = 50) -> list:
    """
    Alternative: pure stratified design where each trial has
    a specific cart position drawn from a grid.

    This gives more uniform coverage of the state space and
    makes statistical analysis cleaner (every controller sees
    identical cart positions).

    Cart positions: 10 evenly spaced from -0.7 to +0.7
    Each repeated num_trials/10 times.
    Ball: natural placement (uncontrolled).
    """
    positions = np.linspace(-0.70, 0.70, 10)
    repeats = max(1, num_trials // len(positions))
    remainder = num_trials - repeats * len(positions)

    configs = []
    for pos in positions:
        for r in range(repeats):
            configs.append({
                "start_position": float(pos),
                "expected_theta_sign": 0,
                "difficulty": _classify_difficulty(pos),
                "description": f"grid_{pos:+.2f}_rep{r}",
            })

    # Fill remainder from extremes (most informative)
    if remainder > 0:
        extras = [CART_LIMIT, -CART_LIMIT] * (remainder // 2 + 1)
        for pos in extras[:remainder]:
            configs.append({
                "start_position": float(pos),
                "expected_theta_sign": 0,
                "difficulty": "extreme",
                "description": f"grid_{pos:+.4f}_extra",
            })

    rng = np.random.default_rng(seed=42)
    rng.shuffle(configs)
    return configs[:num_trials]


def _classify_difficulty(cart_pos: float) -> str:
    ax = abs(cart_pos)
    if ax < 0.25:
        return "easy"
    elif ax < 0.50:
        return "medium"
    elif ax < 0.70:
        return "hard"
    return "extreme"


# ----------------------------------------------------------------------
# Summary for paper methodology section
# ----------------------------------------------------------------------

def print_design_summary(num_trials: int = 50):
    """Print IC design summary for paper methods section."""
    configs = get_hardware_trial_configs(num_trials)
    from collections import Counter
    diff_counts = Counter(c["difficulty"] for c in configs)
    positions = [c["start_position"] for c in configs]

    print(f"IC Design Summary ({num_trials} trials):")
    print(f"  Difficulty distribution:")
    for diff, count in sorted(diff_counts.items()):
        print(f"    {diff:20s}: {count:3d} ({100*count/num_trials:.0f}%)")
    print(f"  Cart position range: [{min(positions):.3f}, {max(positions):.3f}] m")
    print(f"  Left/right balance: {sum(1 for p in positions if p < 0)}/"
          f"{sum(1 for p in positions if p > 0)}")


if __name__ == "__main__":
    print("=" * 60)
    print("DESIGN A: Difficulty-stratified")
    print("=" * 60)
    print_design_summary(50)

    print()
    print("=" * 60)
    print("DESIGN B: Grid-based (uniform coverage)")
    print("=" * 60)
    configs = get_stratified_trial_configs(50)
    from collections import Counter
    diff_counts = Counter(c["difficulty"] for c in configs)
    print(f"  Difficulty distribution:")
    for diff, count in sorted(diff_counts.items()):
        print(f"    {diff:20s}: {count:3d} ({100*count/50:.0f}%)")
