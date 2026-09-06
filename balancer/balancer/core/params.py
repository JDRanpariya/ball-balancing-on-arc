# balancer/core/params.py

import numpy as np

# ============================================================================
# SYSTEM PARAMETERS (best hardware measurements)
# ============================================================================

DEFAULT_PARAMS = dict(
    M=0.351,
    m=0.024,
    R=2.101,
    r=0.009,
    d=0.002,
    sigma=0.020 / 2.354820045,  # FWHM=20mm actual
    g=9.81,
    tau_v=0.15,
    friction_coeff_cart=0.08,
    mu_ball_rolling=0.025,
    mu_ball_viscous=0.009,
)

# ============================================================================
# DOMAIN RANDOMIZATION - ranges centred on measured values
# ============================================================================

DR_RANGES: dict[str, tuple[float, float]] = {
    "gravity":             (9.76,   9.86),     # 9.81  ± 0.05
    "arc_radius":          (2.001,  2.201),    # 2.101 ± 0.10
    "friction_coeff_cart": (0.035,  0.125),    # 0.08  ± 0.045
    "mu_ball_rolling":     (0.010,  0.040),    # 0.025 ± 0.015
    "mu_ball_viscous":     (0.003,  0.015),    # 0.009 ± 0.006
    "tau_v":               (0.10,   0.20),     # 0.15  ± 0.05
    "dip_depth":           (0.001,  0.003),    # 0.002 ± 0.001
    "dip_fwhm":            (0.012,  0.028),    # 0.020 ± 0.008
}

DR_MEANS: dict[str, float] = {
    k: (lo + hi) / 2.0 for k, (lo, hi) in DR_RANGES.items()
}

# ============================================================================
# OBSERVATION NOISE - (low, high) of std-dev per state component
# ============================================================================

OBS_NOISE_RANGES: list[tuple[float, float]] = [
    (0.0002, 0.001),    # cart pos  (encoder)
    (0.005,  0.020),    # cart vel  (filtered)
    (0.0005, 0.002),    # ball pos  (ToF)
    (0.02,   0.04),     # ball vel  (derivative of noisy ToF)
]

NOMINAL_OBS_NOISE: np.ndarray = np.array(
    [(lo + hi) / 2.0 for lo, hi in OBS_NOISE_RANGES],
    dtype=np.float32,
)
# -> [0.0006, 0.015, 0.00125, 0.05]
