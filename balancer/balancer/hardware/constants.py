# balancer/balancer/core/constants.py

"""
Physical system constants - SINGLE SOURCE OF TRUTH.

All physical parameters, track geometry, and sensor calibration values
should be defined here and imported elsewhere.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class SystemConstants:
    """Immutable physical system parameters."""

    # -- Physical parameters --
    ARC_RADIUS_M: float = 2.101
    CART_MASS_KG: float = 0.351
    BALL_MASS_KG: float = 0.024
    BALL_RADIUS_M: float = 0.009
    MAX_CART_VEL: float = 0.9  # in meters
    GRAVITY: float = 9.81

    # -- Track geometry --
    TRACK_START_M: float = 0.200
    TRACK_END_M: float = 1.753

    # -- Sensor calibration --
    # Hardware values (match physical sensor board mounting)
    LEFT_CENTER_MM: int = 192
    RIGHT_CENTER_MM: int = 161
    CENTER_THRESHOLD_RAD: float = 0.001

    # Calibration history in balancer/hardware/calibration_log.json.
    # center_mm values are updated on each recalibration; SENSOR_OFFSET_RAD
    # is kept at 0.0 - any residual offset is absorbed into center_mm directly.
    SENSOR_OFFSET_RAD: float = 0.0

    # -- Environment limits --
    CART_LIMIT: float = 0.7765          # operational limit
    BALL_LIMIT: float = 0.0810         # edge of arc

    # Eval Parameters
    SETTLING_BAND: float = 0.01       # rad - ball angle tolerance
    SETTLING_DURATION: float = 1.0     # s - must stay in band this long
    VELOCITY_BAND: float = 0.05        # rad/s - NOT used in settling check
    FAIL_TIME: float = 30.0            # s - max trial duration / unsettled value
    # Deprecated
    DISCRETE_THRESHOLD: float = 0.05   # threshold for cont->discrete conversion

    # -- Derived properties --

    @property
    def track_center(self) -> float:
        return (self.TRACK_START_M + self.TRACK_END_M) / 2.0

    @property
    def track_half(self) -> float:
        return (self.TRACK_END_M - self.TRACK_START_M) / 2.0


# Single instance - import this

SYSTEM = SystemConstants()

# Legacy aliases (for gradual migration - remove once all code updated)

CENTER_THRESHOLD_RAD = SYSTEM.CENTER_THRESHOLD_RAD
LEFT_CENTER_MM = SYSTEM.LEFT_CENTER_MM
RIGHT_CENTER_MM = SYSTEM.RIGHT_CENTER_MM
ARC_RADIUS_M = SYSTEM.ARC_RADIUS_M