"""
Sensor processing utilities.

Converts raw sensor readings to model coordinates.
All sensor calibration values come from core.constants.SYSTEM.
"""

from __future__ import annotations
import re
import numpy as np
from typing import Tuple, Optional
from collections import deque

from balancer.hardware.constants import SYSTEM


# ============================================================================

# RAW SENSOR -> MODEL COORDINATES

# ============================================================================

def sensor_to_theta(
    sensor_reading_mm: float,
    center_mm: float,
    arc_radius: float = SYSTEM.ARC_RADIUS_M,
) -> float:
    """
    Convert ToF sensor reading to ball angle (radians).
    
    Args:
        sensor_reading_mm: Raw sensor reading in millimeters
        center_mm: Calibrated center offset for this sensor
        arc_radius: Arc radius in meters
        
    Returns:
        Ball angle in radians (positive = ball to the left of center)
        
    Note:
        Uses small-angle approximation: θ ≈ arc_length / radius
    """
    distance_m = (sensor_reading_mm - center_mm) / 1000.0
    return distance_m / arc_radius


def process_dual_sensors(
    left_mm: int,
    right_mm: int,
    left_center: float = SYSTEM.LEFT_CENTER_MM,
    right_center: float = SYSTEM.RIGHT_CENTER_MM,
    center_threshold: float = SYSTEM.CENTER_THRESHOLD_RAD,
) -> float:
    """
    Process dual ToF sensor readings into unified ball angle.
    
    Uses left sensor when ball is on left side, right sensor when on right.
    Returns 0 when ball is within center threshold.
    
    Args:
        left_mm: Left sensor reading (mm)
        right_mm: Right sensor reading (mm)
        left_center: Left sensor calibration offset
        right_center: Right sensor calibration offset
        center_threshold: Angle threshold for "at center" (rad)
        
    Returns:
        Ball angle in radians
    """
    if left_mm <= left_center:
        theta = sensor_to_theta(left_mm, center_mm=left_center)
    elif right_mm <= right_center:
        theta = -sensor_to_theta(right_mm, center_mm=right_center)
    else:
        theta = 0.0
    
    # Apply center deadband
    if abs(theta) < center_threshold:
        theta = 0.0
        
    return theta


# ============================================================================

# VELOCITY ESTIMATION

# ============================================================================

class VelocityEstimator:
    """
    Estimates velocity from position measurements with smoothing.
    
    Uses finite differences with configurable smoothing buffer.
    Clips extreme values to handle sensor glitches.
    
    Args:
        buffer_len: Number of samples for moving average
        clip_value: Maximum absolute velocity (rad/s or m/s)
        min_dt: Minimum time delta to prevent division by zero
    """
    
    def __init__(
        self,
        buffer_len: int = 3,
        clip_value: float = 1.5,
        min_dt: float = 0.02,
    ):
        self.buffer = deque(maxlen=buffer_len)
        self.clip_value = clip_value
        self.min_dt = min_dt
        self._prev_position: float = 0.0
        self._prev_time: float = 0.0
        self._initialized: bool = False
    
    def update(self, position: float, timestamp: float) -> float:
        """
        Update with new position measurement and return smoothed velocity.
        
        Args:
            position: Current position measurement
            timestamp: Current time (seconds, monotonic preferred)
            
        Returns:
            Smoothed velocity estimate
        """
        if not self._initialized:
            # First sample: store reference but don't compute velocity
            self._prev_position = position
            self._prev_time = timestamp
            self._initialized = True
            return 0.0                            

        dt = timestamp - self._prev_time
        
        if dt > self.min_dt:
            raw_vel = (position - self._prev_position) / dt
            clipped_vel = float(np.clip(raw_vel, -self.clip_value, self.clip_value))
            self.buffer.append(clipped_vel)
        
        self._prev_position = position
        self._prev_time = timestamp
        
        if len(self.buffer) == 0:
            return 0.0
        return float(np.clip(np.mean(self.buffer), -self.clip_value, self.clip_value))
    
    def reset(self):
        """Clear buffer and reset state."""
        self.buffer.clear()
        self._prev_position = 0.0
        self._prev_time = 0.0
        self._initialized = False                 


# ============================================================================

# POSITION SMOOTHER

# ============================================================================

class PositionSmoother:
    """Simple moving average smoother for position readings."""
    
    def __init__(self, buffer_len: int = 1):
        self.buffer = deque(maxlen=buffer_len)
    
    def update(self, position: float) -> float:
        self.buffer.append(position)
        return float(np.mean(self.buffer))
    
    def reset(self):
        self.buffer.clear()


# ============================================================================

# PARSING UTILITIES

# ============================================================================

def safe_float(s: str) -> float:
    """Parse float from potentially malformed string."""
    s_clean = re.sub(r'[^\d\.\-eE+]', '', s)
    try:
        return float(s_clean)
    except ValueError:
        return 0.0


def parse_distance_line(line: str) -> Optional[Tuple[int, int, int]]:
    """
    Parse a distance sensor output line.
    
    Expected format: "left_mm right_mm binary_reward"
    
    Returns:
        Tuple of (left_mm, right_mm, reward) or None if invalid
    """
    try:
        parts = line.strip().split()
        if len(parts) != 3:
            return None
        return int(parts[0]), int(parts[1]), int(parts[2])
    except (ValueError, IndexError):
        return None


def parse_ipc_line(line: str) -> Optional[Tuple[float, float]]:
    """
    Parse an IPC (cart) serial line.
    
    Expected format: "position_mm,velocity_mm"
    
    Returns:
        Tuple of (position_m, velocity_m/s) or None if invalid
    """
    try:
        cleaned = line.replace('\x02', '').replace('\x03', '').strip()
        pos_mm, vel_mm = cleaned.split(',')
        pos_m = safe_float(pos_mm) / 1000.0
        vel_mps = safe_float(vel_mm) / 1000.0
        return pos_m, vel_mps
    except (ValueError, AttributeError):
        return None