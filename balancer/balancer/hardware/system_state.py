# balancer/hardware/system_state.py

"""
Thread-safe state container for hardware sensor readings.

Provides a unified interface for accessing cart and ball state
from multiple reader threads.
"""

from __future__ import annotations
import time
import threading
from dataclasses import dataclass, field
from typing import List, Optional
import numpy as np


@dataclass
class SystemState:
    """
    Thread-safe container for all sensor readings.
    
    Attributes:
        cart_position: Cart position in meters (0 = track center)
        cart_velocity: Cart velocity in m/s
        ball_position: Ball angle in radians (0 = balanced)
        ball_velocity: Ball angular velocity in rad/s
        dip_reward: Binary reward signal from IR sensor
        
    Usage:
        state = SystemState()
        
        # From reader thread:
        with state.lock:
            state.cart_position = new_pos
            state.cart_velocity = new_vel
        state.data_event.set()  # Signal new data available
        
        # From controller thread:
        state.data_event.wait(timeout=0.1)
        vec = state.get_state_vector()
    """
    
    # -- State variables --
    cart_position: float = 0.0
    cart_velocity: float = 0.0
    ball_position: float = 0.0
    ball_velocity: float = 0.0
    dip_reward: int = 0
    raw_left_mm: int = 0
    raw_right_mm: int = 0
    
    # -- Timing --
    previous_ball_pos: float = 0.0
    previous_time: float = field(default_factory=time.time)
    # Monotonic timestamp of the last ball-state update (staleness watchdog)
    ball_updated_at: float = field(default_factory=time.monotonic)

    # -- Synchronization --
    lock: threading.Lock = field(default_factory=threading.Lock)
    data_event: threading.Event = field(default_factory=threading.Event)
    
    # -- Status --
    dist_status: bool = True
    
    def get_state_vector(self) -> List[float]:
        """Return [cart_pos, cart_vel, ball_pos, ball_vel] under lock."""
        with self.lock:
            return [
                self.cart_position,
                self.cart_velocity,
                self.ball_position,
                self.ball_velocity,
            ]
    
    def get_state_array(self) -> np.ndarray:
        """Return state as numpy array."""
        return np.array(self.get_state_vector(), dtype=np.float32)
    
    def set_cart_state(self, position: float, velocity: float) -> None:
        """Update cart state under lock."""
        with self.lock:
            self.cart_position = position
            self.cart_velocity = velocity
    
    def set_ball_state(
        self,
        position: float,
        velocity: float,
        dip_reward: int = 0,
        raw_left_mm: int = 0,
        raw_right_mm: int = 0,
    ) -> None:
        """Update ball state under lock and signal new data."""
        with self.lock:
            self.ball_position = position
            self.ball_velocity = velocity
            self.dip_reward = dip_reward
            self.raw_left_mm = raw_left_mm
            self.raw_right_mm = raw_right_mm
            self.ball_updated_at = time.monotonic()
        self.data_event.set()

    def ball_age(self) -> float:
        """Seconds since the last ball-state update (thread-safe)."""
        with self.lock:
            return time.monotonic() - self.ball_updated_at

    def is_stale(self, max_age: float) -> bool:
        """True if the ball sensor hasn't updated in over `max_age` seconds."""
        return self.ball_age() > max_age

    def reset(self) -> None:
        """Reset all state to defaults."""
        with self.lock:
            self.cart_position = 0.0
            self.cart_velocity = 0.0
            self.raw_left_mm = 0
            self.raw_right_mm = 0
            self.ball_position = 0.0
            self.ball_velocity = 0.0
            self.dip_reward = 0
            self.previous_ball_pos = 0.0
            self.previous_time = time.time()
            self.ball_updated_at = time.monotonic()
        self.data_event.clear()