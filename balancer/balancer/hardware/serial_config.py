"""
Serial port and protocol configuration.

Single source of truth for all serial communication parameters.
"""

from dataclasses import dataclass, field
from typing import Optional
import os


@dataclass(frozen=True)
class SerialConfig:
    """Immutable serial port configuration."""
    
    # -- Ports --
    ipc_port: str = "/dev/ttyUSB0"
    distance_port: str = "/dev/distance_sensor"
    
    # -- Baud rates --
    baudrate: int = 115200
    
    # -- Timeouts --
    ipc_timeout: float = 0.01
    distance_timeout: float = 0.05
    
    # -- Protocol --
    frame_start: bytes = b'\x02'
    frame_end: bytes = b'\x03'
    
    # -- Recovery --
    watchdog_timeout: float = 5.0
    reset_cooldown: float = 10.0
    timeout_streak_limit: int = 20
    
    @classmethod
    def from_env(cls) -> "SerialConfig":
        """Create config with environment variable overrides."""
        return cls(
            ipc_port=os.getenv("BALANCER_IPC_PORT", cls.ipc_port),
            distance_port=os.getenv("BALANCER_DIST_PORT", cls.distance_port),
        )


# Default instance

SERIAL_CONFIG = SerialConfig()