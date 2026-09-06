from dataclasses import dataclass, field, asdict
from typing import List, Optional
import time
import os

from balancer.hardware.constants import SYSTEM


@dataclass
class ExperimentConfig:
    """Single source of truth for one experiment run."""

    # Identity
    name:        str = "experiment"
    description: str = ""
    env_label:   str = "nominal_ball"

    # Control
    action_type:  str   = "discrete"
    Ts:           float = 0.05          # control period (s) -> 20 Hz
    run_time:     float = SYSTEM.FAIL_TIME         # max trial duration (s)
    num_trials:   int   = 50

    # Noise
    noise_sigma:  float = 0.0           # 0 = no noise, 0.1 = 10% of range

    # Motor model for sim evaluation
    motor_model:  str   = "first_order"  # "first_order" or "s_curve"

    # Settling criterion - pull from SYSTEM so tuning/eval/analysis agree
    settling_band:     float = SYSTEM.SETTLING_BAND       # was 0.05
    settling_duration: float = SYSTEM.SETTLING_DURATION    # was 2.0
    velocity_band:     float = SYSTEM.VELOCITY_BAND        # diagnostic only

    # Output
    results_dir:  str  = "results"
    resume:       bool = True           # resume from checkpoint if exists

    # Auto-generated
    timestamp:    str  = field(default_factory=lambda: time.strftime('%Y%m%d_%H%M%S'))

    @property
    def output_dir(self) -> str:
        """Auto-generate organised output path."""
        return os.path.join(
            self.results_dir,
            self.env_label,
            f"{self.name}_{self.timestamp}",
        )

    @property
    def checkpoint_path(self) -> str:
        return os.path.join(self.output_dir, "data.json")

    @property
    def config_path(self) -> str:
        return os.path.join(self.output_dir, "config.yaml")

    def to_dict(self) -> dict:
        return asdict(self)
