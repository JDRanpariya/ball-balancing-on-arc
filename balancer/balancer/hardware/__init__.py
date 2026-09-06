# balancer/hardware/__init__.py

"""Hardware interface modules for the ball-balancer."""

from .serial_config import SerialConfig, SERIAL_CONFIG
from .system_state import SystemState
from .sensor_utils import (
    sensor_to_theta,
    process_dual_sensors,
    VelocityEstimator,
    PositionSmoother,
    safe_float,
    parse_distance_line,
    parse_ipc_line,
)
from .action_utils import (
    # Enums
    DiscreteAction,
    # Constants
    DISCRETE_COMMANDS,
    DISCRETE_COMMANDS_BY_SIGN,
    # Low-level encoding
    encode_discrete_action,
    encode_continuous_action,
    continuous_to_discrete,
    continuous_to_discrete_int,
    get_action_command,
    # High-level sending
    send_action,
    # Cart movement
    stop_cart,
    move_cart_left,
    move_cart_right,
    move_cart_to_start_position,
    move_cart_to_position,
    reset_cart_position,
    # Neural network helpers
    action_to_onehot,
    onehot_to_action,
)
from .serial_reader import (
    create_ipc_reader,
    create_distance_reader,
    start_reader_threads,
)

__all__ = [
    # Config
    "SerialConfig",
    "SERIAL_CONFIG",
    # State
    "SystemState",
    # Sensors
    "sensor_to_theta",
    "process_dual_sensors",
    "VelocityEstimator",
    "PositionSmoother",
    "safe_float",
    "parse_distance_line",
    "parse_ipc_line",
    # Actions - enums & constants
    "DiscreteAction",
    "DISCRETE_COMMANDS",
    "DISCRETE_COMMANDS_BY_SIGN",
    # Actions - encoding
    "encode_discrete_action",
    "encode_continuous_action",
    "continuous_to_discrete",
    "continuous_to_discrete_int",
    "get_action_command",
    # Actions - sending
    "send_action",
    # Actions - cart movement
    "stop_cart",
    "move_cart_left",
    "move_cart_right",
    "move_cart_to_start_position",
    "move_cart_to_position",
    "reset_cart_position",
    # Actions - neural network
    "action_to_onehot",
    "onehot_to_action",
    # Readers
    "create_ipc_reader",
    "create_distance_reader",
    "start_reader_threads",
]