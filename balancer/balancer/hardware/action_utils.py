"""
Action encoding/decoding and cart control utilities.

Standardizes the mapping between controller outputs and serial commands.
"""

from __future__ import annotations
from enum import IntEnum
from typing import Union, TYPE_CHECKING
import time
import numpy as np

if TYPE_CHECKING:
    import serial
    from .system_state import SystemState


class DiscreteAction(IntEnum):
    """Discrete action values used by controllers."""
    LEFT  = 0
    STOP  = 1
    RIGHT = 2


# -- Serial command lookup tables --------------------------------------------

# Frame format: \x02 <command> \x03
DISCRETE_VELOCITIES_MMS = [-900.0, 0.0, 900.0]

DISCRETE_COMMANDS = {
    DiscreteAction.LEFT:  b'\x02 left \x03',
    DiscreteAction.STOP:  b'\x02 st \x03',
    DiscreteAction.RIGHT: b'\x02 right \x03',
}

DISCRETE_COMMANDS_BY_SIGN = {
    -1: DISCRETE_COMMANDS[DiscreteAction.LEFT],
     0: DISCRETE_COMMANDS[DiscreteAction.STOP],
     1: DISCRETE_COMMANDS[DiscreteAction.RIGHT],
}


# -- Low-level encoding -------------------------------------------------------
def encode_torque_command(torque: float) -> bytes:
    """
    Convert a raw torque value (N·m or N) to a serial command.

    The PLC expects a framed string like: \x02 <torque_value> \x03

    Args:
        torque: Torque setpoint (positive = push right, negative = pull left)

    Returns:
        Framed serial command bytes.
    """
    return f"\x02 {float(torque):.3f} \x03".encode()


def encode_normalised_torque(
    action: float,
    max_torque: float = 2.0,
) -> bytes:
    """
    Convert a normalised action in [-1, 1] to a serial torque command.

    Args:
        action:     Normalised action in [-1, 1].
        max_torque: Maximum absolute torque value corresponding to |action| == 1.

    Returns:
        Framed serial command bytes.
    """
    scaled = float(action) * max_torque
    return encode_torque_command(scaled)

def encode_discrete_action(action: Union[int, DiscreteAction]) -> bytes:
    """
    Convert discrete action to serial command bytes.

    Args:
        action: 0=left, 1=stop, 2=right (or DiscreteAction enum)

    Returns:
        Framed serial command bytes.
    """
    if isinstance(action, DiscreteAction):
        return DISCRETE_COMMANDS[action]
    return DISCRETE_COMMANDS[DiscreteAction(action)]


def encode_continuous_action(
    action: float,
    max_value: float = 900.0,
) -> bytes:
    """
    Convert a velocity setpoint in **m/s** to a serial velocity command.

    Same semantics as the simulator (``NonLinearDynamics._resolve_vcmd``):
    the action is the commanded cart velocity in m/s, saturated at the
    axis limit.  Converted to mm/s for the PLC.

    Args:
        action:    Velocity setpoint in m/s (0.5 -> 500 mm/s).
        max_value: Axis velocity limit in mm/s (saturation bound).

    Returns:
        Framed serial command bytes.

    Examples:
        encode_continuous_action( 0.5, 900)  ->  b'\x02  500.000 \x03'
        encode_continuous_action( 1.0, 900)  ->  b'\x02  900.000 \x03'  (saturated)
        encode_continuous_action(-0.9, 900)  ->  b'\x02 -900.000 \x03'
    """
    scaled = float(np.clip(action * 1000.0, -max_value, max_value))
    return f"\x02 {scaled:.3f} \x03".encode()


def encode_velocity_command(velocity: float) -> bytes:
    """
    Convert a **raw** velocity value (mm/s or axis units) to a serial command.

    Unlike :func:`encode_continuous_action`, no scaling is applied.
    Use this when you already have a physical velocity.

    Args:
        velocity: Raw velocity, e.g. 900.0 (mm/s).  Negative = left.

    Returns:
        Framed serial command bytes.
    """
    return f"\x02 {float(velocity):.3f} \x03".encode()


def continuous_to_discrete(
    u: float,
    threshold: float = 0.05,
) -> DiscreteAction:
    """
    Convert continuous control signal to discrete action.

    Args:
        u:         Continuous control in [-1, 1].
        threshold: Deadband threshold.

    Returns:
        DiscreteAction enum value.
    """
    if u > threshold:
        return DiscreteAction.RIGHT
    elif u < -threshold:
        return DiscreteAction.LEFT
    return DiscreteAction.STOP


def continuous_to_discrete_int(
    u: float,
    threshold: float = 0.05,
) -> int:
    """
    Convert continuous control signal to discrete action as int.

    Returns:
        -1 (left), 0 (stop), or 1 (right).
    """
    if u > threshold:
        return 1
    elif u < -threshold:
        return -1
    return 0


# -- High-level sending -------------------------------------------------------

def send_action(
    ser: "serial.Serial",
    u: float,
    action_type: str = "discrete",
    threshold: float = 0.05,
    max_value: float = 900.0,
) -> None:
    """
    Send control action to cart via serial.

    For discrete mode, thresholds continuous signal to {left, stop, right}.
    For continuous mode, scales and sends raw value.

    Args:
        ser:         Serial port object.
        u:           Control value: velocity setpoint in m/s for ``"cont"``
                     (same semantics as the simulator), sign for ``"discrete"``.
        action_type: ``"discrete"`` or ``"cont"``.
        threshold:   Deadband for discrete thresholding.
        max_value:   Axis velocity limit in mm/s for continuous mode.
    """
    if action_type == "discrete":
        action = continuous_to_discrete(u, threshold)
        ser.write(encode_discrete_action(action))
    elif action_type == "cont":
        ser.write(encode_continuous_action(u, max_value))
    else:
        raise ValueError(f"Unknown action_type: {action_type!r}")


def get_action_command(
    action: Union[int, float],
    action_type: str,
    threshold: float = 0.05,
    max_value: float = 900.0,
) -> bytes:
    """
    Universal action -> serial command converter (returns bytes, does not send).

    Args:
        action:      Action value (discrete int, or velocity setpoint in m/s
                     for continuous - same semantics as the simulator).
        action_type: ``"discrete"`` or ``"cont"``.
        threshold:   Deadband for continuous -> discrete conversion.
        max_value:   Axis velocity limit in mm/s for continuous mode.

    Returns:
        Serial command bytes.
    """
    if action_type == "discrete":
        if isinstance(action, (float, np.floating)):
            action = continuous_to_discrete(action, threshold)
        return encode_discrete_action(int(action))
    elif action_type == "cont":
        return encode_continuous_action(float(action), max_value)
    else:
        raise ValueError(f"Unknown action_type: {action_type!r}")


# -- Cart movement helpers ----------------------------------------------------

def stop_cart(
    ser: "serial.Serial",
    action_type: str = "discrete",
) -> None:
    """Send stop command to cart."""
    if action_type == "discrete":
        ser.write(encode_discrete_action(DiscreteAction.STOP))
    elif action_type == "cont":
        # Velocity = 0  ->  PLC decelerates to halt
        ser.write(encode_velocity_command(0.0))
    else:
        raise ValueError(f"Unknown action_type: {action_type!r}")


def move_cart_left(
    ser: "serial.Serial",
    duration: float = 5.0,
    action_type: str = "discrete",
    velocity: float = 900.0,
) -> None:
    """
    Move cart to the left for *duration* seconds, then return.

    Note: the caller is responsible for sending a stop command afterwards
    when using ``action_type="cont"`` if continuous motion is not desired.

    Args:
        ser:         Serial port.
        duration:    Time to move in seconds.
        action_type: ``"discrete"`` -> jog command,
                     ``"cont"``     -> raw velocity command (negative = left).
        velocity:    Raw velocity magnitude in axis units (mm/s).
                     Only used for ``"cont"`` mode.
    """
    if action_type == "discrete":
        ser.write(encode_discrete_action(DiscreteAction.LEFT))
    elif action_type == "cont":
        # encode_velocity_command takes the unscaled axis velocity;
        # negative sign -> left per PLC convention.
        ser.write(encode_velocity_command(-abs(velocity)))
    else:
        raise ValueError(f"Unknown action_type: {action_type!r}")

    time.sleep(duration)


def move_cart_right(
    ser: "serial.Serial",
    duration: float = 5.0,
    action_type: str = "discrete",
    velocity: float = 900.0,
) -> None:
    """
    Move cart to the right for *duration* seconds, then return.

    Args:
        ser:         Serial port.
        duration:    Time to move in seconds.
        action_type: ``"discrete"`` -> jog command,
                     ``"cont"``     -> raw velocity command (positive = right).
        velocity:    Raw velocity magnitude in axis units (mm/s).
                     Only used for ``"cont"`` mode.
    """
    if action_type == "discrete":
        ser.write(encode_discrete_action(DiscreteAction.RIGHT))
    elif action_type == "cont":
        # encode_velocity_command takes the unscaled axis velocity;
        # positive sign -> right per PLC convention.
        ser.write(encode_velocity_command(+abs(velocity)))
    else:
        raise ValueError(f"Unknown action_type: {action_type!r}")

    time.sleep(duration)


def move_cart_to_start_position(
    ser: "serial.Serial",
    state: "SystemState",
    direction: str = "left",
    action_type: str = "discrete",   # BUG FIX: was "disc", now consistent
    velocity: float = 900.0,
    duration: float = 3.0,
) -> None:
    """
    Drive cart to start position then stop.

    Works for both discrete (jog) and continuous (velocity) action types.

    Args:
        ser:         Serial port.
        state:       Current system state (unused here, available for subclasses).
        direction:   ``"left"`` or ``"right"``.
        action_type: ``"discrete"`` or ``"cont"``.
        velocity:    Raw velocity in axis units (``"cont"`` mode only).
        duration:    How long to drive before stopping (seconds).
    """
    if direction == "left":
        move_cart_left(ser, duration=duration,
                       action_type=action_type, velocity=velocity)
    elif direction == "right":
        move_cart_right(ser, duration=duration,
                        action_type=action_type, velocity=velocity)
    else:
        raise ValueError(f"Unknown direction: {direction!r}")

    stop_cart(ser, action_type=action_type)


def reset_cart_position(
    ser: "serial.Serial",
    state: "SystemState",
    target_side: str = "left",
    action_type: str = "discrete",   # BUG FIX: was missing, now forwarded
    velocity: float = 900.0,
    duration: float = 3.0,
) -> None:
    """
    Reset cart to a neutral position.

    Args:
        ser:         Serial port.
        state:       Current system state.
        target_side: ``"left"`` or ``"right"``.
        action_type: ``"discrete"`` or ``"cont"``.
        velocity:    Raw velocity in axis units (``"cont"`` mode only).
        duration:    How long to drive before stopping (seconds).
    """
    move_cart_to_start_position(
        ser, state,
        direction=target_side,
        action_type=action_type,   # BUG FIX: was not forwarded
        velocity=velocity,
        duration=duration,
    )


# -- Position control ---------------------------------------------------------

def move_cart_to_position(
    ser: "serial.Serial",
    state: "SystemState",
    target_m: float,
    tolerance_m: float = 0.010,
    max_vel_mms: float = 500.0,
    kp: float = 5.0,
    timeout: float = 10.0,
    settle_time: float = 0.2,
) -> bool:
    """
    Drive cart to a target position using P-control on velocity.

    Uses position feedback from SystemState (updated by IPC reader at 100Hz)
    and sends velocity commands to the PLC.  No TwinCAT changes needed -
    MC_MoveVelocity with MC_Aborting accepts arbitrary floats.

    Args:
        ser:          Serial port connected to PLC.
        state:        SystemState with live cart_position updates.
        target_m:     Target cart position in meters (0 = track center).
        tolerance_m:  Position tolerance in meters (default 10mm).
        max_vel_mms:  Maximum velocity command in mm/s (default 500).
        kp:           Proportional gain (1/s).  error_mm * kp = vel_mms.
                      At 100mm error -> 500mm/s, at 10mm -> 50mm/s.
        timeout:      Safety timeout in seconds.
        settle_time:  Time to wait after stopping for mechanical settle.

    Returns:
        True if target reached within tolerance, False if timed out.
    """
    t_start = time.time()
    poll_dt = 0.020  # 50Hz control loop

    while (time.time() - t_start) < timeout:
        sv = state.get_state_vector()
        current_m = sv[0]
        error_m = target_m - current_m
        error_mm = error_m * 1000.0

        if abs(error_m) < tolerance_m:
            # Within tolerance - stop and settle
            ser.write(encode_velocity_command(0.0))
            time.sleep(settle_time)
            return True

        # P-control: velocity proportional to error
        v_cmd = np.clip(kp * error_mm, -max_vel_mms, max_vel_mms)
        ser.write(encode_velocity_command(v_cmd))
        time.sleep(poll_dt)

    # Timeout - stop cart
    ser.write(encode_velocity_command(0.0))
    time.sleep(settle_time)
    sv = state.get_state_vector()
    print(f"  [move_cart_to_position] TIMEOUT: target={target_m:.3f}m, "
          f"actual={sv[0]:.3f}m, error={abs(sv[0]-target_m)*1000:.1f}mm")
    return abs(sv[0] - target_m) < tolerance_m * 2


# -- One-hot encoding (for neural networks) ----------------------------------

def action_to_onehot(action: int, n_actions: int = 3) -> np.ndarray:
    """Convert discrete action to one-hot vector."""
    onehot = np.zeros(n_actions, dtype=np.float32)
    onehot[action] = 1.0
    return onehot


def onehot_to_action(onehot: np.ndarray) -> int:
    """Convert one-hot vector to discrete action."""
    return int(np.argmax(onehot))
