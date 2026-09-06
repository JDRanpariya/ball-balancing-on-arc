# balancer/hardware/serial_reader.py

"""
Unified serial reader threads for cart and ball sensors.

Provides consistent implementations used by:
- robot.py (Gym environment)
- collect_data.py (data collection)
- inference/eval.py (evaluation scripts)

"""

from __future__ import annotations
import time
import threading
from collections import deque
from typing import Optional, Callable, Dict, Any
import serial

from balancer.hardware.constants import SYSTEM
from .serial_config import SERIAL_CONFIG
from .system_state import SystemState
from .sensor_utils import (
    process_dual_sensors,
    parse_distance_line,
    parse_ipc_line,
    VelocityEstimator,
    PositionSmoother,
)


# ============================================================================

# ERROR THROTTLING

# ============================================================================

class ErrorThrottler:
    """Prevents error message spam by rate-limiting."""
    
    def __init__(self, interval: float = 2.0):
        self._interval = interval
        self._last_error_time: Dict[str, float] = {}
    
    def should_print(self, error_type: str) -> bool:
        now = time.time()
        last = self._last_error_time.get(error_type, 0)
        if now - last >= self._interval:
            self._last_error_time[error_type] = now
            return True
        return False


# ============================================================================

# IPC (CART) READER

# ============================================================================

def create_ipc_reader(
    state: SystemState,
    ser: serial.Serial,
    stop_event: threading.Event,
    track_center: float = SYSTEM.track_center,
) -> Callable[[], None]:
    """
    Factory for IPC (cart) reader thread function.
    
    Args:
        state: Shared state container
        ser: Open serial port for IPC
        stop_event: Threading event to signal shutdown
        track_center: Track center position in meters
        
    Returns:
        Thread target function
    """
    throttler = ErrorThrottler(interval=5.0)
    
    def read_ipc():
        buffer = bytearray()
        MAX_BUFFER = 1024
        
        while not stop_event.is_set():
            # Check stop event FIRST to exit quickly
            if stop_event.is_set():
                break
                
            try:
                # Use short timeout so we can check stop_event frequently
                if not ser.is_open:
                    if throttler.should_print("ipc_closed"):
                        print("IPC serial port closed, exiting reader.")
                    break
                
                data = ser.readline()
                if data:
                    buffer.extend(data)
                    if len(buffer) > MAX_BUFFER:
                        buffer.clear()
                        continue
                
                # Parse complete frames
                while b'\x02' in buffer and b'\x03' in buffer:
                    start = buffer.index(b'\x02')
                    end = buffer.index(b'\x03', start) + 1
                    frame = buffer[start:end].decode('utf-8', errors='ignore')
                    buffer = buffer[end:]
                    
                    result = parse_ipc_line(frame)
                    if result is not None:
                        pos_m, vel_mps = result
                        state.set_cart_state(
                            position=track_center - pos_m,
                            velocity=-vel_mps,
                        )
                        
            except serial.SerialException as e:
                # Serial port closed or disconnected
                if stop_event.is_set():
                    break  # Expected during shutdown
                if throttler.should_print("ipc_serial"):
                    print(f"IPC serial error: {e}")
                time.sleep(0.1)
                
            except OSError as e:
                # Port disconnected
                if stop_event.is_set():
                    break
                if throttler.should_print("ipc_os"):
                    print(f"IPC OS error: {e}")
                time.sleep(0.1)
                
            except Exception as e:
                if stop_event.is_set():
                    break
                if throttler.should_print("ipc_other"):
                    print(f"IPC reader error: {e}")
                time.sleep(0.01)
        
        # Clean exit
        print("IPC reader thread exiting.")
    
    return read_ipc


# ============================================================================

# DISTANCE (BALL) READER

# ============================================================================

def create_distance_reader(
    state: SystemState,
    ser_holder: Dict[str, Any],  # {"ser": serial.Serial, "config": SerialConfig}
    stop_event: threading.Event,
    on_reset: Optional[Callable[[], None]] = None,
    vel_buffer_len: int = 3,
    pos_buffer_len: int = 1,
    vel_clip: float = 1.5,
) -> Callable[[], None]:
    """
    Factory for distance sensor reader thread function.
    """
    vel_estimator = VelocityEstimator(buffer_len=vel_buffer_len, clip_value=vel_clip)
    pos_smoother = PositionSmoother(buffer_len=pos_buffer_len)
    throttler = ErrorThrottler(interval=5.0)
    
    def reset_sensor():
        """Reset the distance sensor serial connection with DTR toggle."""
        cfg = ser_holder.get("config", SERIAL_CONFIG)

        # Do the actual serial I/O and sleeps WITHOUT holding state.lock -
        # this can take >2s (Arduino bootloader) and must not block the
        # control loop's state reads.
        old_ser = ser_holder["ser"]
        try:
            old_ser.close()
        except Exception:
            pass

        new_ser = serial.Serial(
            cfg.distance_port,
            cfg.baudrate,
            timeout=cfg.distance_timeout,
        )
        # Toggle DTR to reset the Arduino - required after USB
        # detach/reattach cycles (WSL usbipd) where the Arduino
        # firmware stays in a stale state without a DTR pulse.
        new_ser.dtr = False
        time.sleep(0.1)
        new_ser.dtr = True
        time.sleep(2.0)  # Arduino bootloader needs ~1.5s
        new_ser.reset_input_buffer()

        # Only the actual state mutation needs the lock.
        with state.lock:
            ser_holder["ser"] = new_ser
            state.dist_status = True

        vel_estimator.reset()
        pos_smoother.reset()

        if on_reset:
            on_reset()

        print("Distance sensor reset (DTR toggled).")
    
    def read_distance():
        timeout_streak = 0
        
        while not stop_event.is_set():
            # Check stop event FIRST
            if stop_event.is_set():
                break
            
            try:
                with state.lock:
                    ser = ser_holder["ser"]
                    if not ser.is_open:
                        if throttler.should_print("dist_closed"):
                            print("Distance serial port closed, exiting reader.")
                        break
                    
                line = ser.readline()
                
            except serial.SerialException as e:
                if stop_event.is_set():
                    break
                if throttler.should_print("dist_serial"):
                    print(f"Distance serial error: {e}")
                time.sleep(0.1)
                continue
                
            except OSError as e:
                if stop_event.is_set():
                    break
                if throttler.should_print("dist_os"):
                    print(f"Distance OS error: {e}")
                time.sleep(0.1)
                continue
                
            except Exception as e:
                if stop_event.is_set():
                    break
                if throttler.should_print("dist_other"):
                    print(f"Distance read error: {e}")
                time.sleep(0.1)
                continue
            
            if not line:
                timeout_streak += 1
                if timeout_streak >= SERIAL_CONFIG.timeout_streak_limit:
                    if not stop_event.is_set():
                        try:
                            reset_sensor()
                        except Exception as e:
                            # Transient I/O error mid-reset (e.g. a WSL usbipd
                            # detach/reattach window) must not kill this thread -
                            # the next timeout streak will simply try again.
                            if throttler.should_print("dist_reset_fail"):
                                print(f"Distance sensor reset failed: {e}")
                            time.sleep(0.5)
                    timeout_streak = 0
                continue
            
            timeout_streak = 0
            decoded = line.decode('utf-8', errors='ignore').strip()
            
            # Handle status messages
            if decoded.startswith('Start'):
                with state.lock:
                    state.dist_status = True
                vel_estimator.reset()
                pos_smoother.reset()
                print("Distance sensor activated.")
                continue
            
            if decoded.startswith('Failed'):
                if not stop_event.is_set():
                    try:
                        reset_sensor()
                    except Exception as e:
                        if throttler.should_print("dist_reset_fail"):
                            print(f"Distance sensor reset failed: {e}")
                        time.sleep(0.5)
                continue
            
            # Parse sensor data
            with state.lock:
                if not state.dist_status:
                    continue
            
            result = parse_distance_line(decoded)
            if result is None:
                continue
            
            left_mm, right_mm, dip_reward = result
            
            # Filter invalid readings
            if left_mm > 300 or right_mm > 300:
                continue
            if left_mm == 65535 or right_mm == 65535:
                continue
            
            # Convert to angle
            theta = process_dual_sensors(left_mm, right_mm)
            smooth_pos = pos_smoother.update(theta)
            
            # Estimate velocity
            now = time.perf_counter()
            smooth_vel = vel_estimator.update(smooth_pos, now)
            
            # Update state
            state.set_ball_state(smooth_pos, smooth_vel, dip_reward,
                                 raw_left_mm=left_mm, raw_right_mm=right_mm)
        
        # Clean exit
        print("Distance reader thread exiting.")
    
    return read_distance


# ============================================================================

# CONVENIENCE STARTER

# ============================================================================

def start_reader_threads(
    state: SystemState,
    ser_ipc: serial.Serial,
    ser_distance: serial.Serial,
    stop_event: threading.Event,
) -> tuple:
    """
    Convenience function to start both reader threads.
    
    Returns:
        Tuple of (ipc_thread, distance_thread)
    """
    # Toggle DTR on the distance sensor to reset Arduino firmware.
    # Required after USB detach/reattach (WSL usbipd) and on fresh start.
    ser_distance.dtr = False
    time.sleep(0.1)
    ser_distance.dtr = True
    time.sleep(2.0)  # Arduino bootloader needs ~1.5s
    ser_distance.reset_input_buffer()

    ser_holder = {"ser": ser_distance, "config": SERIAL_CONFIG}
    
    ipc_func = create_ipc_reader(state, ser_ipc, stop_event)
    dist_func = create_distance_reader(state, ser_holder, stop_event)
    
    ipc_thread = threading.Thread(target=ipc_func, daemon=True, name="IPC-Reader")
    dist_thread = threading.Thread(target=dist_func, daemon=True, name="Dist-Reader")
    
    ipc_thread.start()
    dist_thread.start()
    
    return ipc_thread, dist_thread

