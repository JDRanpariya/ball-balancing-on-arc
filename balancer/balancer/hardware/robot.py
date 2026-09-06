# balancer/hardware/robot.py

import time
import serial
import threading
import numpy as np
import gymnasium as gym

from balancer.hardware.constants import SYSTEM
from balancer.hardware import (
    SERIAL_CONFIG,
    SystemState,
    start_reader_threads,
    get_action_command,
    encode_discrete_action,
    encode_continuous_action,
    move_cart_to_position,
    DiscreteAction,
)
from balancer.envs.rewards import ball_gaussian_distance


class Robot(gym.Env):
    """
    Gymnasium environment for the physical ball-balancer robot.
    
    Uses standardized hardware modules for serial communication,
    sensor processing, and action encoding.
    """
    
    metadata = {"render_modes": ["human"], "render_fps": 20}
    
    def __init__(self, action_type: str = "discrete", control_dt: float = 0.05):
        super().__init__()
        
        self.control_dt = control_dt
        self.action_type = action_type
        self._next_step_time = time.perf_counter()
        self._last_stale_warn_time = 0.0  # rate-limits the stale-sensor warning
        
        # Serial setup
        self.ser_ipc = serial.Serial(
            SERIAL_CONFIG.ipc_port,
            SERIAL_CONFIG.baudrate,
            timeout=SERIAL_CONFIG.ipc_timeout,
        )
        self.ser_dist = serial.Serial(
            SERIAL_CONFIG.distance_port,
            SERIAL_CONFIG.baudrate,
            timeout=SERIAL_CONFIG.distance_timeout,
        )
        
        # State container
        self._state = SystemState()
        self._stop_event = threading.Event()
        
        # Start reader threads using standardized module
        self._ipc_thread, self._dist_thread = start_reader_threads(
            self._state,
            self.ser_ipc,
            self.ser_dist,
            self._stop_event,
        )
        
        # Gym spaces
        self.observation_space = gym.spaces.Box(
            low=np.array([-SYSTEM.CART_LIMIT, -1.0, -SYSTEM.BALL_LIMIT, -1.0], dtype=np.float32),
            high=np.array([SYSTEM.CART_LIMIT, 1.0, SYSTEM.BALL_LIMIT, 1.0], dtype=np.float32),
        )
        if self.action_type == "cont":
            self.action_space = gym.spaces.Box(
                low=-1.0, high=1.0, shape=(1,), dtype=np.float32)
        else:
            self.action_space = gym.spaces.Discrete(3)
        
        print("Robot initialized. Waiting for sensors to stabilize...")
        time.sleep(2)
    
    def step(self, action, action_type=None):
        # Send action using standardized encoding
        atype = action_type if action_type is not None else self.action_type
        cmd = get_action_command(action, atype)
        self.ser_ipc.write(cmd)
        
        # Maintain frequency
        self._next_step_time += self.control_dt
        sleep_time = self._next_step_time - time.perf_counter()
        if sleep_time > 0:
            time.sleep(sleep_time)
        else:
            self._next_step_time = time.perf_counter()
        
        # Get observation
        obs = self._state.get_state_array()
        obs = np.clip(obs, self.observation_space.low, self.observation_space.high)

        # Ball-sensor staleness watchdog - warn (rate-limited) if the ToF
        # reader appears frozen, without altering control flow/return value.
        if self._state.is_stale(0.5):
            now = time.perf_counter()
            if now - self._last_stale_warn_time > 2.0:
                self._last_stale_warn_time = now
                print(f"Warning: ball sensor stale ({self._state.ball_age():.2f}s "
                      f"since last update)")

        # Compute reward
        pos_limits = [SYSTEM.track_half, 0.075]
        reward = ball_gaussian_distance(obs, pos_limits)
        
        return obs, reward, False, False, {"dip": self._state.dip_reward}
    
    # 10 stratified initial cart positions (same as eval.py grid)
    RESET_POSITIONS = np.linspace(-0.70, 0.70, 10).tolist()

    def reset(self, seed=None, options=None):
        """
        Reset the environment by moving cart to a random stratified position.
        
        Uses 10 evenly-spaced positions from -0.7 to +0.7 m (matching the
        evaluation grid) to ensure diverse training experience.
        """
        super().reset(seed=seed)
        
        # Pick a random IC from the stratified grid
        target_pos = float(self.np_random.choice(self.RESET_POSITIONS))
        
        # Use position control to reach target
        ok = move_cart_to_position(
            self.ser_ipc, self._state, target_pos,
            tolerance_m=0.020, max_vel_mms=600.0, timeout=6.0,
        )
        if not ok:
            print(f"Warning: move_cart_to_position timed out (target={target_pos:.3f}m)")
        
        # Reset timing for consistent control loop
        self._next_step_time = time.perf_counter()
        
        # Wait for fresh sensor data
        self._state.data_event.clear()
        if not self._state.data_event.wait(timeout=2.0):
            print("Warning: Sensor data timeout during reset")
        
        # Get observation (thread-safe)
        obs = self._state.get_state_array()
        obs = np.clip(obs, self.observation_space.low, self.observation_space.high)
        
        with self._state.lock:
            cart_pos = self._state.cart_position
        print(f"Env Reset: target={target_pos:+.3f}m, actual={cart_pos:+.3f}m")
        
        return obs, {}
    
    def close(self):
        """
        Clean up resources: stop threads, close serial ports.
        """
        print("Closing Robot environment...")
        
        # Signal threads to stop
        self._stop_event.set()

        time.sleep(0.2)
        
        # Wait for threads to finish (with timeout)
        if self._ipc_thread is not None and self._ipc_thread.is_alive():
            self._ipc_thread.join(timeout=2.0)
            if self._ipc_thread.is_alive():
                print("Warning: IPC thread did not terminate cleanly")
        
        if self._dist_thread is not None and self._dist_thread.is_alive():
            self._dist_thread.join(timeout=2.0)
            if self._dist_thread.is_alive():
                print("Warning: Distance thread did not terminate cleanly")
        
        # Close serial ports
        try:
            if self.ser_ipc is not None and self.ser_ipc.is_open:
                self.ser_ipc.close()
        except Exception as e:
            print(f"Warning: Error closing IPC serial: {e}")
        
        try:
            if self.ser_dist is not None and self.ser_dist.is_open:
                self.ser_dist.close()
        except Exception as e:
            print(f"Warning: Error closing distance serial: {e}")
        
        print("Robot environment closed.")
    
    def render(self):
        """Render is handled by the physical system."""
        pass
    
    # ========================================================================
    # CONVENIENCE PROPERTIES
    # ========================================================================
    
    @property
    def cart_position(self) -> float:
        """Current cart position (thread-safe read)."""
        with self._state.lock:
            return self._state.cart_position
    
    @property
    def cart_velocity(self) -> float:
        """Current cart velocity (thread-safe read)."""
        with self._state.lock:
            return self._state.cart_velocity
    
    @property
    def ball_position(self) -> float:
        """Current ball angle (thread-safe read)."""
        with self._state.lock:
            return self._state.ball_position
    
    @property
    def ball_velocity(self) -> float:
        """Current ball angular velocity (thread-safe read)."""
        with self._state.lock:
            return self._state.ball_velocity