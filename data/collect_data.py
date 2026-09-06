#!/usr/bin/env python3
"""
Data collection for ball-on-arc system.

Uses standardized hardware utilities from balancer.hardware package.
Collects state-action-reward transitions and saves to HDF5.

Usage:
    python collect_data.py
    python collect_data.py --steps 1000000 --output dataset/my_data.h5
"""

import os
import queue
import time
import threading
import random
import signal
import argparse
from dataclasses import dataclass, field, asdict
from typing import Optional, List
from collections import deque
from balancer.hardware.action_utils import encode_velocity_command

import h5py
import numpy as np
import yaml
import json

# ============================================================================

# STANDARDIZED IMPORTS FROM BALANCER PACKAGE

# ============================================================================

from balancer.hardware.constants import SYSTEM
from balancer.hardware import (
    # Config
    SerialConfig,
    SERIAL_CONFIG,
    # State
    SystemState,
    # Actions
    DISCRETE_COMMANDS,
    encode_discrete_action,
    encode_continuous_action,
    DiscreteAction,
    # Readers
    create_ipc_reader,
    create_distance_reader,
)
from balancer.envs.rewards import ball_gaussian_distance

# ============================================================================

# CONFIGURATION

# ============================================================================

# -- Constants at module level -------------------------------
DISCRETE_VELOCITIES_MMS = [-900.0, 0.0, 900.0]   # mm/s -> what TwinCAT sees
MMS_TO_MS = 1.0 / 1000.0                          # dataset stores m/s


@dataclass
class CollectionConfig:
    """Data collection configuration."""

    # -- Collection parameters --
    run_steps: int = 3_000_000
    action_type: str = "discrete"  # "discrete" or "cont"
    control_period_s: float = 0.05  # 20 Hz

    # -- Output --
    hdf5_filename: str = "dataset/arcball_discrete_3M.h5"
    save_every_n: int = 5000

    # -- Policy --
    p_stick: float = 0.5  # Probability of repeating previous action

    # -- Smoothing (passed to readers) --
    vel_buffer_len: int = 3
    pos_buffer_len: int = 1
    vel_clip: float = 0.6

    # -- Reward --
    pos_limit: float = SYSTEM.BALL_LIMIT  # Ball position limit for reward calculation

    # -- Dataset columns --
    columns: List[str] = field(default_factory=lambda: [
        'prev_cart_pos', 'prev_cart_vel', 'prev_ball_pos', 'prev_ball_vel',
        'prev_raw_left_mm', 'prev_raw_right_mm',
        'action',
        'cur_cart_pos', 'cur_cart_vel', 'cur_ball_pos', 'cur_ball_vel',
        'cur_raw_left_mm', 'cur_raw_right_mm',
        'reward', 'dip_reward',
    ])


# ============================================================================

# LOGGING & DIAGNOSTICS

# ============================================================================

@dataclass
class CollectionStats:
    """Runtime statistics."""
    start_step: int = 0           # Step count at start (from resume)
    current_step: int = 0         # Current step count
    sensor_resets: int = 0
    sensor_65535_count: int = 0
    loop_overruns: int = 0
    start_time: float = 0.0       # Set when collection actually starts

    @property
    def steps_collected(self) -> int:
        """Number of steps collected THIS session (not including resume)."""
        return self.current_step - self.start_step

    def start_collection(self):
        """Call this when actual collection begins (after warmup)."""
        self.start_time = time.time()

    def print_summary(self):
        elapsed = time.time() - self.start_time

        # Rate based on NEW steps only
        new_steps = self.steps_collected
        rate = new_steps / elapsed if elapsed > 0 else 0

        print(f"\n{'='*60}")
        print(f"Collection Summary")
        print(f"{'='*60}")
        print(f"  Resumed from:       {self.start_step:,} rows")
        print(f"  New steps:          {new_steps:,}")
        print(f"  Total steps:        {self.current_step:,}")
        print(
            f"  Elapsed time:       {elapsed/3600:.2f} hours ({elapsed:.1f}s)")
        print(f"  Collection rate:    {rate:.1f} Hz")  # Now correct!
        print(f"  Target rate:        20.0 Hz")
        print(f"  Efficiency:         {100*rate/20:.1f}%")
        print(f"  Sensor resets:      {self.sensor_resets}")
        print(f"  65535 errors:       {self.sensor_65535_count}")
        print(f"  Loop overruns:      {self.loop_overruns}")
        print(f"{'='*60}")


class SensorErrorLogger:
    """Logs sensor errors to file with throttling."""

    def __init__(self, log_dir: str = "logs"):
        os.makedirs(log_dir, exist_ok=True)
        self.log_path = os.path.join(log_dir, "sensor_errors.txt")
        self._lock = threading.Lock()

    def log_65535(self, sensor: str, value: int, raw_line: str):
        with self._lock:
            with open(self.log_path, "a") as f:
                f.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} "
                        f"[{sensor}] {value} | {raw_line}\n")


# ============================================================================

# EXTENDED DISTANCE READER (with 65535 logging)

# ============================================================================

def create_data_collection_distance_reader(
    state: SystemState,
    ser_holder: dict,
    stop_event: threading.Event,
    stats: CollectionStats,
    error_logger: SensorErrorLogger,
    config: CollectionConfig,
) -> callable:
    """
    Extended distance reader that logs 65535 errors for data collection.

    Wraps the standard reader with additional error tracking.
    """
    from balancer.hardware.sensor_utils import (
        VelocityEstimator,
        PositionSmoother,
        parse_distance_line,
        process_dual_sensors,
    )

    vel_estimator = VelocityEstimator(
        buffer_len=config.vel_buffer_len,
        clip_value=config.vel_clip,
    )
    pos_smoother = PositionSmoother(buffer_len=config.pos_buffer_len)

    def reset_sensor():
        nonlocal vel_estimator, pos_smoother

        with state.lock:
            try:
                ser_holder["ser"].close()
            except Exception:
                pass

            import serial
            ser_holder["ser"] = serial.Serial(
                SERIAL_CONFIG.distance_port,
                SERIAL_CONFIG.baudrate,
                timeout=SERIAL_CONFIG.distance_timeout,
            )
            ser_holder["ser"].dtr = True
            ser_holder["ser"].rts = True
            state.dist_status = True

        vel_estimator.reset()
        pos_smoother.reset()
        stats.sensor_resets += 1
        print(f"Distance sensor reset (total: {stats.sensor_resets})")

    def read_distance():
        timeout_streak = 0

        # Initial warmup read
        try:
            ser_holder["ser"].readline()
            time.sleep(2)
        except Exception:
            pass

        while not stop_event.is_set():
            if stop_event.is_set():
                break

            try:
                line = ser_holder["ser"].readline()
            except Exception as e:
                if stop_event.is_set():
                    break
                print(f"Distance read error: {e}")
                time.sleep(0.1)
                continue

            if not line:
                timeout_streak += 1
                if timeout_streak >= SERIAL_CONFIG.timeout_streak_limit:
                    if not stop_event.is_set():
                        reset_sensor()
                    timeout_streak = 0
                continue

            timeout_streak = 0
            decoded = line.decode('utf-8', errors='ignore').strip()

            # Status messages
            if decoded.startswith('Start'):
                with state.lock:
                    state.dist_status = True
                vel_estimator.reset()
                pos_smoother.reset()
                print("Distance sensor activated.")
                continue

            if decoded.startswith('Failed'):
                if not stop_event.is_set():
                    reset_sensor()
                continue

            # Parse data
            with state.lock:
                if not state.dist_status:
                    continue

            result = parse_distance_line(decoded)
            if result is None:
                continue

            left_mm, right_mm, dip_reward = result

            # -- 65535 ERROR LOGGING (data collection specific) --
            if left_mm == 65535:
                stats.sensor_65535_count += 1
                error_logger.log_65535("S1", left_mm, decoded)
                continue
            if right_mm == 65535:
                stats.sensor_65535_count += 1
                error_logger.log_65535("S2", right_mm, decoded)
                continue

            # Filter out-of-range
            if left_mm > 300 or right_mm > 300:
                continue

            # Convert to angle
            theta = process_dual_sensors(left_mm, right_mm)
            smooth_pos = pos_smoother.update(theta)

            # Velocity
            now = time.time()
            smooth_vel = vel_estimator.update(theta, now)

            # Update state (including raw mm values)
            state.set_ball_state(smooth_pos, smooth_vel, dip_reward,
                                 raw_left_mm=left_mm, raw_right_mm=right_mm)

        print("Distance reader thread exiting.")

    return read_distance


# ============================================================================

# HDF5 SAVER THREAD

# ============================================================================

def saver_worker(
    save_q: queue.Queue,
    filename: str,
    stop_event: threading.Event,
    config: CollectionConfig,
):
    """Background thread that owns HDF5 file and appends chunks."""
    f = None
    dset = None

    try:
        os.makedirs(os.path.dirname(filename) or '.', exist_ok=True)
        f = h5py.File(filename, 'a')

        if 'dataset' in f:
            dset = f['dataset']

        while not stop_event.is_set() or not save_q.empty():
            try:
                data_np = save_q.get(timeout=0.5)
            except queue.Empty:
                continue

            if data_np is None:
                save_q.task_done()
                break

            if data_np.size == 0:
                save_q.task_done()
                continue

            if dset is None:
                dset = f.create_dataset(
                    'dataset',
                    shape=(0, data_np.shape[1]),
                    maxshape=(None, data_np.shape[1]),
                    chunks=True,
                    compression='gzip',
                    compression_opts=9,
                )
                f.attrs['columns'] = json.dumps(config.columns)

            rows = dset.shape[0]
            dset.resize(rows + data_np.shape[0], axis=0)
            dset[rows:rows + data_np.shape[0]] = data_np
            f.flush()
            save_q.task_done()

    finally:
        if f is not None:
            f.close()
        print("Saver thread exiting.")


# ============================================================================

# CONFIG PERSISTENCE

# ============================================================================

def save_collection_config(config: CollectionConfig):
    """Save config to YAML file alongside HDF5."""
    yaml_path = os.path.splitext(config.hdf5_filename)[0] + '.yaml'
    os.makedirs(os.path.dirname(yaml_path) or '.', exist_ok=True)

    if os.path.exists(yaml_path):
        print(f"Config exists: {yaml_path} (not overwritten)")
        return

    snapshot = {
        **asdict(config),
        'system_constants': {
            'arc_radius_m': SYSTEM.ARC_RADIUS_M,
            'track_start_m': SYSTEM.TRACK_START_M,
            'track_end_m': SYSTEM.TRACK_END_M,
            'left_center_mm': SYSTEM.LEFT_CENTER_MM,
            'right_center_mm': SYSTEM.RIGHT_CENTER_MM,
            'sensor_offset_rad': SYSTEM.SENSOR_OFFSET_RAD,
            'center_threshold_rad': SYSTEM.CENTER_THRESHOLD_RAD,
        },
        '_meta': {
            'start_time': time.strftime('%Y-%m-%d %H:%M:%S'),
            'host': os.uname().nodename,
        },
    }

    with open(yaml_path, 'w') as f:
        yaml.dump(snapshot, f, default_flow_style=False, sort_keys=False)
    print(f"Config saved: {yaml_path}")


def embed_config_in_hdf5(config: CollectionConfig):
    """Store config in HDF5 attributes."""
    path = config.hdf5_filename
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)

    with h5py.File(path, 'a') as f:
        if 'config' not in f.attrs:
            f.attrs['config'] = yaml.dump(asdict(config))
            f.attrs['columns'] = json.dumps(config.columns)
            f.attrs['start_time'] = time.strftime('%Y-%m-%d %H:%M:%S')
            f.attrs['arc_radius_m'] = SYSTEM.ARC_RADIUS_M
            f.attrs['track_center_m'] = SYSTEM.track_center
            print(f"Config embedded in {path}")


def get_existing_row_count(filename: str) -> int:
    """Get number of existing rows for resume capability."""
    if not os.path.exists(filename):
        return 0
    try:
        with h5py.File(filename, 'r') as f:
            if 'dataset' in f:
                count = f['dataset'].shape[0]
                print(f"Resuming: found {count:,} existing rows")
                return count
    except Exception as e:
        print(f"Warning: could not read existing HDF5: {e}")
    return 0


# ============================================================================

# MAIN DATA COLLECTION LOOP

# ============================================================================

def run_collection(config: CollectionConfig):
    """Main data collection loop."""
    import serial

    # -- Setup --
    save_collection_config(config)
    embed_config_in_hdf5(config)

    stats = CollectionStats()
    stats.start_step = get_existing_row_count(config.hdf5_filename)
    stats.current_step = stats.start_step  # Start from resume poin

    remaining = config.run_steps - stats.current_step
    if remaining <= 0:
        print(
            f"Already have {stats.current_step:,} rows >= {config.run_steps:,}. Done.")
        return

    print(
        f"Starting from step {stats.current_step:,}, collecting {remaining:,} more.")

    # -- Threading primitives --
    stop_event = threading.Event()
    save_q = queue.Queue(maxsize=8)

    # -- Signal handler for clean shutdown --
    def signal_handler(sig, frame):
        print("\nCtrl+C received, stopping...")
        stop_event.set()

    signal.signal(signal.SIGINT, signal_handler)

    # -- Serial ports --
    ser_ipc = serial.Serial(
        SERIAL_CONFIG.ipc_port,
        SERIAL_CONFIG.baudrate,
        timeout=SERIAL_CONFIG.ipc_timeout,
    )

    ser_distance = serial.Serial(
        SERIAL_CONFIG.distance_port,
        SERIAL_CONFIG.baudrate,
        timeout=SERIAL_CONFIG.distance_timeout,
    )
    ser_distance.dtr = True
    ser_distance.rts = True

    ser_holder = {"ser": ser_distance}

    # -- State container --
    state = SystemState()

    # -- Error logger --
    error_logger = SensorErrorLogger()

    # -- Start threads --
    ipc_func = create_ipc_reader(state, ser_ipc, stop_event)
    dist_func = create_data_collection_distance_reader(
        state, ser_holder, stop_event, stats, error_logger, config
    )

    ipc_thread = threading.Thread(
        target=ipc_func, daemon=True, name="IPC-Reader")
    dist_thread = threading.Thread(
        target=dist_func, daemon=True, name="Dist-Reader")
    saver_thread = threading.Thread(
        target=saver_worker,
        args=(save_q, config.hdf5_filename, stop_event, config),
        daemon=False,
        name="HDF5-Saver",
    )

    ipc_thread.start()
    dist_thread.start()
    saver_thread.start()

    print("Waiting for sensors to stabilize...")
    time.sleep(4)

    stats.start_collection()

    # -- Collection loop --
    dataset_buffer: List[list] = []
    prev_obs: Optional[list] = None

    prev_action_ms: Optional[float] = None     # velocity in m/s for dataset
    action_ms = 0.0
    prev_raw_left: int = 0
    prev_raw_right: int = 0

    t_next = time.perf_counter()
    pos_limits = [SYSTEM.track_half, config.pos_limit]

    try:
        while stats.current_step < config.run_steps and not stop_event.is_set():
            # Get current state
            cur_obs = state.get_state_vector()

            with state.lock:
                cur_dip_r = state.dip_reward
                cur_raw_left = state.raw_left_mm
                cur_raw_right = state.raw_right_mm

            # Skip if no valid data yet
            if cur_obs[0] == 0.0 and cur_obs[2] == 0.0:
                t_next += config.control_period_s
                sleep_time = t_next - time.perf_counter()
                if sleep_time > 0:
                    time.sleep(sleep_time)
                continue

            # -- Select action (both types use velocity protocol) --
            if config.action_type == 'discrete':
                if random.random() > config.p_stick:
                    vel_mms = random.choice(DISCRETE_VELOCITIES_MMS)
                else:
                    vel_mms = action_ms * 1000.0   # repeat previous

                action_ms = vel_mms * MMS_TO_MS    # -0.9 / 0.0 / +0.9

            else:  # cont
                if random.random() > config.p_stick:
                    vel_mms = random.uniform(-900.0, 900.0)
                else:
                    vel_mms = action_ms * 1000.0

                action_ms = vel_mms * MMS_TO_MS    # [-0.9, +0.9]

            # Same serial command for both - raw mm/s, no scaling
            ser_ipc.write(encode_velocity_command(vel_mms))

            # -- Record transition --
            if prev_obs is not None and prev_action_ms is not None:
                reward = ball_gaussian_distance(prev_obs, pos_limits)
                row = (prev_obs + [prev_raw_left, prev_raw_right]
                       + [prev_action_ms]
                       + cur_obs + [cur_raw_left, cur_raw_right]
                       + [reward, cur_dip_r])
                dataset_buffer.append(row)

            prev_obs = cur_obs
            prev_action_ms = action_ms
            prev_raw_left = cur_raw_left
            prev_raw_right = cur_raw_right

            # -- Save buffer periodically --
            if len(dataset_buffer) >= config.save_every_n:
                try:
                    chunk = np.array(dataset_buffer, dtype=float)
                    dataset_buffer = []
                    save_q.put(chunk, timeout=0.1)
                except queue.Full:
                    pass

            stats.current_step += 1

            # -- Maintain timing --
            t_next += config.control_period_s
            sleep_time = t_next - time.perf_counter()
            if sleep_time > 0:
                time.sleep(sleep_time)
            else:
                stats.loop_overruns += 1
                if stats.loop_overruns % 100 == 1:
                    print(f"WARNING Loop overran by {-sleep_time*1000:.1f}ms "
                          f"(total: {stats.loop_overruns})")
                t_next = time.perf_counter()

    except Exception as e:
        print(f"Collection error: {e}")
        import traceback
        traceback.print_exc()

    finally:
        # -- Cleanup --
        print("\nCleaning up...")

        # Stop threads
        stop_event.set()

        # Save remaining data
        if dataset_buffer:
            try:
                save_q.put(np.array(dataset_buffer, dtype=float), timeout=1.0)
            except queue.Full:
                print("Warning: could not save final buffer")

        # Signal saver to exit and wait
        try:
            save_q.put(None, timeout=1.0)
        except queue.Full:
            pass

        save_q.join()
        saver_thread.join(timeout=5.0)

        # Wait for reader threads
        time.sleep(0.2)
        ipc_thread.join(timeout=1.0)
        dist_thread.join(timeout=1.0)

        # Close serial ports
        try:
            ser_ipc.close()
        except Exception:
            pass
        try:
            ser_holder["ser"].close()
        except Exception:
            pass

        # Print summary
        stats.print_summary()


# ============================================================================

# CLI

# ============================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Collect ball-on-arc transition data",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--steps", type=int, default=3_000_000,
        help="Total steps to collect",
    )
    parser.add_argument(
        "--output", "-o", type=str, default="dataset/arcball_discrete_3M.h5",
        help="Output HDF5 filename",
    )
    parser.add_argument(
        "--hz", type=float, default=20.0,
        help="Collection frequency in Hz",
    )
    parser.add_argument(
        "--action-type", choices=["discrete", "cont"], default="discrete",
        help="Action type",
    )
    parser.add_argument(
        "--p-stick", type=float, default=0.5,
        help="Probability of repeating previous action",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    config = CollectionConfig(
        run_steps=args.steps,
        hdf5_filename=args.output,
        control_period_s=1.0 / args.hz,
        action_type=args.action_type,
        p_stick=args.p_stick,
    )

    print(f"{'='*60}")
    print(f"Ball-on-Arc Data Collection")
    print(f"{'='*60}")
    print(f"  Output:     {config.hdf5_filename}")
    print(f"  Steps:      {config.run_steps:,}")
    print(f"  Frequency:  {1/config.control_period_s:.0f} Hz")
    print(f"  Action:     {config.action_type}")
    print(f"{'='*60}")

    run_collection(config)


if __name__ == "__main__":
    main()
