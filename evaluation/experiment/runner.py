# evaluation/experiment/runner.py

"""
Experiment runner for controller evaluation.

Supports difficulty-stratified initial conditions via initial_conditions.py:
    Easy (20%)              - cart at ±0.10m, ball natural
    Medium (32%)            - cart at ±0.40m, ball natural
    Hard (28%)              - cart at ±0.65m, ball natural
    Extreme-opposite (10%) - cart at wall, ball forced opposite side
    Extreme-same (10%)     - cart at wall, ball forced same side
"""

import json
import os
import time
import threading
from typing import List, Optional

import numpy as np
import yaml

from .config import ExperimentConfig
from .initial_conditions import get_hardware_trial_configs

from balancer.hardware.constants import SYSTEM
from balancer.hardware import (
    SystemState,
    send_action,
    stop_cart,
    move_cart_to_position,
    move_cart_to_start_position,
)
from balancer.hardware.action_utils import encode_velocity_command


TRACK_HALF = SYSTEM.track_half


# =======================================================================
# BALL POSITIONING HELPERS
# =======================================================================

MIN_BALL_OFFSET = 0.02  # rad - minimum ball displacement for trial start


def _ensure_ball_offset(
    ser,
    state: SystemState,
    max_cart_drift_m: float = 0.05,
) -> None:
    """
    Ensure ball is away from center before trial starts.
    Oscillates cart back and forth until ball has sufficient offset.

    The shake velocity and duration are scaled to limit cart drift
    to at most max_cart_drift_m from the current position. This
    prevents the shake from destroying the carefully set cart position
    (e.g., at ±0.10m a full 600mm/s shake for 0.3s = 180mm travel).

    Args:
        ser:               Serial port.
        state:             SystemState.
        max_cart_drift_m:  Maximum allowed cart movement during shake.
    """
    MAX_ATTEMPTS = 6
    # Scale shake: v * t = drift -> t = drift / v
    # Use moderate velocity, short duration to limit cart displacement
    SHAKE_VELOCITY = 400.0  # mm/s (reduced from 600)
    # Duration limited so cart moves at most max_cart_drift_m
    max_duration = max_cart_drift_m / (SHAKE_VELOCITY / 1000.0)  # seconds
    SHAKE_DURATION = min(0.25, max_duration)  # cap at 250ms

    sv = state.get_state_vector()
    if abs(sv[2]) >= MIN_BALL_OFFSET:
        print(f"  Ball offset: {sv[2]:.4f} rad - ready.")
        return

    print(f"  Ball near center, shaking (v={SHAKE_VELOCITY:.0f}mm/s, "
          f"t={SHAKE_DURATION*1000:.0f}ms, max_drift={max_cart_drift_m*1000:.0f}mm)...")
    direction = 1.0
    for i in range(MAX_ATTEMPTS):
        ser.write(encode_velocity_command(direction * SHAKE_VELOCITY))
        time.sleep(SHAKE_DURATION)
        ser.write(encode_velocity_command(0.0))
        time.sleep(0.3)  # settle

        sv = state.get_state_vector()
        if abs(sv[2]) >= MIN_BALL_OFFSET:
            print(f"  Ball offset: {sv[2]:.4f} rad - ready (after {i+1} shakes).")
            return

        direction *= -1.0

    sv = state.get_state_vector()
    print(f"  WARN: Ball offset only {sv[2]:.4f} rad after {MAX_ATTEMPTS} shakes.")


def _position_ball_direction(
    ser,
    state: SystemState,
    cart_pos: float,
    desired_sign: int,
) -> None:
    """
    Position ball on a specific side (by sign of theta).

    Args:
        ser:          Serial port.
        state:        SystemState.
        cart_pos:     Current cart position (meters).
        desired_sign: +1 (ball positive theta) or -1 (ball negative theta).

    Strategy:
        - To get ball on SAME side as cart (e.g. cart right, want +theta): jolt
          AWAY from wall. Ball inertia keeps it stationary; relative to the
          moving cart it appears to roll toward the wall side -> same sign.
        - To get ball on OPPOSITE side: jolt TOWARD wall. Cart moves toward
          wall; ball lags behind, appearing to roll to the opposite side.
    """
    MAX_ATTEMPTS = 4
    JOLT_VEL = 500.0      # mm/s
    JOLT_DURATION = 0.35  # seconds
    SETTLE_WAIT = 0.4     # seconds for ball to roll
    RETREAT_DURATION = 0.2   # seconds (~100mm at JOLT_VEL) - room to build speed
    RETREAT_ROOM_MIN = 0.03  # m - below this, the cart has no room to jolt in
                              # that direction at all (it's pinned at/near the rail)

    cart_sign = np.sign(cart_pos) if abs(cart_pos) > 0.01 else 1.0
    # Determine if desired is "same" or "opposite" relative to cart
    want_same = (desired_sign == cart_sign)

    for attempt in range(MAX_ATTEMPTS):
        sv = state.get_state_vector()
        ball_theta = sv[2]

        # Check if ball is already where we want it
        if abs(ball_theta) >= MIN_BALL_OFFSET:
            ball_sign = np.sign(ball_theta)
            if ball_sign == desired_sign:
                print(f"  Ball direction OK: θ={ball_theta:.4f} "
                      f"(sign={desired_sign:+d})")
                return

        # Jolt to get ball on desired side
        if want_same:
            # Jolt AWAY from wall - ball inertia keeps it at wall side -> same sign
            jolt_dir = -cart_sign
        else:
            # Jolt TOWARD wall - ball lags, ends up on opposite side
            jolt_dir = cart_sign

        label = "away from wall" if want_same else "toward wall"

        # If the cart is already pinned at (or has no meaningful room left
        # toward) the rail the jolt direction points at, a single jolt is a
        # no-op: there's nowhere left to move, so no inertial impulse ever
        # reaches the ball (this is exactly the extreme_opposite-at-a-wall
        # case). Retreat off the wall first, then jerk back into it so the
        # cart actually has room to build speed before recontacting the rail.
        room_in_jolt_dir = SYSTEM.CART_LIMIT - jolt_dir * cart_pos
        if room_in_jolt_dir < RETREAT_ROOM_MIN:
            print(f"  Jolting {label} (attempt {attempt+1}/{MAX_ATTEMPTS}, "
                  f"retreating first - no room at the wall)...")
            ser.write(encode_velocity_command(-jolt_dir * JOLT_VEL))
            time.sleep(RETREAT_DURATION)
            ser.write(encode_velocity_command(0.0))
            time.sleep(0.1)
        else:
            print(f"  Jolting {label} (attempt {attempt+1}/{MAX_ATTEMPTS})...")

        ser.write(encode_velocity_command(jolt_dir * JOLT_VEL))
        time.sleep(JOLT_DURATION)
        ser.write(encode_velocity_command(0.0))
        time.sleep(SETTLE_WAIT)

    # Final check
    sv = state.get_state_vector()
    print(f"  Ball positioning done: θ={sv[2]:.4f} rad "
          f"(wanted sign={desired_sign:+d})")


def _setup_initial_condition(
    ser,
    state: SystemState,
    trial_config: dict,
    action_type: str,
) -> None:
    """
    Move cart to target position and set up ball direction.

    Args:
        ser:          Serial port.
        state:        SystemState.
        trial_config: Dict from get_hardware_trial_configs() with keys:
                      start_position, expected_theta_sign, difficulty.
        action_type:  "discrete" or "cont".
    """
    target = trial_config["start_position"]
    theta_sign = trial_config["expected_theta_sign"]
    difficulty = trial_config["difficulty"]
    WALL_THRESHOLD = SYSTEM.CART_LIMIT - 0.02  # 0.7565m

    print(f"  Moving cart to {target:+.3f}m "
          f"(difficulty={difficulty}, theta_sign={theta_sign:+d})")

    if abs(target) >= WALL_THRESHOLD:
        # Wall targets: use duration-based drive (guaranteed wall contact)
        direction = "left" if target < 0 else "right"
        move_cart_to_start_position(
            ser, state,
            direction=direction,
            action_type="cont",
            velocity=900.0,
            duration=3.0,
        )
    else:
        # Intermediate targets: closed-loop position control
        ok = move_cart_to_position(
            ser, state,
            target_m=target,
            tolerance_m=0.010,
            max_vel_mms=500.0,
            timeout=10.0,
        )
        if not ok:
            print(f"  WARN: Position control failed for target={target:.3f}m")

    # Brief settle for cart vibration
    time.sleep(0.3)

    # Ball direction setup
    if theta_sign == 0:
        # Natural: just ensure ball is displaced from center
        # Limit shake drift based on distance from wall
        room_to_wall = SYSTEM.CART_LIMIT - abs(target)
        max_drift = min(0.05, room_to_wall * 0.3)  # at most 30% of room, cap 50mm
        _ensure_ball_offset(ser, state, max_cart_drift_m=max_drift)
    else:
        # Forced direction
        _position_ball_direction(ser, state, target, theta_sign)
        # Ensure minimum offset after direction positioning
        sv = state.get_state_vector()
        if abs(sv[2]) < MIN_BALL_OFFSET:
            room_to_wall = SYSTEM.CART_LIMIT - abs(target)
            max_drift = min(0.05, room_to_wall * 0.3)
            if max_drift > 0.005:  # only shake if there's room (>5mm)
                _ensure_ball_offset(ser, state, max_cart_drift_m=max_drift)
            else:
                print(f"  WARN: Ball still near center but no room to shake "
                      f"(at wall). Proceeding anyway.")

    # Re-verify cart position - shake/jolt may have drifted it
    # Only correct for non-wall targets (wall targets are at physical limit)
    WALL_THRESHOLD = SYSTEM.CART_LIMIT - 0.02
    if abs(target) < WALL_THRESHOLD:
        sv = state.get_state_vector()
        cart_error_m = abs(sv[0] - target)
        if cart_error_m > 0.020:  # more than 20mm drift
            print(f"  Cart drifted {cart_error_m*1000:.0f}mm during ball setup, "
                  f"correcting...")
            move_cart_to_position(
                ser, state,
                target_m=target,
                tolerance_m=0.010,
                max_vel_mms=300.0,  # gentler to avoid disturbing ball
                timeout=5.0,
            )
            time.sleep(0.2)


# ======================================================================
# USB DEVICE RESET (WSL usbipd workaround)
# =======================================================================

def _reset_distance_sensor_usb(
    busid: str = "2-5",
    timeout: float = 5.0,
) -> bool:
    """
    Detach and reattach the distance sensor USB device via usbipd.

    This is a WSL-specific workaround for when the distance sensor
    stops responding. Requires usbipd-win installed on the Windows host.

    Args:
        busid:   USB bus ID of the distance sensor (default "2-5").
        timeout: Max seconds to wait for reattach.

    Returns:
        True if commands succeeded, False otherwise.
    """
    import subprocess

    try:
        print(f"  [USB] Detaching distance sensor (busid={busid})...")
        subprocess.run(
            ["usbipd.exe", "detach", "--busid", busid],
            timeout=timeout, capture_output=True,
        )
        time.sleep(1.0)

        print(f"  [USB] Reattaching distance sensor (busid={busid})...")
        result = subprocess.run(
            ["usbipd.exe", "attach", "--wsl", "--busid", busid],
            timeout=timeout, capture_output=True, text=True,
        )
        time.sleep(2.0)  # Wait for device to enumerate in WSL

        if result.returncode == 0:
            print(f"  [USB] Distance sensor reattached successfully.")
            return True
        else:
            print(f"  [USB] Reattach failed: {result.stderr.strip()}")
            return False

    except FileNotFoundError:
        print("  [USB] usbipd.exe not found - not running in WSL?")
        return False
    except subprocess.TimeoutExpired:
        print("  [USB] usbipd command timed out.")
        return False
    except Exception as e:
        print(f"  [USB] Error: {e}")
        return False


# =======================================================================
# SINGLE TRIAL
# =======================================================================

def run_trial(
    controller,
    ser_ipc,
    state: SystemState,
    stop_event: threading.Event,
    cfg: ExperimentConfig,
    trial_config: Optional[dict] = None,
) -> dict:
    """
    Run one trial and return a log dict.
    Returns empty dict if sensor timed out.

    Args:
        controller:    Controller instance with .step() and .reset().
        ser_ipc:       Serial port to PLC.
        state:         Shared SystemState.
        stop_event:    Stop signal.
        cfg:           Experiment configuration.
        trial_config:  Dict from get_hardware_trial_configs().
                       If None, uses legacy wall-drive behavior.
    """
    log = {
        "states": [],
        "raw_states": [],
        "actions": [],
        "timestamps": [],
        "controller_time_ms": [],
        "controller_cpu_time_ms": [],
        "send_time_ms": [],
        "loop_period_ms": [],
        "settling_time": None,
        "success": False,
        "ise_theta": None,
        "ise_vel": None,
        "ise_u": None,
        "difficulty": trial_config["difficulty"] if trial_config else "unknown",
        "cart_target_m": trial_config["start_position"] if trial_config else None,
        "expected_theta_sign": trial_config["expected_theta_sign"] if trial_config else 0,
        "description": trial_config.get("description", "") if trial_config else "",
    }

    max_vals = np.array([TRACK_HALF, 1.0, SYSTEM.BALL_LIMIT, 1.0])

    controller.reset()

    # Set up initial condition
    if trial_config:
        _setup_initial_condition(ser_ipc, state, trial_config, cfg.action_type)
    else:
        # Legacy: drive to left wall
        move_cart_to_start_position(
            ser_ipc, state,
            direction="left",
            action_type=cfg.action_type,
            velocity=900.0,
        )
        _ensure_ball_offset(ser_ipc, state)

    # Wait for fresh sensor data
    print(f"  Waiting for sensor data...")
    state.data_event.clear()
    if not state.data_event.wait(timeout=5.0):
        print("  ERROR: Sensor timed out. Aborting trial.")
        return {}
    print("  Sensor ready! Starting control.")
    state.data_event.clear()

    # Capture initial state
    initial_state = state.get_state_vector()
    log["initial_state"] = list(initial_state)

    start_time = time.perf_counter()
    next_loop_time = start_time + cfg.Ts

    settle_enter_time = None
    prev_noisy_pos = np.zeros(4)
    first_step = True

    while (time.perf_counter() - start_time) < cfg.run_time:
        loop_start = time.perf_counter()
        if stop_event.is_set():
            break

        state_vec = state.get_state_vector()

        if cfg.noise_sigma > 0:
            pos_noise_cart = np.random.normal(
                0, cfg.noise_sigma * SYSTEM.CART_LIMIT)
            pos_noise_ball = np.random.normal(
                0, cfg.noise_sigma * SYSTEM.BALL_LIMIT)

            noisy_cart_pos = state_vec[0] + pos_noise_cart
            noisy_ball_pos = state_vec[2] + pos_noise_ball

            if first_step:
                noisy_cart_vel = state_vec[1]
                noisy_ball_vel = state_vec[3]
                first_step = False
            else:
                noisy_cart_vel = (noisy_cart_pos - prev_noisy_pos[0]) / cfg.Ts
                noisy_ball_vel = (noisy_ball_pos - prev_noisy_pos[2]) / cfg.Ts

            noisy_state = np.clip(
                np.array([noisy_cart_pos, noisy_cart_vel,
                         noisy_ball_pos, noisy_ball_vel]),
                -max_vals, max_vals,
            )
            prev_noisy_pos[0] = noisy_cart_pos
            prev_noisy_pos[2] = noisy_ball_pos
        else:
            noisy_state = np.array(state_vec)

        t0, cpu0 = time.perf_counter(), time.process_time()
        u = controller.step(noisy_state, cart_limit=TRACK_HALF - 0.01)
        t1, cpu1 = time.perf_counter(), time.process_time()

        t2 = time.perf_counter()
        send_action(ser_ipc, u, cfg.action_type)
        t3 = time.perf_counter()

        elapsed = time.perf_counter() - start_time
        log["controller_time_ms"].append((t1 - t0) * 1000)
        log["controller_cpu_time_ms"].append((cpu1 - cpu0) * 1000)
        log["send_time_ms"].append((t3 - t2) * 1000)
        log["states"].append(noisy_state.tolist())
        log["raw_states"].append(list(state_vec))
        log["actions"].append(float(u))
        log["timestamps"].append(elapsed)
        log["loop_period_ms"].append((time.perf_counter() - loop_start) * 1000)

        next_loop_time += cfg.Ts
        sleep_time = next_loop_time - time.perf_counter()
        if sleep_time > 0:
            time.sleep(sleep_time)
        else:
            next_loop_time = time.perf_counter()

        in_band = abs(state_vec[2]) < cfg.settling_band

        if in_band:
            if settle_enter_time is None:
                settle_enter_time = elapsed
            if elapsed - settle_enter_time >= cfg.settling_duration:
                log["settling_time"] = settle_enter_time
                log["success"] = True
                break
        else:
            settle_enter_time = None

    # Compute ISE metrics
    if log["states"] and log["timestamps"]:
        arr = np.asarray(log["raw_states"] or log["states"], float)
        theta_arr = arr[:, 2]
        time_arr = np.asarray(log["timestamps"], float)
        u_arr = np.asarray(log["actions"], float)

        from experiment.runner_sim import _compute_ise
        ise_theta, ise_vel, ise_u = _compute_ise(time_arr, theta_arr, u_arr)
        log["ise_theta"] = ise_theta
        log["ise_vel"] = ise_vel
        log["ise_u"] = ise_u

    stop_cart(ser_ipc, action_type=cfg.action_type)

    return log


# =======================================================================
# EXPERIMENT (MULTI-CONTROLLER, MULTI-TRIAL)
# =======================================================================

def run_experiment(
    controllers: list,
    ser_ipc,
    state: SystemState,
    stop_event: threading.Event,
    cfg: ExperimentConfig,
    dist_sensor_busid: str = "2-5",
) -> dict:
    """
    Run all controllers for num_trials each.

    Uses difficulty-stratified initial conditions from initial_conditions.py.
    The trial schedule is deterministic and identical for every controller.
    """
    os.makedirs(cfg.output_dir, exist_ok=True)

    with open(cfg.config_path, "w") as f:
        yaml.dump(cfg.to_dict(), f, default_flow_style=False)
    print(f"Config saved -> {cfg.config_path}")

    results = _load_or_init(cfg)

    # Build the trial schedule once - same for all controllers
    schedule = get_hardware_trial_configs(cfg.num_trials)

    # Log the schedule in results metadata
    if "schedule" not in results:
        results["schedule"] = schedule

    for controller in controllers:
        ctrl_name = _controller_name(controller)
        trials = results["controllers"].setdefault(ctrl_name, [])
        done = len([t for t in trials if t.get("success") is not None])
        remaining = cfg.num_trials - done

        print(f"\n-- {ctrl_name} --  ({done}/{cfg.num_trials} done)")

        sensor_fail_streak = 0
        MAX_SENSOR_FAILS = 3  # trigger USB reset after this many consecutive

        for trial_idx in range(done, cfg.num_trials):
            if stop_event.is_set():
                print("Stop requested - exiting experiment.")
                break

            trial_config = schedule[trial_idx]

            print(f"  Trial {trial_idx + 1}/{cfg.num_trials}"
                  f"  ({trial_config['difficulty']}, "
                  f"cart={trial_config['start_position']:+.3f}m, "
                  f"θ_sign={trial_config['expected_theta_sign']:+d}, "
                  f"{trial_config['description']})")

            log = run_trial(
                controller, ser_ipc, state, stop_event, cfg,
                trial_config=trial_config,
            )

            if not log:
                sensor_fail_streak += 1
                print(f"  Trial aborted (sensor timeout). "
                      f"Streak: {sensor_fail_streak}/{MAX_SENSOR_FAILS}")

                if sensor_fail_streak >= MAX_SENSOR_FAILS:
                    print(f"  WARNING  {MAX_SENSOR_FAILS} consecutive sensor failures "
                          f"- attempting USB reset...")
                    if _reset_distance_sensor_usb(busid=dist_sensor_busid):
                        sensor_fail_streak = 0
                        time.sleep(3.0)  # Extra settle after USB reset
                    else:
                        print("  [FAIL] USB reset failed. Continuing anyway...")
                continue

            # Successful trial (even if controller failed to settle)
            sensor_fail_streak = 0
            trials.append(log)
            results["controllers"][ctrl_name] = trials

            _save_results(results, cfg.checkpoint_path)

            st_str = (f"{log['settling_time']:.2f}s"
                      if log['settling_time'] is not None else "FAIL")
            print(
                f"  -> {st_str} | "
                f"success={log['success']} | "
                f"{trial_config['difficulty']}")

    _save_results(results, cfg.checkpoint_path)
    print(f"\nResults saved -> {cfg.checkpoint_path}")
    return results


# =======================================================================
# HELPERS
# =======================================================================

def _controller_name(controller) -> str:
    name = controller.__class__.__name__
    if hasattr(controller, "model_name"):
        name += f"_{controller.model_name}"
    return name


def _load_or_init(cfg: ExperimentConfig) -> dict:
    if cfg.resume and os.path.exists(cfg.checkpoint_path):
        with open(cfg.checkpoint_path) as f:
            data = json.load(f)
        print(f"Resuming from checkpoint -> {cfg.checkpoint_path}")
        return data
    return {
        "name": cfg.name,
        "env_label": cfg.env_label,
        "noise_sigma": cfg.noise_sigma,
        "run_time": cfg.run_time,
        "Ts": cfg.Ts,
        "timestamp": cfg.timestamp,
        "controllers": {},
    }


def _save_results(results: dict, path: str):
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(results, f, indent=2)
    os.replace(tmp, path)
