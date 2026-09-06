"""
Shared simulation utilities for controller tuning and testing.

Simulation model
----------------
All controllers output u ∈ [-1, 1].  The simulation plant can interpret
this in two ways, matching the two hardware deployment modes:

  Continuous (action_type="cont"):
      v_cmd = clip(u, ±max_cart_velocity) m/s
      Smooth proportional velocity command.  Matches TwinCAT MC_MoveVelocity.

  Discrete (action_type="discrete"):
      u is thresholded to {LEFT, STOP, RIGHT} -> v_cmd ∈ {-0.9, 0, +0.9}
      Bang-bang control.  Matches TwinCAT McJogging.

Both modes use the same underlying NonLinearDynamics with S-curve motor
model (τ_v = 0.15 s, command_delay = 1 step).

Tuning recommendation:
  - Tune on "cont" for smooth CMA-ES objectives
  - Validate on "discrete" to check quantization robustness
  - Tune on "both" (averaged cost) for maximum deployment flexibility

"""

import numpy as np
import matplotlib.pyplot as plt
import os
from balancer.core.dynamics import NonLinearDynamics
from balancer.hardware.constants import SYSTEM

# --- Constants (unified with hardware eval) ---

CART_MIN, CART_MAX = -SYSTEM.CART_LIMIT, SYSTEM.CART_LIMIT
THETA_MIN, THETA_MAX = -SYSTEM.BALL_LIMIT, SYSTEM.BALL_LIMIT
CART_LIMIT = SYSTEM.CART_LIMIT    # operational limit for controllers

FAIL_TIME = SYSTEM.FAIL_TIME
SETTLING_TOL = 0.003           # tighter for tuning - avoids in-band oscillation
SETTLING_DURATION = 0.5         # shorter duration for tuning convergence speed
DISCRETE_THRESHOLD = SYSTEM.DISCRETE_THRESHOLD

# Hardware eval uses SYSTEM.SETTLING_BAND=0.01 (wider due to dip/jerk)
# Tuning uses 0.003 to force controllers to achieve true precision


# --- Dynamics cache (keyed by (Ts, action_type)) ---

_DYN_CACHE: dict = {}


def _get_dyn(Ts: float, action_type: str = "cont") -> NonLinearDynamics:
    key = (Ts, action_type)
    if key not in _DYN_CACHE:
        _DYN_CACHE[key] = NonLinearDynamics(
            action_type=action_type,
            kinematics_integrator="rk4",
            velocity_lag_tau=0.15,
            motor_model="first_order",
            command_delay_steps=1,
            tau=Ts,
        )
    return _DYN_CACHE[key]

# --- Action conversion ---


def continuous_to_discrete_action(u, threshold=DISCRETE_THRESHOLD):
    """
    Threshold continuous u ∈ [-1, 1] to discrete action {0, 1, 2}.

    Matches balancer.hardware.action_utils.continuous_to_discrete logic:
        u >  threshold -> 2 (RIGHT, v_cmd = +0.9)
        u < -threshold -> 0 (LEFT,  v_cmd = -0.9)
        else           -> 1 (STOP,  v_cmd =  0.0)
    """
    if u > threshold:
        return 2
    elif u < -threshold:
        return 0
    return 1

# --- Metrics ---


def settling_time(theta_hist, time_hist, tol=SETTLING_TOL,
                  duration=SETTLING_DURATION):
    """
    Time at which theta enters and STAYS within ±tol for `duration` seconds.

    Returns FAIL_TIME if never achieved.
    Identical definition used in tuning, sim eval, and hardware eval.
    """
    theta_abs = np.abs(theta_hist)
    if len(theta_abs) == 0 or np.any(np.isnan(theta_abs)):
        return FAIL_TIME

    in_band = theta_abs < tol
    # Walk forward: find first index where ball stays in band for `duration`
    enter_time = None
    for i in range(len(time_hist)):
        if in_band[i]:
            if enter_time is None:
                enter_time = time_hist[i]
            # Check if we've been in band long enough
            if time_hist[i] - enter_time >= duration:
                return enter_time  # settling happened at enter_time
        else:
            enter_time = None  # reset - left the band

    return FAIL_TIME  # never settled


def compute_metrics(time, theta, u):
    st = settling_time(theta, time)
    overshoot, boundary_violations = compute_overshoot(theta)
    energy = float(np.sqrt(np.nanmean(u**2)))          # RMS control effort
    jerk = float(np.sqrt(np.nanmean(np.diff(u)**2))) if len(u) > 2 else 0.0
    return {
        'settling_time':       st,
        'overshoot':           overshoot,
        'boundary_violations': boundary_violations,
        'energy':              energy,
        'jerk':                jerk,
    }


def compute_metrics_ise(time, theta, u, theta_dot=None):
    """
    ISE-based metrics for tuning.
    theta_dot: if None, finite-differenced from theta (matches hardware).
    """
    dt = np.diff(time)
    T = max(float(time[-1] - time[0]), 1e-6)

    # theta_dot is length N-1 when finite-differenced, same as dt
    if theta_dot is None:
        theta_dot = np.diff(theta) / np.maximum(dt, 1e-6)

    # All arrays used in integration are length N-1
    # theta[:-1], dt, gate, theta_dot must all be the same length
    n = min(len(theta) - 1, len(dt), len(theta_dot))
    th = theta[:n]          # N-1 values at left edges of intervals
    u_ = u[:n]              # same
    td = theta_dot[:n]      # already N-1, slice to n for safety
    dt_ = dt[:n]

    # -- ISE theta --------------------------------------------------------
    ise_theta = float(np.sum(th**2 * dt_) / T)

    # -- ISE velocity gated near center -----------------------------------
    sigma_gate = 0.020 / 2.354820045  # FWHM=20mm matches hardware dip
    gate = np.exp(-th**2 / (2 * sigma_gate**2))
    ise_vel = float(np.sum(td**2 * gate * dt_) / T)   # td not td[:-1]

    # -- ISE effort -------------------------------------------------------
    ise_u = float(np.sum(u_**2 * dt_) / T)

    # -- Boundary violations -----------------------------------------------
    bv = int(np.sum(np.abs(theta) >= (abs(THETA_MAX) - 1e-4)))

    # -- Settling time: eval only ------------------------------------------
    st = settling_time(theta, time)
    overshoot, _ = compute_overshoot(theta)

    return {
        "ise_theta":           ise_theta,
        "ise_vel":             ise_vel,
        "ise_u":               ise_u,
        "boundary_violations": bv,
        "settling_time":       st,
        "overshoot":           overshoot,
    }


def compute_overshoot(theta, tol=SETTLING_TOL):
    """
    Max |θ| after first entering the settling band.

    If the controller never enters the band, overshoot = max|θ| (whole trace).
    Also counts how many times θ hits the physical limits (±THETA_MAX).
    """
    theta_abs = np.abs(theta)

    # Boundary violations: times theta hits the rail
    boundary_violations = int(np.sum(
        theta_abs >= (abs(THETA_MAX) - 1e-4)
    ))

    # Find first time ball enters settling band
    in_band = theta_abs < tol
    first_entry = np.argmax(in_band)  # first True index

    if not in_band[first_entry]:
        # Never entered band
        return float(np.nanmax(theta_abs)), boundary_violations

    # Max |θ| after first entry into band
    post_entry = theta_abs[first_entry:]
    overshoot = float(np.nanmax(post_entry))

    return overshoot, boundary_violations

# --- Initial Conditions ---


def get_tuning_initial_conditions():
    """
    7 ICs that cover the important failure modes for CMA-ES tuning.
    Full 14-IC set is used only for final validation (_save_best_sim).

    Selection rationale:
        ic0 - near zero: tests fine regulation, dip behaviour
        ic1 - large positive theta: tests recovery from edge
        ic2 - large negative theta: symmetry check
        ic3 - cart left + positive theta: coupled recovery (opposite side)
        ic4 - cart right + negative theta: opposite side
        ic5 - cart right + positive theta: same-side (hardest constraint case)
        ic6 - cart left + negative theta: same-side (symmetry)
    """
    eps = 1e-3
    return [
        # small perturbation
        np.array([0.0,          0.0,  0.01,          0.0]),
        np.array([0.0,          0.0,  THETA_MAX-eps,  0.0]),  # large positive
        np.array([0.0,          0.0,  THETA_MIN+eps,  0.0]),  # large negative
        np.array([CART_MIN+eps, 0.0,  0.06,           0.0]),  # cart left, ball right
        np.array([CART_MAX-eps, 0.0, -0.03,           0.0]),  # cart right, ball left
        np.array([CART_MAX-eps, 0.0,  0.03,           0.0]),  # same-side: cart right, ball right
        np.array([CART_MIN+eps, 0.0, -0.03,           0.0]),  # same-side: cart left, ball left
    ]


def get_test_initial_conditions():
    eps = 1e-3
    return [
        np.array([0.0,           0.0, 0.0,           0.0]),
        np.array([0.0,           0.0, 0.01,          0.0]),
        np.array([0.0,           0.0, -0.01,         0.0]),
        np.array([0.0,           0.0, THETA_MAX-eps, 0.0]),
        np.array([0.0,           0.0, THETA_MIN+eps, 0.0]),
        np.array([CART_MIN,      0.0, THETA_MIN,     0.0]),
        np.array([CART_MIN,      0.0, THETA_MAX,     0.0]),
        np.array([CART_MAX,      0.0, THETA_MIN,     0.0]),
        np.array([CART_MAX,      0.0, THETA_MAX,     0.0]),
        np.array([CART_MIN+eps,  0.0, THETA_MIN+eps, 0.0]),
        np.array([CART_MAX-eps,  0.0, THETA_MAX-eps, 0.0]),
        np.array([-0.5,           0.0, 0.03,          0.0]),
        np.array([-0.74,         0.0, 0.0,           0.0]),
        np.array([0.76,          0.0, 0.0,           0.0]),
    ]

# --- Simulation ---


def simulate_controller(controller, s0, Ts=0.05, sim_T=FAIL_TIME,
                        action_type="cont"):
    """
    Simulate a controller against the velocity-input plant.

    The controller always outputs u ∈ [-1, 1].  The plant mode determines
    how that signal drives the cart:

        action_type="cont":
            v_cmd = clip(u, ±0.9) m/s  - smooth proportional velocity.
            Best for CMA-ES tuning (differentiable objective).

        action_type="discrete":
            u is thresholded to action ∈ {0=LEFT, 1=STOP, 2=RIGHT}
            v_cmd ∈ {-0.9, 0, +0.9} m/s  - bang-bang.
            Matches hardware McJogging mode exactly.

    Both modes use NonLinearDynamics with S-curve τ_v=0.15, delay=1.

    Args:
        controller:  has .step(state, cart_limit) -> float ∈ [-1, 1]
        s0:          initial state [x, ẋ, θ, θ̇]
        Ts:          sampling period (s)
        sim_T:       total simulation time (s)
        action_type: "cont" or "discrete"

    Returns:
        time, theta_hist, u_hist  (numpy arrays)
        u_hist contains the raw controller output (not the discretised action)
    """
    dyn = _get_dyn(Ts, action_type)

    controller.reset()
    steps = int(sim_T / Ts)
    s = s0.copy()
    theta_hist, u_hist, time_hist = [], [], []

    # Ball sensor staleness: ~10Hz sensor in 20Hz control loop
    # Mean hold = control_rate / sensor_rate = (1/Ts) / 10
    ball_obs_rate = 10.0
    ball_hold_mean = max(1.0, (1.0 / Ts) / ball_obs_rate)  # 2.0 at 20Hz
    ball_countdown = 1
    last_ball_obs = np.array([s[2], s[3]])
    rng = np.random.default_rng(42)

    for k in range(steps):
        t = k * Ts

        # Build observation: fresh cart + possibly stale ball
        obs = np.array([s[0], s[1], last_ball_obs[0], last_ball_obs[1]])

        # Controller always outputs u ∈ [-1, 1]
        u = controller.step(state=obs, cart_limit=CART_LIMIT)

        # Feed to plant - plant handles the mapping internally
        if action_type == "discrete":
            action = continuous_to_discrete_action(u)
            s = dyn(s, action)
        else:
            s = dyn(s, u)

        if np.any(np.isnan(s)) or np.any(np.isinf(s)):
            break

        # Update ball sensor (stale model)
        ball_countdown -= 1
        if ball_countdown <= 0:
            last_ball_obs[0] = s[2]
            last_ball_obs[1] = s[3]
            ball_countdown = int(rng.geometric(p=1.0 / ball_hold_mean))

        # Hard wall: clamp position AND zero velocity on contact
        if s[0] <= CART_MIN or s[0] >= CART_MAX:
            s[0] = np.clip(s[0], CART_MIN, CART_MAX)
            s[1] = 0.0

        if s[2] <= THETA_MIN or s[2] >= THETA_MAX:
            s[2] = np.clip(s[2], THETA_MIN, THETA_MAX)
            s[3] = 0.0

        theta_hist.append(s[2])
        u_hist.append(u)
        time_hist.append(t + Ts)

    if len(theta_hist) == 0:
        return np.zeros(1), np.ones(1) * np.inf, np.zeros(1)

    return np.array(time_hist), np.array(theta_hist), np.array(u_hist)

# --- Batch Simulation ---


def simulate_all_ics(controller, Ts=0.05, sim_T=FAIL_TIME, action_type="cont"):
    s0_list = get_test_initial_conditions()
    results = []
    for i, s0 in enumerate(s0_list):
        time, theta, u = simulate_controller(
            controller, s0, Ts, sim_T, action_type=action_type
        )
        metrics = compute_metrics(time, theta, u)
        results.append({
            'ic_index': i,
            's0': s0,
            'time': time,
            'theta': theta,
            'u': u,
            'metrics': metrics,
        })
    return results

# --- Plotting ---


def plot_comparison(controller_results, save_dir=None):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    for name, (time, theta, u) in controller_results.items():
        ax1.plot(time, theta, label=name)
        ax2.plot(time, u, label=name)
    ax1.set_xlabel('Time [s]')
    ax1.set_ylabel('Theta [rad]')
    ax1.set_title('Ball Angle')
    ax1.legend()
    ax1.grid(True)
    ax2.set_xlabel('Time [s]')
    ax2.set_ylabel('u ∈ [-1, 1]')
    ax2.set_title('Control Effort')
    ax2.legend()
    ax2.grid(True)
    plt.tight_layout()
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        plt.savefig(os.path.join(save_dir, 'comparison.png'), dpi=150)
    plt.show()


def plot_batch_results(results, controller_name, save_dir=None):
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
    for i, res in enumerate(results):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        ax1.plot(res['time'], res['theta'])
        ax1.set_xlabel('Time [s]')
        ax1.set_ylabel('Theta [rad]')
        ax1.set_title(f'IC #{i}: {np.array2string(res["s0"], precision=3)}')
        ax1.grid(True)
        ax2.plot(res['time'], res['u'])
        ax2.set_xlabel('Time [s]')
        ax2.set_ylabel('u ∈ [-1, 1]')
        ax2.set_title(f'Settling: {res["metrics"]["settling_time"]:.2f}s')
        ax2.grid(True)
        plt.tight_layout()
        if save_dir:
            plt.savefig(os.path.join(
                save_dir, f'{controller_name}_ic{i}.png'), dpi=150)
            plt.close()
        else:
            plt.show()
