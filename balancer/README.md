# balancer

Core Python package for the ball-on-arc control system. Provides controllers, simulation environments, hardware interface, and utilities.

```bash
pip install -e .
```

---

## Package Structure

| Subpackage | Purpose |
|---|---|
| **controllers/** | PID, LQR, MPC, NMPC, SMC, RL, Offline RL, MPPI |
| **core/** | Nonlinear dynamics, linearization, parameters, state definition |
| **envs/** | Gymnasium environments (sim + real + world model) |
| **hardware/** | Serial comms, sensor processing, robot interface |
| **wrappers/** | Domain randomization, history stacking, curriculum, logging |
| **world_model/** | LSTM dynamics model + inference wrapper for MPPI / PPO-WM |
| **rendering/** | 2D PyGame visualization |
| **utils/** | Training callbacks, seeding, logging |

---

## Controllers

All controllers implement `BaseController`:

```python
class BaseController(ABC):
    def step(self, state: np.ndarray, cart_limit: float) -> float: ...
    def reset(self) -> None: ...
```

| Controller | Class | Key Args |
|---|---|---|
| PID | `PIDController` | `Kp, Ki, Kd, Ts` |
| LQR | `LQRController` | `Q, R` |
| MPC | `MPCController` | `Q, R, N` (horizon) |
| NMPC | `NMPCController` | `dyn, N, dt` |
| SMC | `SMCController` | `lam, k` |
| RL | `RLController` | `path_to_model, algo` |
| Offline RL | `OfflineRLController` | `path_to_model` |
| MPPI | `MPPIController` | `predictor, horizon, n_samples` |

### Example

```python
from balancer.controllers import PIDController, LQRController, RLController

# Classical
pid = PIDController(Kp=20.0, Ki=0.5, Kd=3.0, Ts=0.05)
lqr = LQRController(Q=np.diag([0, 0, 30, 5]), R=np.array([[0.1]]))

# RL (algorithm inferred from filename)
rl = RLController(path_to_model="path/to/ppo_model.zip", action_type="cont")

# All share the same interface
action = pid.step(state, cart_limit=0.77)
```

---

## Environments

| Environment | Description |
|---|---|
| `BalancerSim-v0` | Simulation, 200 steps/episode |
| `BalancerSim-v1` | Simulation, 600 steps/episode (30s @ 20 Hz) |
| `BalancerSim-v2` | Simulation, 1200 steps/episode (60s @ 20 Hz; 2× the 30 s FAIL_TIME used for hardware trials) |
| `BalancerReal-v0` | Hardware interface |

```python
import gymnasium as gym
import balancer  # registers envs

env = gym.make("BalancerSim-v1", reward="ball_gaussian_distance")
obs, info = env.reset()

for _ in range(600):
    action = env.action_space.sample()
    obs, reward, done, truncated, info = env.step(action)
    if done:
        obs, info = env.reset()
```

### Simulation Features

- S-curve motor model (jerk/accel/decel limits)
- First-order velocity lag (tau_v = 0.15s)
- Command delay (1 step)
- Ball sensor staleness (geometric distribution)
- Domain randomization (`fixed_param=False`)

---

## Wrappers

| Wrapper | Purpose |
|---|---|
| `HistoryWrapper` | Frame stacking for partial observability |
| `ParameterAugmentedObs` | Appends physics params for DR conditioning |
| `DynResetEpisode` | Curriculum: random IC sampling |
| `BalancerLogger` | Per-step diagnostic logging |

---

## Dynamics

Nonlinear Euler-Lagrange dynamics with Gaussian dip geometry.

**State:** `[cart_pos, cart_vel, ball_angle, ball_angular_vel]`

```python
from balancer.core.dynamics import dynamics_first_order, DEFAULT_PARAMS
from balancer.core.linear_dynamics import get_discrete_system

# Nonlinear dynamics
state_dot = dynamics_first_order(state, force, DEFAULT_PARAMS)

# Linearized (for LQR/MPC)
Ad, Bd = get_discrete_system(Ts=0.05)
```

**Key parameters** (see `core/params.py`):
- Arc radius: 2.101 m
- Cart mass: 0.351 kg
- Ball mass: 0.024 kg
- Dip depth: 2 mm (Gaussian profile)

---

## Hardware Interface

```python
from balancer.hardware.robot import Robot

robot = Robot(control_dt=0.05)  # 20 Hz
obs, info = robot.reset()
for _ in range(1000):
    obs, reward, done, truncated, info = robot.step(action)
robot.close()
```

Components: `Robot` (Gym env) > `SystemState` (thread-safe) > Serial readers (IPC + distance) > Sensor/action utils.

---

## Tests

```bash
cd balancer
python -m tests.test_dynamics  # linearization, controllability, discretization
python -m tests.test_sim       # smoke test
```
