# Ball-on-Arc Real Hardware Demonstrations

Real-world trajectories from a motorized arc balancing system, collected using
multiple controller families on physical hardware.

## System

A steel ball rolls on a curved arc (radius 2.1 m) actuated by a linear motor.
Two ToF distance sensors measure ball position at 20 Hz. The control task is to
balance the ball at the arc center from various starting positions.

| Parameter          | Value               |
| ------------------ | ------------------- |
| Arc radius         | 2.101 m             |
| Ball mass          | 24 g                |
| Ball limit         | +/-0.081 rad        |
| Cart limit         | +/-0.777 m          |
| Max cart velocity  | 0.9 m/s             |
| Control frequency  | 20 Hz (Ts = 0.05 s) |
| Settling band      | 0.01 rad            |
| Settling hold time | 1.0 s               |
| Trial timeout      | 30.0 s              |

## Dataset Statistics

| Metric                     | Value         |
| -------------------------- | ------------- |
| Total episodes             | 3,862         |
| Successful episodes        | 2,620 (67.8%) |
| Failed episodes            | 1,242 (32.2%) |
| Total timesteps            | ~1.23 M       |
| Total transitions (s,a,s') | ~1.23 M       |
| File size                  | ~70 MB        |

### Controller families

| Family | Episodes | Description                        |
| ------ | -------- | ---------------------------------- |
| RL     | 1,908    | PPO, TD3, SAC, TQC, TRPO policies  |
| NMPC   | 490      | Nonlinear model predictive control |
| MPC    | 362      | Linear MPC variants                |
| PID    | 410      | PID with various tunings           |
| LQR    | 260      | Linear quadratic regulator         |
| SMC    | 240      | Sliding mode control               |
| MPPI   | 192      | Model predictive path integral     |

## State and Action Spaces

**State** (4-dimensional):

| Index | Name        | Unit    | Typical range   |
| ----- | ----------- | ------- | --------------- |
| 0     | cart_pos    | meters  | [-0.78, 0.78]   |
| 1     | cart_vel    | m/s     | [-0.93, 0.93]   |
| 2     | ball_angle  | radians | [-0.083, 0.081] |
| 3     | ball_angvel | rad/s   | [-0.5, 0.5]     |

**Action** (1-dimensional):

- Cart velocity command in m/s, range [-0.9, 0.9]

## HDF5 Schema

```
arcball_real_demonstrations.h5
|
|-- episodes/
|   |-- 0/
|   |   |-- observations     float32 [T, 4]   # state at each timestep
|   |   |-- actions          float32 [T]       # velocity command
|   |   |-- timestamps       float32 [T]       # seconds from episode start
|   |-- 1/
|   |   |-- ...
|   |-- N/
|
|-- flat/                                      # Concatenated for ML pipelines
|   |-- observations         float32 [T_total, 4]
|   |-- next_observations    float32 [T_total, 4]
|   |-- actions              float32 [T_total]
|   |-- terminals            bool    [T_total]  # True at episode boundary
|   |-- episode_ids          int32   [T_total]  # maps step -> episode index
|   |-- episode_boundaries   int64   [N+1]      # cumulative start indices
|
|-- episode_metadata/                          # Per-episode arrays (length N)
|   |-- controller_name      str     [N]       # e.g. PIDController_PID_WO
|   |-- controller_family    str     [N]       # PID/LQR/MPC/NMPC/SMC/RL/MPPI
|   |-- episode_length       int32   [N]
|   |-- success              bool    [N]
|   |-- settling_time_s      float32 [N]       # NaN if failed
|   |-- difficulty           str     [N]       # easy/medium/hard/extreme
|   |-- initial_cart_pos_m   float32 [N]
|   |-- ise_theta            float32 [N]       # integral squared angle error
|   |-- ise_u                float32 [N]       # integral squared effort
|   |-- duration_s           float32 [N]
|   |-- source_file          str     [N]       # origin JSON path
|
|-- system/                                    # Physical constants (attrs)
|   |-- arc_radius_m = 2.101
|   |-- ball_mass_kg = 0.024
|   |-- ball_limit_rad = 0.081
|   |-- cart_limit_m = 0.7765
|   |-- max_cart_vel_ms = 0.9
|   |-- control_freq_hz = 20.0
|   |-- Ts_s = 0.05
|   |-- settling_band_rad = 0.01
|   |-- settling_duration_s = 1.0
|   |-- fail_time_s = 30.0
|
|-- (root attrs)
    |-- description, version, n_episodes, state_labels, ...
```

## Quick Start

### Load a single episode

```python
import h5py
f = h5py.File("arcball_real_demonstrations.h5", "r")
obs = f["episodes/42/observations"][:]   # shape (T, 4)
act = f["episodes/42/actions"][:]         # shape (T,)
ts  = f["episodes/42/timestamps"][:]     # shape (T,)
```

### Get all successful expert episodes (e.g. MPC)

```python
import numpy as np
meta = f["episode_metadata"]
families = meta["controller_family"][:]
success = meta["success"][:]
mask = (families == b"MPC") & success
expert_ids = np.where(mask)[0]
for eid in expert_ids:
    obs = f[f"episodes/{eid}/observations"][:]
    # ... use for imitation learning
```

### Flat arrays for offline RL (d3rlpy / CORL)

```python
obs      = f["flat/observations"][:]       # (1122854, 4)
next_obs = f["flat/next_observations"][:] # (1122854, 4)
actions  = f["flat/actions"][:]            # (1122854,)
terms    = f["flat/terminals"][:]          # (1122854,) bool

# Compute your own reward from states:
ball_angle = obs[:, 2]
rewards = np.exp(-100 * ball_angle**2)  # example Gaussian reward
```

### System identification (fit dynamics model)

```python
# Transition pairs: s_t, a_t -> s_{t+1}
obs      = f["flat/observations"][:]
next_obs = f["flat/next_observations"][:]
actions  = f["flat/actions"][:]
# Fit: next_obs = f(obs, actions)
```

### Filter by quality

```python
# Top-tier episodes: settled in under 5 seconds
settling = meta["settling_time_s"][:]
fast_mask = settling < 5.0
fast_ids = np.where(fast_mask)[0]
print(f"{len(fast_ids)} fast-settling expert episodes")
```

## Rewards

No reward is stored. The task goal is to bring ball_angle (state[2]) to zero.
Users should compute rewards appropriate for their method. Examples:

- **Sparse:** +1 if |ball_angle| < 0.01, else 0
- **Dense Gaussian:** exp(-k * ball_angle^2)
- **Shaped:** position + velocity penalty

## Appending New Data

After running new hardware trials (results saved as evaluation JSONs):

```bash
cd data/
python build_real_demonstrations.py
```

The build script is idempotent: it scans all
`evaluation/results/real/**/data.json` and regenerates the
full dataset. No manual bookkeeping needed.

Flags:

- `--dry-run` Preview stats without writing
- `--success-only` Only include successful episodes

## Provenance

Each episode's `episode_metadata/source_file` field traces it back
to the original evaluation JSON. The JSON schema is documented in
`evaluation/RESULT_FORMAT.md`.

## Differences from arcball_cont_1M_*.h5 (random exploration datasets)

|                    | This dataset                 | arcball_cont_1M_*.h5     |
| ------------------ | ---------------------------- | ------------------------ |
| Behavior policy    | Trained controllers          | Random uniform           |
| Structure          | Episodes with boundaries     | Continuous stream        |
| State coverage     | Concentrated near balance    | Broad (ball often far)   |
| Action correlation | State-dependent              | Independent of state     |
| Best for           | IL, offline RL, benchmarking | Dynamics modeling, sysid |

## License

This dataset is released under the Creative Commons Attribution 4.0
International License (CC BY 4.0); see the repository's root `LICENSE`.
