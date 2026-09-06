# Data Collection and Dataset Tools

Tools for collecting transition data from the physical ball-on-arc-on-cart
system and managing the HDF5 datasets used for world-model and offline-RL
training. The world-model **training pipeline** itself lives in the
`balancer` package (`balancer/balancer/world_model/`); this directory
contains the dataset tools and the committed HDF5 datasets.

______________________________________________________________________

## Directory Structure

```
data/
|-- collect_data.py               # Real-time data collection from hardware
|-- build_real_demonstrations.py  # Build full episodic HDF5 from eval JSONs
|-- build_flat_dataset.py         # Build flat 11-col HDF5 from eval JSONs (IQL input)
|-- dataset/                      # HDF5 datasets and metadata
|   |-- arcball_cont_1M_pre_neg.h5              # 1M pre-calibration, raw vel
|   |-- arcball_cont_1M_calib196_148.h5         # 1M calibrated (log entry #4: LEFT=193, RIGHT=148)
|   |-- arcball_cont_1M_calib196_160.h5         # 1M recalibrated (log entry #6: LEFT=189, RIGHT=158)
|   |-- arcball_real_demonstrations/            # Post-recalibration IQL + demo datasets
|   |   |-- arcball_real_demonstrations.h5       # 3.9K episodes, 1.2M trans.
|   |   |-- arcball_post_recalib_flat.h5        # 333K flat transitions (IQL input)
|   |   |-- arcball_demo_70k_balanced.h5        # 70K balanced demo (world-model sweep)
|   |   `-- README.md
|   `-- sample_data.png                    # dataset preview image
|-- README.md                     # This file
```

All datasets are committed to the repository (some via Git LFS;
run `git lfs pull` to materialize LFS pointers).

______________________________________________________________________

## Dataset Formats

There are two HDF5 layouts used across datasets:

**15-column format** (produced by `collect_data.py`): includes raw ToF readings
and pre-computed rewards.

```
[prev_state (4)] [prev_raw_tof (2)] [action (1)] [cur_state (4)] [cur_raw_tof (2)] [reward (1)] [dip_reward (1)]
```

| Col | Name                | Description                                                                          |
| --- | ------------------- | ------------------------------------------------------------------------------------ |
| 0   | `prev_cart_pos`     | Cart position (m)                                                                    |
| 1   | `prev_cart_vel`     | Cart velocity (m/s)                                                                  |
| 2   | `prev_ball_pos`     | Ball angle on arc (rad)                                                              |
| 3   | `prev_ball_vel`     | Ball angular velocity (rad/s)                                                        |
| 4   | `prev_raw_left_mm`  | Left sensor raw reading before step (mm) - saved for checking, not used for training |
| 5   | `prev_raw_right_mm` | Right sensor raw reading before step (mm) - same                                     |
| 6   | `action`            | Velocity command [-0.9, 0.9] (continuous) or 0/1/2 (discrete)                        |
| 7   | `cur_cart_pos`      | Cart position after step (m)                                                         |
| 8   | `cur_cart_vel`      | Cart velocity after step (m/s)                                                       |
| 9   | `cur_ball_pos`      | Ball angle after step (rad)                                                          |
| 10  | `cur_ball_vel`      | Ball angular velocity after step (rad/s)                                             |
| 11  | `cur_raw_left_mm`   | Left sensor raw reading after step (mm) - saved for checking, not used for training  |
| 12  | `cur_raw_right_mm`  | Right sensor raw reading after step (mm) - same                                      |
| 13  | `reward`            | Reward value (Gaussian on ball position + velocity penalty)                          |
| 14  | `dip_reward`        | 1 if ball is at center (IR beam broken), else 0                                      |

**11-column format** (produced by `build_flat_dataset.py`, used by IQL and
world-model training): drops the 4 raw sensor columns (not needed for
training). Column order: `prev_state(4)`, `action(1)`, `cur_state(4)`,
`reward(1)` (always 0), `dip_reward(1)`.

**Storage:** All files use HDF5 chunking + GZIP compression. All data is `float32`.

______________________________________________________________________

## Datasets

### Random-exploration (collected via `collect_data.py`)

<details>
<summary><b>arcball_cont_1M_calib196_148.h5</b> - 1,000,000 rows, 29 MB, 15-col</summary>

|                  |                                                        |
| ---------------- | ------------------------------------------------------ |
| **Format**       | 15-column                                              |
| **Calibration**  | Log entry #4, 2026-05-20 14:51 (LEFT=193, RIGHT=148) - bias ≈ −1.3 mrad |
| **Sampling**     | 20 Hz, negated velocity sign convention                |
| **Used by**      | Not the v12 training set - see `arcball_cont_1M_calib196_160.h5` below |
| **Source**       | First ~1.185M rows of the cleaned 2.26M collection campaign, collected under entry #4 (before recalibration to entry #6) |
| **Action range** | [-1.00, 1.00], diverse continuous                      |
| **State range**  | cart: ±0.78 m, ball: ±0.09 rad                         |

</details>

<details>
<summary><b>arcball_cont_1M_calib196_160.h5</b> - 1,000,000 rows, 28 MB, 15-col</summary>

|                  |                                                             |
| ---------------- | ----------------------------------------------------------- |
| **Format**       | 15-column                                                   |
| **Calibration**  | Log entry #6, 2026-05-22 (LEFT=189, RIGHT=158) - near-zero bias (+0.1 mrad) |
| **Sampling**     | 20 Hz                                                       |
| **Used by**      | MPPI world model (v12 checkpoint) - confirmed by matching `evaluation/models/world_model/v12/norm_stats.npz` normalization statistics (see Sensor Calibration section below) |
| **Source**       | Last ~1M rows of the cleaned 2.26M collection campaign, collected under entry #6 |
| **Action range** | [-1.00, 1.00], diverse continuous                           |

</details>

<details>
<summary><b>arcball_cont_1M_pre_neg.h5</b> - 999,999 rows, 27 MB, 11-col</summary>

|                  |                                                          |
| ---------------- | -------------------------------------------------------- |
| **Format**       | 11-column (no raw ToF, no reward)                        |
| **Calibration**  | Log entry #1 (LEFT=175, RIGHT=170) - bias ≈ +5-7 mrad    |
| **Sampling**     | 20 Hz, raw encoder velocity (different sign convention)  |
| **Used by**      | PPO-WM world model (v1 checkpoint)                       |
| **Source**       | Separate earlier collection campaign (pre-recalibration) |
| **Action range** | [-1.00, 1.00], diverse continuous                        |

This dataset uses different velocity
sign convention than the calib196_148/160 datasets. Both are needed to
reproduce the MPPI and PPO-WM checkpoints respectively (see supplementary G).

</details>

### Hardware demonstrations (built from evaluation JSONs)

<details>
<summary><b>arcball_real_demonstrations.h5</b> - 3,862 episodes, 80 MB, episodic</summary>

Episodic dataset of hardware trials across 7 controller families collected
after sensor recalibration. Each episode stores `actions` (1D), `observations`
(4-dim state), and `timestamps`. A flat representation is included with
episode boundaries, episode IDs, and terminal flags.

|                       |                                                                     |
| --------------------- | ------------------------------------------------------------------- |
| **Format**            | Episodic (`episodes/{id}/{actions,observations,timestamps}`) + flat |
| **Calibration**       | Post-recalibration                                                  |
| **Total transitions** | 1,225,345 (flat)                                                    |
| **Overall SR**        | 67.8%                                                               |
| **Build script**      | `build_real_demonstrations.py`                                      |

**Controller breakdown:**

| Family | Episodes | Success Rate |
| ------ | -------- | :----------: |
| RL     | 1,908    |    84.5%     |
| PID    | 410      |    75.9%     |
| NMPC   | 490      |    67.1%     |
| LQR    | 260      |    51.9%     |
| MPC    | 362      |    43.1%     |
| MPPI   | 192      |    22.4%     |
| SMC    | 240      |    13.8%     |

</details>

<details>
<summary><b>arcball_post_recalib_flat.h5</b> - 333,497 rows, 7.9 MB, 11-col</summary>

|                  |                                                                   |
| ---------------- | ----------------------------------------------------------------- |
| **Format**       | 11-column                                                         |
| **Calibration**  | Post-recalibration (LEFT≈189-196, RIGHT≈148-158)                  |
| **Used by**      | Deployed IQL controller (model_280000, 280k steps, expectile=0.7) |
| **Source**       | Filtered from post-recalibration evaluation JSONs (May 20+ 2026)  |
| **Build script** | `build_flat_dataset.py --date-filter post_recalib`                |
| **Composition**  | Mixed success + failure, all controller families                  |

This is the direct input to the offline RL (IQL) training pipeline.

</details>

<details>
<summary><b>arcball_demo_70k_balanced.h5</b> - 70,310 rows, 4.0 MB, 15-col</summary>

|                  |                                                         |
| ---------------- | ------------------------------------------------------- |
| **Format**       | 15-column                                               |
| **Calibration**  | Pre-calibration                                         |
| **Used by**      | Standalone side experiment (small-scale world-model sanity check) |
| **Action range** | [-1.00, 1.00], 17,461 unique values                     |
| **State range**  | cart: ±0.78 m, ball: ±0.09 rad                          |
| **Dip reward**   | 0% (random exploration, no controlled demos)            |

Random exploration data collected as a one-off side experiment, separate
from the official world-model data-sweep. It is *not* the pool used by
the data-sweep experiment: the sweep
([`wm_data_sweep.py`](../paper/ram/scripts/wm_data_sweep.py)) trains on
10k-1,000,000-row subsets of the random-exploration campaign (the same
2.26M-row collection `arcball_cont_1M_calib196_148.h5` and
`arcball_cont_1M_calib196_160.h5` are drawn from), not on this 70K file.

</details>

______________________________________________________________________

## Sensor Calibration

### Background

All VL53L0X ToF sensors have per-unit offsets. The ball-on-arc system
uses two sensors (left and right edge) to compute ball angle from raw
distance readings. The **center calibration** values (`LEFT_CENTER_MM`
and `RIGHT_CENTER_MM`) convert raw mm to ball angle - they represent
the raw reading each sensor gives when the ball is at physical center.

The physical center is determined by the IR beam-break sensor (Adafruit
#2167): when `beam_state = 1`, the ball is at the arc apex.

### Calibration Entries Over the Project Lifespan

Sensor readings drift over time (temperature, mounting, per-unit variance) -
there is no single "true" calibration. Each log entry below reflects the
sensor's empirical center reading at that point in time. The authoritative
historical record is
[`calibration_log.json`](../balancer/balancer/hardware/calibration_log.json)
(18 entries as of this writing); the **currently active** values always
live in `constants.py:SystemConstants` and should never be inferred from
this table alone.

| Log Entry     | LEFT_CENTER_MM | RIGHT_CENTER_MM | Date                    | Notes                                                                          |
| ------------- | -------------- | --------------- | ------------------------ | ------------------------------------------------------------------------------ |
| #1 (original) | 175            | 170             | 2026-01-01               | Initial factory / mechanical-drawing estimate; offset not yet characterized |
| #2            | 175            | 170             | 2026-05-15               | First empirical equilibrium check: ball reads LEFT=193, RIGHT=160 at true center, but `center_mm` left unchanged; +4.76 mrad residual absorbed via `SENSOR_OFFSET_RAD=0.007` instead |
| #3            | 196            | 144             | 2026-05-20 09:55         | Second empirical calibration; `center_mm` updated directly for the first time. **Source of the "calib196" filename token** - this is when the random-exploration campaign began |
| #4            | 193            | 148             | 2026-05-20 14:51         | Recalibration ~5h later (1k-sample median). Active for the first ~1.185M rows of the campaign -> shipped as `arcball_cont_1M_calib196_148.h5` |
| #5            | 190            | 159             | 2026-05-21               | Recalibration                                                                  |
| #6            | 189            | 158             | 2026-05-22               | "website-video-collection" entry. Active for the later/last rows of the campaign -> last ~1M shipped as `arcball_cont_1M_calib196_160.h5` |
| #7-#17        | various        | various         | 2026-05-29 to 2026-06-23 | Auto-recalibration cycles between RL hardware experiments; not tied to any single committed dataset (see `calibration_log.json` for the full list) |
| #18 (latest)  | 186            | 160             | 2026-06-30 16:41         | Most recent recalibration. **Current active values** - matches `constants.py:SystemConstants`, the single source of truth |

### Calibration History by Dataset

**Pre-calibration period** (`arcball_cont_1M_pre_neg.h5`, `arcball_demo_70k_balanced.h5`):
Collected under log entry #1 (LEFT=175, RIGHT=170; offset not yet
characterized). The ball angle is biased by approximately +5--7 mrad
(confirmed by analyzing `ball_theta` when `dip_reward = 1`; consistent
with entry #2's later empirical finding that the true equilibrium reading
at this same `center_mm` setting was LEFT=193, RIGHT=160).

**First calibration period** (`arcball_cont_1M_calib196_148.h5`):
Collected under entry #4 (LEFT=193, RIGHT=148). RIGHT=148 differs from the
value the sensor drifted to later in the campaign (158, entry #6) - this
is ordinary sensor drift over the collection window, not an error.
Resulting bias ≈ −1.3 mrad at center.

**Second calibration period** (`arcball_cont_1M_calib196_160.h5`):
Collected under entry #6 (LEFT=189, RIGHT=158). Resulting bias ≈ +0.1 mrad
at center - near-zero. This is the dataset the v12 world model was
actually trained on (see the Datasets section above): the released
`evaluation/models/world_model/v12/norm_stats.npz` normalization
statistics match this file's state distribution, not `calib196_148.h5`.

**Post-recalibration** (`arcball_real_demonstrations/`):
Spans hardware trials from 2026-05-20 onward (`build_flat_dataset.py`'s
`post_recalib` date filter), collected across many recalibration entries
(#3 through #18 and beyond) as the sensors were periodically re-zeroed
between hardware experiments - see `calibration_log.json` for the full
per-entry history. There is no single calibration value for this dataset:
each trial reflects whatever `constants.py` held at that trial's run
time. The values active as of the latest entry (#18, 2026-06-30) are
LEFT=186, RIGHT=160.

### Impact

1. **`ball_theta` in pre-calibration HDF5 files is biased by +5--7 mrad**
1. **`ball_theta` in `calib196_148` is biased by −1.3 mrad**
1. **`calib196_160` is near-zero bias (+0.1 mrad)**
1. **Post-recalibration files have small, per-trial-varying bias** (whichever
   `constants.py` values were active at each trial's run time - see
   `calibration_log.json`), not a single fixed bias value
1. The bias does NOT affect:
   - Success rate measurements (bias < settling band of ±10 mrad)
   - Relative controller comparisons (all controllers see same bias within a dataset)
   - World model training (model learns biased dynamics consistently)
1. The bias DOES affect:
   - Absolute settling time (policy targets θ=0, not true equilibrium)
   - Fine-tuning with dip reward (conflicting signals: balanced peaks at θ=0, dip peaks at θ≈+7 mrad)

### Sensor Noise

With ball stationary at physical center (5000 samples):

- Left sensor: 187--201 mm (range 14 mm, ±7 mm)
- Right sensor: 151--168 mm (range 17 mm, ±9 mm)
- Angular noise: ±3.3--4.3 mrad per reading

Noise is consistent with the 34.5 ms timing budget characterization
(σ ≈ 1.6--1.7 mm reported in the supplementary sensor-characterization
table, `paper/ram/supplementary/sections/A_sensors.tex` - this is in the
supplementary material, not the main paper's Table 1).

______________________________________________________________________

## Scripts

### `collect_data.py` - Real-Time Data Collection

Collects state-action-reward transitions from the physical system. Reads cart
state over serial (IPC) and ball position from dual ToF distance sensors,
applies random actions, and logs transitions to HDF5.

**Requires the physical hardware to be connected.**

```bash
# Default: 3M steps at 20 Hz
python collect_data.py

# Custom settings
python collect_data.py --steps 1000000 --output dataset/arcball_cont_1M.h5 --hz 20

# Change action type and sticky probability
python collect_data.py --action-type discrete --p-stick 0.6
```

______________________________________________________________________

### Dataset-build scripts

- `build_real_demonstrations.py` - builds the full episodic HDF5
  (`arcball_real_demonstrations.h5`) from evaluation JSONs. Idempotent:
  re-run after new trials to regenerate.
  ```bash
  python data/build_real_demonstrations.py
  python data/build_real_demonstrations.py --dry-run  # preview
  python data/build_real_demonstrations.py --success-only
  ```
- `build_flat_dataset.py` - builds a flat 11-column HDF5 from evaluation
  JSONs for offline RL (IQL) training. Supports date filtering and
  step capping.
  ```bash
  # Build the deployed IQL input:
  python data/build_flat_dataset.py
  # Cap transitions (data-sweep experiments):
  python data/build_flat_dataset.py --max-steps 250000
  # Include all dates (not just post-recalibration):
  python data/build_flat_dataset.py --date-filter all
  ```

**Removed (consolidated into `build_flat_dataset.py`):**
`build_post_recalib_dataset.py`, `build_new_calib_demos_flat.py`,
`convert_demos_to_flat.py`, `trim_dataset.py`.

______________________________________________________________________

## Collection Architecture

```
                    +------------------+
                    |   Main Thread    |
                    |  (control loop)  |
                    |  random actions  |
                    |  transition log  |
                    +--------+---------+
                             |
              +--------------+--------------+
              |              |              |
     +--------v---+  +------v------+  +----v--------+
     | IPC Reader |  | Dist Reader |  | HDF5 Saver  |
     |  (thread)  |  |  (thread)   |  |  (thread)   |
     | cart pos/  |  | ball angle/ |  | buffered    |
     | velocity   |  | velocity    |  | GZIP writes |
     +-----+------+  +------+------+  +----+--------+
           |                |              |
     [serial port]    [serial port]   [HDF5 file]
      /dev/ttyUSB0   /dev/distance     dataset/
```

**Thread-safe state sharing** via `SystemState` (lock-protected,
event-signaled).

**Sensor processing pipeline:**

1. Raw ToF readings (mm) from dual VL53L0X sensors
1. Convert to ball angle (rad) via `process_dual_sensors()`
1. Smooth position via `PositionSmoother` (moving average)
1. Estimate velocity via `VelocityEstimator` (finite difference + clipping)
1. Update shared `SystemState`

**Config persistence:** Each collection run saves its configuration as a
YAML sidecar file (same name as the HDF5, `.yaml` extension) and as HDF5
file attributes.

______________________________________________________________________

## World-Model Training

The world-model architecture (`LSTMWorldModel`, `WorldModelConfig`) and
inference wrapper (`WorldModelPredictor`) live in
[`balancer/balancer/world_model/`](../balancer/balancer/world_model/). The
training entry point and sweep driver are
[`paper/ram/scripts/wm_data_sweep.py`](../paper/ram/scripts/wm_data_sweep.py)
(it regenerates the committed `paper/data/wm_data_sweep/` results from the
v12 checkpoint). See `reproducibility_guide.md` for the full training
procedure.
