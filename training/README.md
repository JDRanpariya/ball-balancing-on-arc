# Training

RL training pipeline for the ball-on-arc balancing task. Uses [Stable-Baselines3](https://github.com/DLR-RM/stable-baselines3) for RL algorithms, [Hydra](https://hydra.cc/) for configuration management, and [Weights & Biases](https://wandb.ai/) for experiment tracking.

The `balancer` package (in `../balancer/`) provides the Gym environments, wrappers, and utilities consumed by this module.

---

## Structure

````
training/
|-- configs/
|   |-- train.yaml                    # Top-level training config
|   |-- algo/                          # Algorithm configs
|   |   |-- ppo.yaml, ppo_ft.yaml      #   PPO (base + fine-tune)
|   |   |-- sac.yaml                   #   SAC
|   |   |-- td3.yaml, td3_ft.yaml      #   TD3 (base + fine-tune)
|   |   |-- tqc.yaml, tqc_ft.yaml      #   TQC (base + fine-tune)
|   |   |-- trpo.yaml                  #   TRPO
|   |   |-- crossq.yaml                #   CrossQ
|   |   |-- a2c.yaml                   #   A2C
|   |   |-- ars.yaml                   #   ARS
|   |   |-- dqn.yaml                   #   DQN
|   |   |-- qrdqn.yaml                 #   QR-DQN
|   |   +-- recurrent_ppo.yaml         #   Recurrent PPO
|   |   +-- tqc_scratch.yaml           #   TQC from scratch (real hardware)
|   |-- env/                           # Environment configs
|   |   |-- sim.yaml                   #   Simulation (default)
|   |   |-- real.yaml                  #   Real hardware
|   |   |-- world_model.yaml           #   World model env config
|   |   |-- dyn/                       #   Dynamics model configs
|   |   |   |-- tiny.yaml              #     S-curve motor + delays (ablations)
|   |   |   |-- first_order.yaml       #     First-order velocity lag (default)
|   |   |   +-- original.yaml          #     Force-based (legacy)
|   |   +-- robot/                     #   Robot-specific overrides
|   +-- wrappers/                      # Wrapper stack configs
|       |-- sim_wrappers.yaml          #   Default (TimeLimit only)
|       |-- sim_wrappers_hist4.yaml    #   With HistoryWrapper(steps=4)
|       |-- real_wrappers.yaml         #   Real hardware
|       |-- real_ft_param_wrappers.yaml#   Real fine-tune with param augmentation
|       +-- wm_wrappers.yaml           #   World model wrappers
|-- scripts/
|   |-- train.py                       # Single-run training (Hydra entrypoint)
|   |-- train_all_cont.py              # Batch training (multi-algo, DR, ablations)
|   |-- train_world_model.py           # World model training
|   `-- sweep.yaml                     # W&B grid sweep definition
|-- offline_rl/                        # d3rlpy offline RL
|   |-- offline_train_arcball.py       #   Discrete CQL/BCQ
|   |-- offline_train_cont.py          #   Continuous CQL/IQL/TD3+BC/BCQ
|   |-- eval_offline_sim.py            #   Sim evaluation for offline models
|   |-- env.yaml                       #   Conda env spec (Python 3.13, d3rlpy 2.8.1)
|   |-- discrete_cql_model.pt          #   Trained discrete CQL model
|   +-- README.md
|-- runs/                              # Hydra output (auto-generated, gitignored)
|   |-- <timestamp>/                   #   Per-run: checkpoints, models, tb logs
|   +-- cont_batch_*/                  #   Batch training output
````

---

## Installation

Install the `balancer` package before running training:

```bash
pip install -e ../balancer/
```

---

## Quick Start

### Single Run (via Hydra)

```bash
cd training
python scripts/train.py \
  algo=ppo \
  n_envs=16 \
  total_timesteps=3000000
```

### Batch Training (Recommended for Benchmarks)

`train_all_cont.py` is the **single entry point** for all batch training -- algorithm comparison, ablations, and DR variants.

```bash
cd training

# Train all algorithms with balanced reward (full benchmark zoo)
python scripts/train_all_cont.py --rewards balanced --timesteps 3000000

# Train specific algos
python scripts/train_all_cont.py --algos ppo sac td3 --rewards balanced --timesteps 3000000

# Train with domain randomisation
python scripts/train_all_cont.py --algos ppo --rewards balanced --dr --timesteps 3000000

# Train both base + DR in one batch (for ablation studies)
python scripts/train_all_cont.py --algos ppo sac td3 tqc --rewards balanced --both-dr --timesteps 3000000

# Custom tag for output model name
python scripts/train_all_cont.py --algos ppo --rewards balanced --dr --tag ppo_dr --timesteps 3000000

# Quick test (dry run)
python scripts/train_all_cont.py --algos ppo --rewards balanced --both-dr --dry-run
```

**Output:** Models are copied to `evaluation/models/cont/<tag>.zip` and a summary JSON is written to `evaluation/models/cont/training_summary.json`.

---

## Ablation Studies

To reproduce the paper's ablation study (reward x DR x algorithm):

```bash
# 1. PPO reward ablation
python scripts/train_all_cont.py --algos ppo --rewards balanced ball_gaussian_distance --timesteps 3000000

# 2. PPO with DR
python scripts/train_all_cont.py --algos ppo --rewards balanced --dr --tag ppo_balanced_dr --timesteps 3000000

# 3. Multi-algorithm comparison (base + DR)
python scripts/train_all_cont.py --algos ppo sac td3 tqc trpo --rewards balanced --both-dr --timesteps 3000000
```

Checkpoints are saved at configured intervals. For checkpoint-sweep evaluation, set:
```bash
python scripts/train_all_cont.py --algos ppo --rewards balanced --timesteps 3000000 \
  --eval-freq 50000
```

---

## Domain Randomisation

DR is controlled by the `fixed_param` flag on `BalancerSim`:
- `fixed_param=True` (default): nominal dynamics from `DR_MEANS`
- `fixed_param=False`: randomises physics each episode from `DR_RANGES`

In batch training, use `--dr` or `--both-dr`. In single runs:
```bash
python scripts/train.py +env.fixed_param=False
```

---

## Configuration

The config system is composed from four groups, set in `configs/train.yaml`:

| Group | Default | Description |
|---|---|---|
| `env` | `sim.yaml` | Environment instantiation and reward function |
| `wrappers` | `sim_wrappers.yaml` | Observation/action wrapper stack |
| `algo` | `ppo.yaml` | Algorithm hyperparameters |
| *(root)* | `train.yaml` | Training loop settings |

Switch configs by passing the group override, e.g. `algo=sac`, `env=real`, `env=sim`.

### Key Parameters

| Parameter | Default | Description |
|---|---|---|
| `total_timesteps` | `3000000` | Total environment steps |
| `n_envs` | `16` | Parallel environments (1 for off-policy/real) |
| `seed` | `1` | Random seed |
| `model_path` | `null` | Resume from checkpoint |
| `evaluation.eval_freq` | `5000` | Steps between evaluations |
| `checkpoint.save_freq` | `50000` | Steps between checkpoint saves |
| `checkpoint.max_checkpoints` | `50` | Max checkpoints on disk |

### Available Algorithms

| Algorithm | Type | Envs | Config |
|---|---|---|---|
| PPO | On-policy | 16 | `algo/ppo.yaml` |
| TRPO | On-policy | 16 | `algo/trpo.yaml` |
| A2C | On-policy | 16 | `algo/a2c.yaml` |
| ARS | On-policy | 16 | `algo/ars.yaml` |
| SAC | Off-policy | 1 | `algo/sac.yaml` |
| TD3 | Off-policy | 1 | `algo/td3.yaml` |
| TQC | Off-policy | 1 | `algo/tqc.yaml` |
| CrossQ | Off-policy | 1 | `algo/crossq.yaml` |
| DQN | Off-policy | 1 | `algo/dqn.yaml` |
| QR-DQN | Off-policy | 1 | `algo/qrdqn.yaml` |
| Recurrent PPO | On-policy | 16 | `algo/recurrent_ppo.yaml` |

Fine-tuning configs (`ppo_ft.yaml`, `td3_ft.yaml`, `tqc_ft.yaml`) use reduced learning rates for real-hardware fine-tuning.

### Available Rewards

| Reward | Max/step | Max/episode | Description |
|---|---|---|---|
| `balanced` | ~0.6 | ~360 | Composite: position + velocity + effort + jerk + boundary |
| `ball_gaussian_distance` | ~1.0 | ~600 | Gaussian centred on balance point |

---

## Training Environment Details

The simulation env (`BalancerSim-v1`, 600 steps = 30s @ 20 Hz) uses:
- **Motor model:** s-curve with jerk_limit=50, accel_limit=15, decel_limit=15
- **Velocity lag:** tau_v = 0.15s first-order lag
- **Command delay:** 1 step (50ms)
- **Ball sensor staleness:** geometric distribution, mean = 2 steps (simulates the ~12 Hz hardware ToF rate)
- **Observation noise:** nominal std per `NOMINAL_OBS_NOISE` when `fixed_param=True`

This matches the hardware dynamics model used in evaluation.

---

## Fine-Tuning on Real Hardware

```bash
python scripts/train.py \
  env=real \
  wrappers=real_ft_param_wrappers \
  n_envs=1 \
  total_timesteps=100000 \
  model_path=<checkpoint.zip>
```

---

## Outputs

Each run produces a timestamped directory under `runs/`:

```
runs/<timestamp>/
|-- checkpoints/    # Periodic policy checkpoints (.zip)
|-- best_model/     # EvalCallback best model
|-- models/         # Final saved model (.zip)
|-- eval_logs/      # evaluations.npz (reward curves)
|-- videos/         # Rollout video
+-- tb/             # TensorBoard logs (synced to W&B)
```

Batch training additionally writes:
- `evaluation/models/cont/<tag>.zip` -- best model per run
- `evaluation/models/cont/training_summary.json` -- timing + milestones

---

## Offline RL

Offline RL pipeline using d3rlpy (included in the main `balancer` conda env). See [`offline_rl/README.md`](offline_rl/README.md).

| Script | Action space | Algorithms |
|---|---|---|
| `offline_train_arcball.py` | Discrete | DiscreteCQL, DiscreteBCQ |
| `offline_train_cont.py` | Continuous | CQL, IQL, TD3+BC, BCQ |
| `eval_offline_sim.py` | Both | Sim evaluation for trained models |

```bash
cd training/offline_rl

# Continuous CQL (default)
python offline_train_cont.py --data-path ../../data/dataset/arcball_cont_1M_calib196_160.h5

# IQL on GPU with W&B
python offline_train_cont.py --algo iql --n-steps 500000 --gpu --wandb

# Discrete CQL (legacy; dataset not in repo)
python offline_train_arcball.py --n-actions 3 --data-path ../../data/dataset/arcball_cont_1M_calib196_160.h5
```

---

## Hyperparameter Sweeps

```bash
wandb sweep scripts/sweep.yaml
wandb agent <sweep_id>
```

