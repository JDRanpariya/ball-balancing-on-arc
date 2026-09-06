# iql_offline_rl: IQL Offline RL Hardware Evaluation Data

Implicit Q-Learning (IQL) trained on ~333k real transitions
(post-sensor-recalibration, May 20+ 2026). Implemented via d3rlpy 2.8.1.

## Model

- **Algorithm:** IQL (Implicit Q-Learning, Kostrikov et al. 2022)
- **Checkpoint:** model_280000 (280k gradient steps)
- **Architecture:** MLP [256, 256], ReLU activations (d3rlpy default)
- **Training data:** ~333k real transitions (1,415 trials) from
  post-sensor-recalibration hardware evaluations (May 20+ 2026).
  Mixed dataset: 83% successful, 17% failed episodes from all
  controllers (MPPI, NMPC, RL variants, MPC, PD, LQR, SMC).
  Dataset file: arcball_post_recalib_flat.h5
  Build script: data/build_flat_dataset.py --date-filter post_recalib

> Note: some raw eval configs under `evaluation/results/real/` describe
> earlier IQL *v2* variants (e.g. `expectile=0.9`). Those runs contributed
> trials to the training pool but are **distinct experiments** from the
> deployed checkpoint `model_280000`, which uses **expectile 0.7**
> (`data/verify_iql_disjoint.py` confirms `model_280000` is absent from the
> source controllers). The `expectile=0.9` in those configs is historical
> and intentionally left unchanged.
- **Key hyperparameters:**
  - Expectile: 0.7
  - Weight temperature: 3.0
  - Batch size: 256
  - Discount: 0.99
  - Actor/critic LR: 3e-4
  - n_critics: 2
  - Soft target tau: 0.005
  - Reward: balanced + dip_weight=1.0
  - Observation scaler: StandardObservationScaler
  - Action scaler: MinMaxActionScaler

## Hardware Evaluation Protocol

- 20 Hz control rate (Ts=0.05s)
- Settling band: 0.01 rad
- Settling duration: 1.0s continuous
- Run time: 30s per trial
- Stratified initial conditions (same protocol as Experiment 1)

## Files

| File | Trials | Description |
|------|--------|-------------|
| iql_best_demos_280k_50t.json | 50 | Full 50-trial evaluation |
| iql_best_demos_280k_10t.json | 10 | 10-trial screening run |
| *_config.yaml | (config) | Evaluation configuration |

## Results Summary

### 50-trial evaluation

| Metric | Value |
|--------|-------|
| Success Rate | 92% (46/50) |
| Mean settling | 10.62s |
| Median settling | 9.80s |
| Std settling | 7.24s |
| IQR | 11.14s |
| p90 | 20.40s |
| Max settling | 25.65s |
| RMS control effort | 0.410 |

### SR by difficulty (50 trials)

| Difficulty | SR |
|-----------|----|
| Easy | 80% (8/10) |
| Medium | 100% (16/16) |
| Hard | 86% (12/14) |
| Extreme opposite | 100% (5/5) |
| Extreme same | 100% (5/5) |

### 10-trial screening

| Metric | Value |
|--------|-------|
| Success Rate | 100% (10/10) |
| Mean settling | 6.90s |
| Median settling | 5.88s |

## Failure Analysis

4 failures in 50 trials:
- 2× easy (cart_target=±0.100m): ball near center, insufficient initial corrective action
- 2× hard (cart_target=0.650m): wall-proximate starts

Notably, extreme-same trials (100%) succeed: the policy handles
wall starts well when the ball is displaced far from center.
Failures on easy trials suggest the policy's conservative nature
(offline RL) produces insufficient action when the ball is barely
displaced.

## Key Finding

IQL achieves 92% SR on 50 stratified trials as an **offline RL**
policy trained on only 333k real transitions with **zero simulation** -
the first purely data-driven offline RL result on this hardware platform.

## Source

- Training: training/offline_rl/d3rlpy_logs/IQL_20260528150516/
- Model: evaluation/models/offline_rl/ (OfflineRLController_model_280000)
- Raw eval: evaluation/results/real/iql_best_demos/nominal_ball/iql_new_calib_20260528_155023/

## Paper Reference

Sections: sec:implementation (Offline RL subsection), sec:results (Table 1)
Appendix: appendix:rl_training (Offline RL subsection)
