# Paper Data Directory

Canonical experimental data backing all tables and figures in:
"A Ball-on-Arc Benchmark: Classical and Learning-Based Control on
Realistic Deployment Hardware"

## Directory Map -> Paper Reference

| Directory | Paper Reference | Content |
|-----------|----------------|---------|
| exp1_hardware/ | Table II (results), Figs. 3-4 (boxplot, ranking) | 50-trial HW results (all 13 controllers) |
| exp1_hardware/sources/ | Table II | Per-controller raw JSON files |
| exp1_hardware/ppo_wm_v1.json | Appendix VI (World Model RL) | PPO world-model v1 (50 trials) |
| exp1_hardware/ppo_wm_v1_ft.json | Appendix XIII (Fine-Tuning) | PPO WM v1 fine-tuned on hardware |
| exp1_sim/ | Appendix X (Simulation Validation) | Simulation predictions (first-order motor) |
| exp1_sim/lqr_basic_pidlike_sim/ | Appendix XI (excluded variants) | Sim run of the deployed PD-equivalent LQR gain without override (42/50 = 84%) |
| rl_ablation/ | Appendix V (RL Training) | Blind-zone, multi-algo, checkpoint sweeps |
| rl_new4/ | Appendix V (RL Training) | SAC/TD3/TQC/TRPO (1M/3M, base+DR) |
| rl_5m_ablation/ | Appendix V (RL Training) | Extended 5M-step training ablation |
| ppo_wm_training/ | Appendix VI (World Model RL) | PPO-WM training logs (v1, v12, CPU/CUDA bench) |
| iql_offline_rl/ | §VI-D, Appendix V | IQL offline RL (~333k demo transitions; checkpoint at 280k training steps) |
| sim_validation/ | Appendix X (Sim Validation) | `.npz` validation results for all controllers |
| world_model/ | Appendix VI (World Model RL) | PPO-on-WM sim evaluation results |
| world_model_models/ | Appendix VI (World Model RL) | Trained PPO-WM checkpoints (`.zip`) |
| wm_data_sweep/ | Appendix VI (World Model RL) | Data-sweep: WM accuracy vs. dataset size |
| wm_hardware_results/ | Appendix VI (World Model RL) | Per-data-size HW evaluation results |
| tuned_params/ | Appendix IV (Classical Tuning) | CMA-ES / grid-search tuning artifacts |

## Data Format

All JSON evaluation files share a common schema. Top-level keys = controller
names. Each controller maps to a list of trial dicts with fields:
- `trial_id`, `success`, `settling_time`, `max_theta`, `rms_effort`
- `trajectory` (list of timestep dicts with `t`, `x`, `xdot`, `theta`, `thetadot`, `action`)

## Dataset (for training)

Small datasets live in the repo at `../../data/dataset/`. Large datasets are
archived (see repo README for archive access).

**In-repo:**
- `arcball_cont_1M_calib196_148.h5` - 1M continuous transitions (calibrated, main campaign)
- `arcball_cont_1M_calib196_160.h5` - 1M continuous transitions (calibrated, main campaign)
- `arcball_cont_1M_pre_neg.h5` - 1M continuous transitions (pre-negative-reward collection)
- `arcball_real_demonstrations/arcball_post_recalib_flat.h5` - post-recalibration demos (~6.5 MB)

**Git LFS** (requires `git lfs pull`):
- None currently

**Previously archived** (available on request / external storage):
- `arcball_cont_2_3M_calib196_148.h5` - 2.3M continuous transitions (full calibration-era collection)
- `arcball_cont_3M.h5`, `arcball_discrete_3M.h5` - early experimental datasets (pre-negative-reward)
- `arcball_cont_1M_pre_vfix.h5`, `arcball_discrete_3M_pre_vfix.h5` - before velocity-estimation fix
- `arcball_cont_10k_v2.h5`, `arcball_discrete_10k_v2.h5` - small/debug
- `arcball_real_demonstrations_flat.h5` - expert demo HDF5 for offline RL (~26 MB)
- `arcball_all_demos_flat.h5`, `arcball_all_demos_plus_random_flat.h5` - flattened demonstrations

## Known Issues

1. `tuned_params/` is the canonical directory; if a `"Tuned Params/"` directory
   exists elsewhere, it's a legacy name with a space - use quotes in scripts.
2. Two `\setupimg` definitions exist in `main.tex` (`experimental_setup.jpeg`
   and `system_setup.jpg`); the second overrides the first. Only
   `system_setup.jpg` is used in the compiled PDF.
3. The per-controller source files in `exp1_hardware/sources/` are the raw
   experiment records; `consolidated.json` is the merged superset used by
   figure-generation scripts.
4. Two PPO (BZ+DR) hardware runs exist and are named by servo-model regime:
   `sources/ppo_bz_dr_first_order.json` is the deployed 96% run at the
   first-order-lag checkpoint (the one in Table II / consolidated.json), and
   `sources/ppo_bz_dr_scurve.json` is the 98% S-curve-regime run used by the
   DR/blind-zone ablation S-curve.
