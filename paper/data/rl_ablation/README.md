# RL Ablation Study - Paper Data

This folder contains the raw data and scripts needed to reproduce all
RL-related figures and tables in Appendix V (Reinforcement Learning Training
Details). Table and figure numbers below are the supplementary PDF's.

## Files

| File | Description | Used in |
|---|---|---|
| `multi_algo_sim_results.json` | 10 RL models × 50 sim trials (s-curve motor) | Appendix V, S-curve development family |
| `ppo_checkpoint_sweep.json` | PPO variants × 6 checkpoints × 20 trials | Figure 4 |
| `hw_validation.json` | 3 PPO variants × 10 hardware trials | Table 7 |
| `blind_zone_ablation.json` | 4 PPO blind zone variants × 50 sim trials | Table 12 |
| `blind_zone_hw_validation.json` | 2 blind zone variants × 10 hw trials | Table 13 |
| `multi_algo_hw_screening.json` | 4 off-policy DR variants × 10 hw trials | Appendix V, raw screening logs; no shipped table |
| `gen_ppo_ablation_fig.py` | Script to regenerate Figure 4 | Figure 4 |
| `gen_ablation_summary_fig.py` | Script to regenerate Figure 5 | Figure 5 |

## Reproducing

```bash
# Regenerate checkpoint figure
cd paper/data/rl_ablation
python gen_ppo_ablation_fig.py
# Output: paper/figures/ppo_ablation_checkpoints.png

# Regenerate summary ablation figure
python gen_ablation_summary_fig.py
# Output: paper/figures/ppo_ablation_summary.png

# Re-run full sim evaluation (requires trained models)
cd evaluation
make eval-sim \
    --models models/cont/ppo_ablation/ppo_balanced_best.zip \
            models/cont/ppo_balanced_dr_3M.zip \
            models/cont/ppo_blind_base.zip \
            models/cont/ppo_blind_dr.zip
```

## Evaluation Configuration

- **Sim trials:** 50, stratified ICs, seed=42
- **HW trials:** 10, stratified ICs
- **Settling criterion:** |θ| < 0.01 rad for ≥ 1.0s
- **Episode timeout:** 30s
- **Motor model (eval):** s-curve (jerk=50, accel=15, decel=15, τ_v=0.15, delay=1)
- **Integrator:** RK4, dt=0.05s (20 Hz)

## Training Configuration

- **Reward:** balanced (composite: position + velocity + effort + jerk + boundary)
- **Observation:** [x, ẋ, θ, θ̇] (4-dim)
- **DR variants:** fixed_param=False samples from DR_RANGES in params.py
- **Motor model (training):** s-curve (same params as eval)
- **Ball sensor staleness:** geometric(p) with mean=2 steps
