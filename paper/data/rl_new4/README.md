# rl_new4 - New 4-Algorithm Hardware Evaluation Data

All models trained with: **first-order motor | balanced reward | seed 1**.
Protocol: stratified ICs, 20 Hz, same schedule as Exp 1.

## Directory Structure

``                                                                                                                                                                                                                                                             `                      {run}_{ckpt}_{date}_data.json`

## hw_screening_dr/ - DR Variants

50-trial best checkpoints (used in paper Table 1):

| File pattern | Algo | Ckpt | 50t HW SR | 50t Mean |
|-------------|------|------|-----------|----------|
| sac_fo_1M_dr_best_50t_* | SAC | best_model | 82% | 10.10s |
| td3_fo_1M_dr_150k_50t_* | TD3 | 0.15M | 92% | 11.45s |
| tqc_fo_1M_dr_best_50t_* | TQC | best_model | 86% | 9.77s |
| trpo_fo_3M_dr_best_50t_* | TRPO | best_model | 90% | 8.09s |

10-trial screening (4 checkpoints per algo):
- SAC: 0.20M, 0.45M, best, final
- TD3: 0.15M, 0.55M, best, final
- TQC: 0.15M, 0.90M, best, final
- TRPO: 2.10M, 2.35M, best, final

## hw_screening_base/ - Base (no DR) Variants

10-trial screening results (4 checkpoints per algo):

| Algo | Best ckpt | HW SR | HW Mean |
|------|----------|-------|--------|
| SAC base | 0.40M | 100% | 7.67s |
| TD3 base | best | 90% | 7.33s |
| TQC base | 0.85M | 90% | 14.18s |
| TRPO base | final | 100% | 6.19s |

Base vs DR finding:
- SAC: base > DR (100% vs 80%) - entropy+DR amplifies BZ variability
- TD3: base = DR (90% both) - deterministic policy unaffected by DR
- TQC: DR > base (90% vs 80%) - distributional critic benefits from DR
- TRPO: base = DR (100% both) - trust-region constraint dominates

## sim_evals/ - Checkpoint Sim Evaluations

50-IC fixed-param sim eval for all checkpoints (20 per run + best/final):
- sac_fo_1M_dr_new_checkpoint_eval.json
- td3_fo_1M_dr_new_checkpoint_eval.json
- tqc_fo_1M_dr_new_checkpoint_eval.json
- trpo_fo_3M_dr_new_checkpoint_eval.json
- sac_fo_1M_base_checkpoint_eval.json
- td3_fo_1M_base_new_checkpoint_eval.json
- tqc_fo_1M_base_new_checkpoint_eval.json
- trpo_fo_3M_base_new_checkpoint_eval.json

## Source Models (evaluation/models/cont/)

| File | Run |
|------|-----|
| trpo_fo_3M_dr_best.zip  | training/runs/trpo_fo_3M_dr_new/best_model/ |
| tqc_fo_1M_dr_best.zip   | training/runs/tqc_fo_1M_dr_new/best_model/ |
| td3_fo_1M_dr_150k.zip   | training/runs/td3_fo_1M_dr_new/checkpoints/checkpoint_step_150000.zip |
| sac_fo_1M_dr_best.zip   | training/runs/sac_fo_1M_dr_new/best_model/ |

## Paper Reference

Sections: sec:sac_analysis, sec:td3_analysis, sec:tqc_analysis, sec:trpo_analysis
Tables: tab:sac_checkpoints, tab:td3_checkpoints, tab:tqc_checkpoints, tab:trpo_checkpoints,
        tab:new_rl_50t, tab:sac_base_vs_dr, tab:td3_base_vs_dr, tab:tqc_base_vs_dr, tab:trpo_base_vs_dr
In: paper/ram/sections/07_results.tex

## Coverage

- 50-trial HW evals cover the DR best checkpoints (Table 1). Base (no-DR)
  variants have 10-trial screening only (SAC 0.40M, TD3 best, TQC 0.85M,
  TRPO final).
- TQC (DR) 50t is reported in Table 1 at 86%.
