# Evaluation

Experiment runner, simulation/hardware evaluation, controller tuning, and the
analysis utilities backing every Table I cell and figure in the paper.

## Directory layout

```
evaluation/
+-- experiment/                 # Experiment harness used by eval.py / eval_sim.py
|   +-- config.py               #   ExperimentConfig (action type, trials, ICs, paths)
|   +-- initial_conditions.py   #   Stratified initial conditions (50 per controller)
|   +-- runner.py               #   Hardware trial runner
|   +-- runner_sim.py           #   Simulation trial runner
+-- scripts/
|   +-- eval.py                 # Hardware experiment CLI  (make eval-hw)
|   +-- eval_sim.py             # Simulation experiment CLI (make eval-sim)
|   +-- recalibrate_sensors.py  # Sensor recalibration utility (HW maintenance)
+-- tuning/                     # CMA-ES + MPPI tuning
|   +-- tune_controllers.py     #   Canonical classical-controller tuner (--compare, --controller)
|   +-- tune_mppi_final.py      #   Final MPPI refinement (combines best dims from sweeps)
|   +-- simulate.py             #   Shared simulation + metric primitives for tuning
|   +-- tuning/                 #   Live runtime tuned parameters: cont/ and mppi/
+-- utils/
|   +-- analyze_exp.py          # Wilson SR CIs, settling bootstrap CIs, Mann-Whitney + Bonferroni, Cohen's d
|   +-- load_controllers.py     # RL model discovery + classical param loading
|   +-- utils.py                # Shared helpers (imported by training/ scripts)
+-- models/                     # Trained checkpoints (Git LFS)
    +-- world_model/{v1,v12,sweep_*}/  # LSTM world-model checkpoints
    +-- cont/                   # Continuous-action RL .zip files - see note below
```

## Key entry points

### Simulation evaluation

```bash
cd evaluation
PYTHONPATH=. python scripts/eval_sim.py --exp 1 --action-type cont --trials 50
# or via the repo-root Makefile:
make eval-sim
```

`eval_sim.py --exp 1` regenerates the §IV sim prediction (`paper/data/exp1_sim/`).
The paper does not use eval_sim's `--exp 2` (noise) or `--exp 3` (frequency)
modes - those exist for development but are not cited.

### Hardware evaluation (requires the physical platform)

```bash
cd evaluation
PYTHONPATH=. python scripts/eval.py --exp 1 --action-type cont --trials 50
# or:
make eval-hw
```

### Controller tuning (CMA-ES)

```bash
cd evaluation/tuning
python tune_controllers.py --controller all          # tune classical controllers
python tune_controllers.py --compare --action-type cont  # load + compare tuned params
python tune_mppi_final.py                              # refine MPPI (post-sweep)
```

### Analysis (statistics for Table I + significance matrix)

```bash
cd paper
python scripts/analyze_exp.py --exp 1 \
  --files data/exp1_hardware/consolidated.json \
  --outdir /tmp/exp1_stats
```

`analyze_exp.py` rebuilds the Wilson score success-rate intervals and
settling-time bootstrap CIs, Bonferroni-corrected Mann-Whitney rank tests on
the successful-trial settling times (α over all 78 controller pairs), and
Cohen's *d*; its output is byte-identical to the released
`exp1_significance.csv`. The single source of truth for the Table I and
Section V statistics is [`../paper/scripts/compute_statistics.py`](../paper/scripts/compute_statistics.py)
(Wilson SR intervals, Mann-Whitney on settling, Fisher's exact on success
rate); `analyze_exp.py` reports the same statistics alongside compute-time
diagnostics.

## RL checkpoints

The continuous-action RL checkpoints (`evaluation/models/cont/*.zip`) for
PPO, SAC, TD3, TQC, and TRPO are required by `eval_sim.py` to reproduce
the RL controllers' simulation results (Table I RL rows, `rl_new4` ablation).
**See `evaluation/models/cont/README.md`** - these `.zip` files are committed
via Git LFS (`git lfs pull` to materialise). The sim benchmark loads them
automatically.

## Related

- Full reproduction pipeline: [`../reproducibility_guide.md`](../reproducibility_guide.md)
- Figure regeneration (canonical): `paper/ram/regenerate_all_figures.py`

## Offline RL (IQL) checkpoint

`evaluation/utils/load_controllers.py`'s `discover_offline_rl_models()`
discovers the d3rlpy-format IQL model at
`evaluation/models/offline_rl/iql_best_demos/model_280000.d3` (Git LFS).
`git lfs pull` to materialise; `eval_sim` loads it automatically.
