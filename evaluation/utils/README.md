# Evaluation Utilities

Controller evaluation, comparison, and result analysis tools.

---

## Contents

| File | Type | Purpose |
|---|---|---|
| `analyze_exp.py` | CLI | Full statistical analysis pipeline for Exp 1/2/3 JSON files |
| `utils.py` | CLI | JSON experiment file management (remove, clear, summarize, compare) |
| `load_controllers.py` | library | Shared factory used by `eval.py` / `eval_sim.py` |

---

## `analyze_exp.py` (Benchmark Analysis)

Full statistical pipeline matching the three paper experiments. Produces tables, figures (PDF + PNG), and summary CSVs.

### Arguments

| Argument | Default | Description |
|---|---|---|
| `--exp` | (required) | One of `1`, `2`, `3`, `all` |
| `--files` | `[]` | JSON files for the selected experiment (1 file for exp1, 4 for exp2/3) |
| `--noise_files` | `[]` | 4 JSON files for the noise axis when `--exp all` |
| `--freq_files` | `[]` | 4 JSON files for the frequency axis when `--exp all` |
| `--labels` | `[]` | Condition labels for exp2/3 (e.g. `"0%" "10%" "20%" "30%"`) |
| `--outdir` | `figures` | Output directory for figures and CSVs |
| `--traj` | off | Also plot representative theta trajectories (exp1 only) |
| `--n_boot` | `10000` | Bootstrap iterations for settling-time 95% CIs (success rate uses a Wilson score interval) |

### Experiment 1 (baseline performance)

```bash
python analyze_exp.py --exp 1 --files ../results/real/exp1.json --traj
```

Produces:
- Paper table: median settling time with 95% bootstrap CI, success rate with Wilson score interval, RMS control effort
- Distribution stats (median, IQR, p5/p95)
- Pairwise significance tests (Mann-Whitney rank test on successful-trial settling times, Bonferroni correction, Cohen's d)
- Boxplot, CI bar chart, significance heatmap
- Theta trajectory plots (with `--traj`)
- Compute-time analysis

### Experiment 2 (noise robustness)

```bash
python analyze_exp.py --exp 2 \
    --files ../results/real/noise_0.json ../results/real/noise_10.json \
            ../results/real/noise_20.json ../results/real/noise_30.json \
    --labels "0%" "10%" "20%" "30%"
```

Produces settling-time-vs-noise curves, success-rate curves, per-condition boxplots, degradation bar chart.

### Experiment 3 (frequency robustness)

```bash
python analyze_exp.py --exp 3 \
    --files freq_5hz.json freq_10hz.json freq_20hz.json freq_50hz.json \
    --labels "5 Hz" "10 Hz" "20 Hz" "50 Hz"
```

### All experiments combined

```bash
python analyze_exp.py --exp all \
    --files baseline.json \
    --noise_files n0.json n10.json n20.json n30.json \
    --freq_files f5.json f10.json f20.json f50.json \
    --outdir figures/
```

Writes a combined summary across all three axes plus an overall controller ranking.

---

## `utils.py` (Experiment File Management)

CLI for cleaning, merging, and summarising experiment JSONs.

### Subcommands

| Subcommand | Description |
|---|---|
| `remove_below` | Drop trials with `settling_time` below a threshold |
| `remove_by_index` | Drop trials by index range for a specific controller |
| `clear_controller` | Drop all trials for a specific controller |
| `summarize` | Print summary table and plots for one file |
| `compare` | Compare mean settling time across multiple files |
| `compare_noise` | Compare noise experiments (files matching `*noise_*.json` in CWD) |

```bash
# Remove trivial trials
python utils.py remove_below experiment.json --threshold 1.0

# Drop indices 75-100 of NMPCController_NMPC_dip
python utils.py remove_by_index experiment.json NMPCController_NMPC_dip 75 100

# Wipe a controller entirely (e.g. re-run needed)
python utils.py clear_controller experiment.json SMCController_SMC_dip

# Single-file summary
python utils.py summarize experiment.json

# Cross-file comparison
python utils.py compare exp1.json exp2.json exp3.json

# Noise experiment sweep (discovers files automatically)
python utils.py compare_noise
```

---

## `load_controllers.py` (Shared Controller Factory)

Library module (no CLI). Builds classical and RL controllers with consistent parameter loading across every benchmark script. Used by `eval.py` and `eval_sim.py`; new scripts should import from here rather than constructing controllers directly.

### Public API

| Function | Purpose |
|---|---|
| `build_controller(name, action_type, Ts, tuning_root, **kw)` | Build one controller by name |
| `build_benchmark_controllers(action_type, ...)` | Build the full default classical + RL stack |
| `build_controllers(...)` | Lower-level variant accepting explicit controller lists |
| `discover_rl_models(model_dir, action_type)` | Scan a models directory and return {algo: path} |
| `print_loaded_params(tuning_root)` | Pretty-print tuned parameters for all classical controllers |
| `find_params_file(controller, tuning_root)` | Locate `best_params.json` for one controller |
| `load_params(controller, tuning_root)` | Load those params as a dict |

### Constants

| Name | Contents |
|---|---|
| `CLASSICAL_CONTROLLERS` | `("pid", "lqr", "smc_wo", "mpc_wo", "nmpc")` - the 5 paper Table 1 classical controllers |
| `DISCRETE_RL_ALGOS` | Discrete-action SB3 / SB3-contrib algorithms |
| `CONTINUOUS_RL_ALGOS` | Continuous-action SB3 / SB3-contrib algorithms |

### Minimal example

```python
from utils.load_controllers import build_benchmark_controllers

controllers = build_benchmark_controllers(
    action_type="discrete",
    Ts=0.05,
    tuning_root="../tuning/tuning/discrete",
    model_dir="../models",
    skip_rl=False,
)
for name, ctrl in controllers.items():
    print(name, type(ctrl).__name__)
```

