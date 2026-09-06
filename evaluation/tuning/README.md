# Controller Tuning - CMA-ES

Gradient-free tuning of the five classical controllers (PID, LQR, MPC, NMPC,
SMC) using [CMA-ES](https://cma-es.github.io). Each run evaluates candidate
gains on a small `NonLinearDynamics` rollout set and writes
`best_params.json` + `summary.json` + diagnostic plots per
controller.

---

## Pipeline

```
DEFAULT_PARAMS (balancer.core.params)
         -> NonLinearDynamics (motor_model=first_order, tau=0.05 [control period]; velocity lag tau_v=0.15)
         -> simulate.simulate_controller()   <- 5 tuning ICs
         -> scalar cost                      <- weighted sum of
                                                 settling time, ISE(theta),
                                                 ISE(theta_dot), ISE(u),
                                                 overshoot, boundary
                                                 violations, jerk, energy
         -> CMA-ES loop (popsize=32, maxfevals=980)
         -> best_params.json + summary.json
         -> validation on full 14-IC grid -> per-IC PNGs + convergence.png
```

Shared simulation utilities live in [`simulate.py`](simulate.py):

- `simulate_controller(controller, s0, Ts, sim_T)` runs one rollout,
  returns trajectories + metrics.
- `get_tuning_initial_conditions()` - 5 ICs that cover dip,
  ±edge, and cart-corner recovery. Used during optimisation.
- `get_test_initial_conditions()` - 14 ICs used for final
  validation and the paper tables.

---

## CLI

```bash
cd evaluation/tuning

# Tune one controller (default: --action-type cont)
python tune_controllers.py --controller pid
python tune_controllers.py --controller lqr
python tune_controllers.py --controller mpc
python tune_controllers.py --controller nmpc
python tune_controllers.py --controller smc

# Tune all five at once
python tune_controllers.py --controller all

# Compare classical tunings side-by-side on the 14-IC grid
python tune_controllers.py --compare --action-type cont

# Include a trained RL model in the comparison
python tune_controllers.py --compare \
    --rl-model ../models/cont/ppo_cont_balanced.zip

# Debugging: single worker
python tune_controllers.py --controller pid --jobs 1
```

Defaults (from `argparse`):

| Flag | Default | Notes |
|---|---|---|
| `--controller` | `all` | `pid / lqr / mpc / smc / nmpc / all` |
| `--Ts` | 0.05 | Control period (s) - must match the plant |
| `--action-type` | `cont` | **Only `cont` is supported.** See note below. |
| `--output-dir` | `tuning/cont/<ctrl>/` | Per-controller subfolder |
| `--jobs` | all CPU cores | Parallel CMA-ES population |
| `--compare` | off | Re-simulate saved tunings on the 14-IC grid |
| `--rl-model` | - | Optional SB3 .zip to include in `--compare` |

### Why cont-only?

CMA-ES is a **gradient-free** method, but the cost surface for discrete
(bang-bang) commands is piecewise-constant and chattering: tiny parameter
changes either change no switches (zero gradient in the rank-based CMA-ES
objective) or flip many switches at once (huge cost jumps). The population
converges on artefacts rather than real design improvements.

The continuous plant is the same linearised velocity model all classical
controllers are designed on. Tuning there, then deploying to the discrete
plant via `continuous_to_discrete()`, reproduces the deployed behaviour
and gives CMA-ES a smooth-enough surface to converge in under
1 000 evaluations.

`discrete` / `both` modes from the legacy implementation have been removed
from the CLI; only the `cont` mode is shipped.

---

## Output Layout

```
evaluation/tuning/tuning/
+-- cont/                       # current canonical runs
|   +-- pid/
|   |   +-- best_params.json     # the tuned gains
|   |   +-- summary.json         # averages + per-IC metrics
|   |   +-- convergence.png      # cost vs CMA-ES generation
|   |   +-- pid_ic0.png … pid_ic13.png
|   +-- lqr/
|   +-- mpc/
|   +-- nmpc/
|   +-- smc/
|
+-- {pid,lqr,mpc,nmpc,smc}/    # legacy flat layout from earlier runs
|                               # kept for diff-against-paper plots
|
+-- archive/
    +-- cont/                   # earlier cont runs
    +-- discrete/               # legacy discrete-mode tunings
    +-- both/                   # legacy averaged-cost tunings
    +-- comparison/             # --compare snapshots
    +-- manual/                 # hand-tuned baselines
```

### Cleanup note

`evaluation/tuning/tuning/` is double-nested - the tuner writes its
results into a subdirectory named `tuning` **inside** the
`tuning/` module. This is a historical wart: the output directory and
the Python module share a name. The double nesting is preserved because every
existing result pointer (paper plots, READMEs that predate this rewrite,
cached figures under `paper/ram/figures/`) references
`tuning/tuning/cont/...` and moving it would quietly break those.

If the directory is relocated, update:

- `tune_controllers.py` - `_DEFAULT_OUTDIR` constant.
- `evaluation/utils/load_controllers.py` - `TUNED_DIR` const.
- Any paper scripts that cite the old path.

---

## `best_params.json` schema

```json
{
    "Kp":    17.999,
    "Ki":     0.0012,
    "Kd":     0.3768,
    "Ts":     0.05,
    "tuned_on": "cont"
}
```

Keys per controller:

| Controller | Keys |
|---|---|
| PID | `Kp`, `Ki`, `Kd`, `Ts`, `tuned_on` |
| LQR | `Q_diag` (4), `R`, `K` (4), `Ts`, `tau_v`, `plant`, `tuned_on` |
| MPC | `Q_diag` (4), `R`, `N`, `Ts`, `tau_v`, `plant`, `tuned_on` |
| NMPC | `Q_diag` (4), `R`, `N`, `Ts`, `tuned_on` |
| SMC | `lam`, `k`, `Ts`, `tuned_on` |

`summary.json` contains the mean metrics over all tuning ICs plus a
per-IC breakdown (settling time, overshoot, boundary violations, energy,
jerk, ISE(θ), ISE(θ̇), ISE(u)). `load_controllers.py` in
`evaluation/utils/` reads this format directly.

---

## CMA-ES configuration

Defaults (set in each `tune_<ctrl>` function):

| Param | Value | Notes |
|---|---:|---|
| Population size | 32 | Held fixed across controllers |
| Max evaluations | 980 | ≈ 30 generations |
| Initial sigma | 0.5 | Relative to parameter bounds |
| Parallel workers | all cores | Override with `--jobs` |

Per-controller bounds + `x0` live in the corresponding
`tune_<ctrl>` function. Adjust there if you add a new parameter (e.g.
an integral limit for PID).

---

## See also

- [`../README.md`](../README.md) - benchmarking the tuned controllers on hardware and in sim.
- [`../utils/README.md`](../utils/README.md) - `load_controllers.py` consumes `best_params.json`.
- Result JSONs produced by the evaluation scripts use a different schema from `best_params.json` / `summary.json` above; see `evaluation/README.md`.
- [`../../balancer/README.md`](../../balancer/README.md) - controller implementations and design rationale.
