"""
analyze_exp.py  -  Ball-and-Arc Benchmark Analysis
====================================================
Usage
-----

# Experiment 1 - baseline (single JSON, 100 trials):

python analyze_exp.py --exp 1 --files exp1_baseline.json

# Experiment 2 - noise robustness (4 JSONs):

python analyze_exp.py --exp 2 \
    --files noise_0.json noise_10.json noise_20.json noise_30.json \
    --labels "0%" "10%" "20%" "30%"

# Experiment 3 - frequency robustness (4 JSONs):

python analyze_exp.py --exp 3 \
    --files freq_5hz.json freq_10hz.json freq_20hz.json freq_50hz.json \
    --labels "5 Hz" "10 Hz" "20 Hz" "50 Hz"

# All experiments together:

python analyze_exp.py --exp all \
    --files        exp1.json \
    --noise_files  noise_0.json noise_10.json noise_20.json noise_30.json \
    --freq_files   freq_5hz.json freq_10hz.json freq_20hz.json freq_50hz.json

Output
------
All figures are written as both .pdf and .png to --outdir (default: ./figures/).
A summary CSV is written for every table that appears in the paper.
Paper-ready strings (e.g. "8.90 [8.38, 9.42]") are printed to stdout.
"""

import os
import sys
from pathlib import Path

# Paper-reproducibility bootstrap: ensure the ``balancer`` package is importable
# when this script is run directly from ``paper/scripts/``.  The project root
# is three levels up from this file (paper/scripts/analyze_exp.py -> repo).
_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from balancer.hardware.constants import SYSTEM
from itertools import combinations
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from compute_statistics import wilson_ci  # 95% Wilson score interval (percent)
import matplotlib.ticker as mticker
import matplotlib.pyplot as plt
import argparse
import json
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")

# Default dataset backing Table I / Figs 4-6 of the paper.  When the caller
# does not pass ``--files``, we fall back to this 13-controller 50-trial file.
DEFAULT_EXP1_FILE = _REPO / "paper" / "data" / "exp1_hardware" / "consolidated.json"

warnings.filterwarnings("ignore", category=RuntimeWarning)

# -----------------------------------------------------------------------------

#  GLOBAL CONFIG  (mirrors eval.py values exactly)

# -----------------------------------------------------------------------------

VELOCITY_BAND = SYSTEM.VELOCITY_BAND
FAIL_TIME = SYSTEM.FAIL_TIME
SETTLING_BAND = SYSTEM.SETTLING_BAND
SETTLING_DURATION = SYSTEM.SETTLING_DURATION

# Design / reference frequencies for each controller (for Exp-3 delta column)

DESIGN_FREQ = {
    "PID":  20.0,
    "LQR":  20.0,
    "SMC":  20.0,
    "MPC":  20.0,
    "NMPC": 20.0,
    "RL":   100.0,   # deployed at 50 Hz (trained at 100 Hz sim)
}

CONTROLLER_ORDER = ["PID", "LQR", "SMC", "MPC", "NMPC", "RL"]

NAME_MAP = {
    # classical
    "PIDController_PID":                                    "PID",
    "LQRController_LQR":                           "LQR",
    "LQRController_LQR_custom":                           "LQR",
    "SMCController_SMC":                           "SMC",
    "MPCController_MPC":                           "MPC",
    "MPCController_MPC_Q0_0_11_0_R0.31_N30_vel":   "MPC",
    "MPCController_MPC_Q0_0_12_9_R0.00_N31_vel":   "MPC",
    "NMPCController_NMPC":                         "NMPC",
    # RL baselines discrete
    "RLController_ppo_ppo":                        "PPO",
    "RLController_a2c_a2c":                        "A2C",
    "RLController_trpo_trpo":                        "TRPO",
    "RLController_dqn_dqn":                        "DQN",
    "RLController_qrdqn_qrdqn":                        "QRDQN",
    # RL baselines cont
    "RLController_sac_sac":                        "SAC",
    "RLController_td3_td3":                        "TD3",
    "RLController_tqc_tqc":                        "TQC",
    "RLController_crossq_crossq_500k":             "CrossQ",
    # RL baselines
    "RLController_3M_eq3_gaussR_18N":                  "RL",
    "RLController_3M_eq3_dipR_18N":                    "RL",
    "RLController_3M_noH_noDR":                        "RL-3M-NoH-NoDR",
    "RLController_3M_noH_noDR_correct_dip_width_original_sim": "RL-dip-cor",
    "RLController_3M_noH_noDR_correct_old_eqn_sim":    "RL-old-eq",
    "RLController_3M_noH_noDR_mass_352":               "RL-m352",
    # RL fine-tuned
    "RLController_ft_50Hz_8_H_900k":                   "RL_ft_900k",
    "RLController_ft_50Hz_8_H_800k":                   "RL_ft_800k",
    "RLController_tilt_600k_ft":                       "RL_ft_600k",
    "RLController_tilt_1000k_ft":                      "RL_ft_1000k",
    # variants
    "RLController_10M_noH_noDR_2g_dip_reward":         "RL-NoH-NoDR-2gDip",
    "RLController_10M_withH_noDR":                     "RL-H-NoDR",
    "RLController_10M_noH_DR_2g_no_param_in_state":    "RL-NoH-DR-2g",
    "50Hz_gaussR_18_H_20_semi_euler":                  "RL-50Hz-H20-SE",
    "50Hz_gaussR_18_rk4":                              "RL-50Hz-rk4",
    "50Hz_gaussR_18_semi_euler":                       "RL-50Hz-SE",
}

# Colours - one per controller slot, consistent across all plots

_PAL = plt.cm.tab10.colors


# -----------------------------------------------------------------------------

#  I/O HELPERS

# -----------------------------------------------------------------------------

def _remap(data: dict) -> dict:
    renamed = {}
    for raw, trials in data["controllers"].items():
        renamed[NAME_MAP.get(raw, raw)] = trials
    out = dict(data)
    out["controllers"] = renamed
    return out


def load_json(path: str) -> dict:
    with open(path, "r") as f:
        return _remap(json.load(f))


def sort_controllers(names):
    def _k(n):
        try:
            return CONTROLLER_ORDER.index(n)
        except:
            return len(CONTROLLER_ORDER)
    return sorted(names, key=_k)


def _save(fig, path_no_ext: str):
    os.makedirs(os.path.dirname(path_no_ext) or ".", exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(f"{path_no_ext}.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> {path_no_ext}.pdf / .png")


def _ctrl_color(name: str) -> tuple:
    idx = CONTROLLER_ORDER.index(name) if name in CONTROLLER_ORDER \
        else hash(name) % len(_PAL)
    return _PAL[idx]


# -----------------------------------------------------------------------------

#  STATISTICS

# -----------------------------------------------------------------------------

def bootstrap_ci(data, n_boot: int = 10_000, ci: float = 0.95, seed: int = 0):
    rng = np.random.default_rng(seed)
    arr = np.asarray(data, float)
    if arr.size == 0:
        return np.nan, np.nan
    boots = np.array([np.mean(rng.choice(arr, size=arr.size, replace=True))
                      for _ in range(n_boot)])
    a = (1 - ci) / 2
    return float(np.percentile(boots, 100 * a)), \
        float(np.percentile(boots, 100 * (1 - a)))


def cohens_d(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    pooled = np.sqrt((np.var(a, ddof=1) + np.var(b, ddof=1)) / 2.0)
    return float((np.mean(a) - np.mean(b)) / pooled) if pooled > 0 else 0.0


def _pval_stars(p, alpha_corr):
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < alpha_corr:
        return "*"
    return "n.s."


# -----------------------------------------------------------------------------

#  TRIAL-LEVEL METRICS

# -----------------------------------------------------------------------------

SS_WINDOW = 5.0   # seconds of steady-state window at end of trial


def _trial_metrics(trial: dict) -> dict:
    st_raw = trial.get("settling_time")

    # -- Prefer raw_states for recomputation (unaffected by noise) --
    # Controller saw noisy_state, but settling/overshoot should be
    # evaluated on the TRUE ball angle, same as the online check.
    raw_states = trial.get("raw_states", [])
    states = raw_states if raw_states else trial.get("states", [])

    timestamps = trial.get("timestamps",  [])
    actions = trial.get("actions",     [])
    ct = trial.get("controller_time_ms", [])
    loop = trial.get("loop_period_ms",     [])
    ise_theta = trial.get("ise_theta")
    ise_vel = trial.get("ise_vel")
    ise_u = trial.get("ise_u")

    # -- Recompute settling time from states if available -----------
    if states and timestamps:
        arr = np.asarray(states, float)
        thetas = arr[:, 2]
        times = np.asarray(timestamps, float)
        st_recomputed = _settling_time_from_arrays(thetas, times)
        st_stat = st_recomputed
        success = st_recomputed < FAIL_TIME
    else:
        st_stat = st_raw if st_raw is not None else FAIL_TIME
        success = bool(trial.get("success", st_raw is not None))

    if ise_theta is None and states and timestamps:
        # Recompute post-hoc - works on old JSON files too
        arr = np.asarray(states, float)
        theta_arr = arr[:, 2]
        time_arr = np.asarray(timestamps, float)
        u_arr = np.asarray(
            actions, float) if actions else np.zeros(len(theta_arr))

        from experiment.runner_sim import _compute_ise
        ise_theta, ise_vel, ise_u = _compute_ise(time_arr, theta_arr, u_arr)

    # RMS of the clipped command actually sent (+/-0.9 m/s axis limit),
    # matching Table I and compute_statistics.py.
    rms_u = (float(np.sqrt(np.mean(np.clip(np.array(actions, float),
                                           -0.9, 0.9) ** 2)))
             if actions else np.nan)
    mean_ct = float(np.mean(ct)) if ct else np.nan
    mean_lp = float(np.mean(loop)) if loop else np.nan

    # -- Overshoot & boundary violations ---------------------------
    overshoot = np.nan
    boundary_violations = 0

    if states:
        arr = np.asarray(states, float)
        thetas = arr[:, 2]
        times = (np.asarray(timestamps, float)
                 if len(timestamps) == len(thetas)
                 else np.arange(len(thetas)) * 0.05)

        overshoot, boundary_violations = _overshoot_from_arrays(
            thetas, times)

    return dict(
        settling_time=st_stat,
        settling_raw=st_raw,
        success=success,
        overshoot=overshoot,
        boundary_violations=boundary_violations,
        rms_u=rms_u,
        ctrl_time_ms=mean_ct,
        loop_ms=mean_lp,
        ise_theta=float(ise_theta) if ise_theta is not None else np.nan,
        ise_vel=float(ise_vel) if ise_vel is not None else np.nan,
        ise_u=float(ise_u) if ise_u is not None else np.nan,
    )


def _overshoot_from_arrays(thetas, times, tol=SETTLING_BAND):
    """
    Max |θ| after first entering settling band + boundary violation count.
    Same definition as tuning/simulate.py.
    """
    theta_abs = np.abs(thetas)

    boundary_violations = int(np.sum(
        theta_abs >= (SYSTEM.BALL_LIMIT - 1e-4)
    ))

    in_band = theta_abs < tol
    first_entry = np.argmax(in_band)

    if not in_band[first_entry]:
        return float(np.nanmax(theta_abs)), boundary_violations

    post_entry = theta_abs[first_entry:]
    overshoot = float(np.nanmax(post_entry))

    return overshoot, boundary_violations


def _settling_time_from_arrays(thetas, times,
                               tol=SETTLING_BAND,
                               duration=SETTLING_DURATION):
    """Recompute settling time from logged state arrays - same as tuning."""
    theta_abs = np.abs(thetas)
    in_band = theta_abs < tol
    enter_time = None
    for i in range(len(times)):
        if in_band[i]:
            if enter_time is None:
                enter_time = times[i]
            if times[i] - enter_time >= duration:
                return float(enter_time)
        else:
            enter_time = None
    return FAIL_TIME


# -----------------------------------------------------------------------------

#  CONTROLLER-LEVEL SUMMARY

# -----------------------------------------------------------------------------

def controller_summary(name: str, trials: list, n_boot: int = 10_000) -> dict:
    mets = [_trial_metrics(t) for t in trials]
    n = len(mets)

    st_all = [m["settling_time"] for m in mets]
    st_succ = [m["settling_raw"]
               for m in mets if m["settling_raw"] is not None]
    n_succ = sum(m["success"] for m in mets)

    rms_us = [m["rms_u"] for m in mets if not np.isnan(m["rms_u"])]
    cts = [m["ctrl_time_ms"] for m in mets if not np.isnan(m["ctrl_time_ms"])]
    loops = [m["loop_ms"] for m in mets if not np.isnan(m["loop_ms"])]

    # --- metrics kept only for diagnostic / discussion (not paper table) ---
    overs = [m["overshoot"] for m in mets if not np.isnan(m["overshoot"])]
    rmses = []

    ise_thetas = [m["ise_theta"] for m in mets if not np.isnan(m["ise_theta"])]
    ise_vels = [m["ise_vel"] for m in mets if not np.isnan(m["ise_vel"])]
    ise_us = [m["ise_u"] for m in mets if not np.isnan(m["ise_u"])]

    # Settling statistics over successful trials only, matching the paper
    # table and compute_statistics.py (st_all, which counts failures at
    # FAIL_TIME, is kept for diagnostics only).
    arr = np.asarray(st_succ if st_succ else [np.nan], float)
    ci_lo, ci_hi = bootstrap_ci(st_succ if st_succ else [np.nan],
                                n_boot=n_boot)
    # Success rate uses a Wilson score interval (correct at the 100% boundary,
    # where a bootstrap degenerates to [100, 100]); see compute_statistics.py.
    sr_lo, sr_hi = wilson_ci(n_succ, n)

    return dict(
        controller=name,
        n_trials=n,
        # -- PRIMARY METRICS (paper table) ---------------------------------
        mean_settling=float(np.mean(arr)),
        ci_lo=ci_lo,
        ci_hi=ci_hi,
        success_rate=100.0 * n_succ / n,
        sr_ci_lo=sr_lo,
        sr_ci_hi=sr_hi,
        n_success=n_succ,
        mean_rms_u=float(np.mean(rms_us)) if rms_us else np.nan,
        # -- DISTRIBUTION METRICS (boxplot / results text, not table) ------
        median_settling=float(np.median(arr)),
        std_settling=float(np.std(arr, ddof=1)),
        iqr_settling=float(np.percentile(arr, 75) - np.percentile(arr, 25)),
        p5=float(np.percentile(arr, 5)),
        p95=float(np.percentile(arr, 95)),
        # -- DIAGNOSTIC ONLY (similar across controllers, do not table) ----
        mean_overshoot=float(np.mean(overs)) if overs else np.nan,
        mean_rmse_ss=float(np.mean(rmses)) if rmses else np.nan,
        mean_ctrl_ms=float(np.mean(cts)) if cts else np.nan,
        mean_loop_ms=float(np.mean(loops)) if loops else np.nan,
        # -- ISE Metrics -------------------------------------------------------
        mean_ise_theta=float(np.mean(ise_thetas)) if ise_thetas else np.nan,
        mean_ise_vel=float(np.mean(ise_vels)) if ise_vels else np.nan,
        mean_ise_u=float(np.mean(ise_us)) if ise_us else np.nan,
        # -- raw lists for stats --------------------------------------------
        settling_all=st_all,
        settling_success=st_succ,
    )


# -----------------------------------------------------------------------------

#  FORMAT HELPERS

# -----------------------------------------------------------------------------

def _fmt(s):
    return f"{s['mean_settling']:.2f} [{s['ci_lo']:.2f}, {s['ci_hi']:.2f}]"


def _fmt_sr(s):
    return f"{s['success_rate']:.0f} [{s['sr_ci_lo']:.0f}, {s['sr_ci_hi']:.0f}]"


# -----------------------------------------------------------------------------

#  STATISTICAL SIGNIFICANCE TABLE

# -----------------------------------------------------------------------------

def compute_stat_tests(summaries, alpha=0.05):
    names = [s["controller"] for s in summaries]
    n = len(names)
    n_comp = n * (n - 1) // 2
    alpha_c = alpha / max(n_comp, 1)

    pmat = pd.DataFrame(np.nan, index=names, columns=names)
    dmat = pd.DataFrame(0.0,   index=names, columns=names)
    ksmat = pd.DataFrame(np.nan, index=names, columns=names)

    for i, j in combinations(range(n), 2):
        # Success-only settling times, tested with a rank test (matches
        # Table I and compute_statistics.py); a mean-based t-test is
        # inappropriate for these right-skewed distributions.
        a, b = summaries[i]["settling_success"], summaries[j]["settling_success"]
        if len(a) < 2 or len(b) < 2:
            continue
        _, p_t = stats.mannwhitneyu(a, b, alternative="two-sided")
        _, p_ks = stats.ks_2samp(a, b)
        d = cohens_d(a, b)
        pmat.iloc[i, j] = pmat.iloc[j, i] = p_t
        ksmat.iloc[i, j] = ksmat.iloc[j, i] = p_ks
        dmat.iloc[i, j] = dmat.iloc[j, i] = abs(d)

    return pmat, dmat, ksmat, alpha_c


def build_sig_table(summaries, alpha=0.05):
    names = [s["controller"] for s in summaries]
    pmat, dmat, _, a = compute_stat_tests(summaries, alpha)
    rows = []
    for i, ni in enumerate(names):
        row = {"": ni}
        for j, nj in enumerate(names):
            if i == j:
                row[nj] = "-"
            elif i < j:
                row[nj] = _pval_stars(pmat.iloc[i, j], a)
            else:
                row[nj] = ""
        rows.append(row)
    df = pd.DataFrame(rows).set_index("")
    print(
        f"\nBonferroni α = {a:.5f}  (α_orig=0.05 / {len(names)*(len(names)-1)//2} comparisons)")
    return df, dmat


# -----------------------------------------------------------------------------

#  SHARED PLOT PRIMITIVES

# -----------------------------------------------------------------------------

def _boxplot_ax(ax, data_list, labels,
                title="", ylabel="Settling Time (s)",
                show_fail_line=True):
    bp = ax.boxplot(
        data_list,
        tick_labels=labels,
        patch_artist=True,
        showmeans=True,
        meanprops=dict(marker="^", markerfacecolor="green",
                       markeredgecolor="green", markersize=8),
        medianprops=dict(color="black", linewidth=2),
        flierprops=dict(marker="o", markerfacecolor="grey",
                        markeredgecolor="grey", markersize=4, alpha=0.5),
    )
    for patch, lbl in zip(bp["boxes"], labels):
        patch.set_facecolor(_ctrl_color(lbl))
        patch.set_alpha(0.70)
    if show_fail_line:
        ax.axhline(FAIL_TIME, color="red", linestyle="--",
                   linewidth=0.8, alpha=0.5,
                   label=f"Max trial ({FAIL_TIME:.0f} s)")
        ax.legend(fontsize=8)
    ax.set_title(title, fontsize=11)
    ax.set_ylabel(ylabel, fontsize=10)
    ax.tick_params(axis="x", rotation=25)
    ax.grid(axis="y", alpha=0.3)


def _plot_ise_bars(summaries, outdir):
    """Bar chart showing ISE decomposition per controller."""
    labels = [s["controller"] for s in summaries]
    ise_thetas = [s["mean_ise_theta"] for s in summaries]
    ise_vels = [s["mean_ise_vel"] for s in summaries]
    ise_us = [s["mean_ise_u"] for s in summaries]
    x = np.arange(len(labels))
    width = 0.25

    fig, ax = plt.subplots(figsize=(9, 4))
    ax.bar(x - width, ise_thetas, width, label="ISE-θ (position)",  alpha=0.85)
    ax.bar(x,         ise_vels,   width,
           label="ISE-θ̇ (vel@center)", alpha=0.85)
    ax.bar(x + width, ise_us,     width,
           label="ISE-u (effort)",     alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=25, ha="right")
    ax.set_ylabel("ISE (lower = better)")
    ax.set_title("ISE Metric Decomposition - Tuning Objective")
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    _save(fig, os.path.join(outdir, "exp1_ise_bars"))

# -----------------------------------------------------------------------------

#  EXPERIMENT 1  -  replace the table-building block

# -----------------------------------------------------------------------------


def run_exp1(data: dict, outdir: str, n_boot: int = 10_000):
    print("\n" + "="*60)
    print("EXPERIMENT 1 - BASELINE PERFORMANCE")
    print("="*60)

    ctrlnames = sort_controllers(list(data["controllers"].keys()))
    summaries = [controller_summary(n, data["controllers"][n], n_boot)
                 for n in ctrlnames]

    # -- Paper table output -------------------------------------------------
    print("\n[PAPER TABLE - Settling Time / Success Rate / Control Effort]")
    print(f"  {'Controller':<10}  {'Settling (s)':<25}  "
          f"{'SR (%)':<15}  {'RMS u':<8}")
    print("  " + "-"*65)
    for s in summaries:
        ru = f"{s['mean_rms_u']:.3f}" if not np.isnan(s["mean_rms_u"]) else "-"
        print(f"  {s['controller']:<10}  {_fmt(s):<25}  "
              f"{_fmt_sr(s):<15}  {ru}")

    print(f"\n[ISE METRICS - tuning objective, lower = better]")
    print(f"  {'Controller':<10}  {'ISE-θ':>10}  {'ISE-θ̇':>10}  {'ISE-u':>8}")
    print("  " + "-"*45)
    for s in summaries:
        ise_t = f"{s['mean_ise_theta']:.5f}" if not np.isnan(
            s['mean_ise_theta']) else "-"
        ise_v = f"{s['mean_ise_vel']:.5f}" if not np.isnan(
            s['mean_ise_vel']) else "-"
        ise_u = f"{s['mean_ise_u']:.5f}" if not np.isnan(
            s['mean_ise_u']) else "-"
        print(f"  {s['controller']:<10}  {ise_t:>10}  {ise_v:>10}  {ise_u:>8}")

    # -- Distribution stats for results text / boxplot captions ------------
    print("\n[DISTRIBUTION - for results text and boxplot caption]")
    print(f"  {'Controller':<10}  {'Median':>7}  {'IQR':>7}  "
          f"{'p5':>6}  {'p95':>6}")
    print("  " + "-"*48)
    for s in summaries:
        print(f"  {s['controller']:<10}  "
              f"{s['median_settling']:>7.2f}  "
              f"{s['iqr_settling']:>7.2f}  "
              f"{s['p5']:>6.2f}  "
              f"{s['p95']:>6.2f}")

    # -- Diagnostic (not in paper) -----------------------------------------
    print("\n[DIAGNOSTIC - overshoot & SS-RMSE: similar across controllers, "
          "NOT in paper table]")
    print(f"  {'Controller':<10}  {'Overshoot':>10}  {'SS-RMSE':>10}  "
          f"{'Note'}")
    print("  " + "-"*60)
    overs = [s["mean_overshoot"]
             for s in summaries if not np.isnan(s["mean_overshoot"])]
    rmses = [s["mean_rmse_ss"]
             for s in summaries if not np.isnan(s["mean_rmse_ss"])]
    o_range = max(overs) - min(overs) if overs else 0
    r_range = max(rmses) - min(rmses) if rmses else 0
    for s in summaries:
        ov = f"{s['mean_overshoot']:.4f}" if not np.isnan(
            s["mean_overshoot"]) else "-"
        rm = f"{s['mean_rmse_ss']:.4f}" if not np.isnan(
            s["mean_rmse_ss"]) else "-"
        print(f"  {s['controller']:<10}  {ov:>10}  {rm:>10}")
    print(f"  Range: overshoot={o_range:.4f} rad  "
          f"SS-RMSE={r_range:.4f} rad  "
          f"(both < ToF noise floor ~0.005 rad -> not discriminating)")

    # -- CSVs --------------------------------------------------------------
    # Paper table CSV
    paper_rows = []
    for s in summaries:
        ru = f"{s['mean_rms_u']:.3f}" if not np.isnan(s["mean_rms_u"]) else "-"
        paper_rows.append({
            "Controller":        s["controller"],
            "Settling Time (s)": _fmt(s),
            "Success Rate (%)":  _fmt_sr(s),
            "RMS Control Effort": ru,
        })
    pd.DataFrame(paper_rows).to_csv(
        os.path.join(outdir, "exp1_paper_table.csv"), index=False)

    # Full diagnostic CSV (everything)
    diag_rows = []
    for s in summaries:
        diag_rows.append({
            "Controller":        s["controller"],
            "Settling Time (s)": _fmt(s),
            "Success Rate (%)":  _fmt_sr(s),
            "RMS Control Effort": f"{s['mean_rms_u']:.3f}" if not np.isnan(s["mean_rms_u"]) else "-",
            "Median (s)":        f"{s['median_settling']:.2f}",
            "IQR (s)":           f"{s['iqr_settling']:.2f}",
            "p5 (s)":            f"{s['p5']:.2f}",
            "p95 (s)":           f"{s['p95']:.2f}",
            # diagnostic only
            "Overshoot (rad) [NOT in table]":
                f"{s['mean_overshoot']:.4f}" if not np.isnan(
                    s["mean_overshoot"]) else "-",
            "SS-RMSE θ (rad) [NOT in table]":
                f"{s['mean_rmse_ss']:.4f}" if not np.isnan(
                    s["mean_rmse_ss"]) else "-",
            "Ctrl time (ms)":
                f"{s['mean_ctrl_ms']:.2f}" if not np.isnan(
                    s["mean_ctrl_ms"]) else "-",
        })
    pd.DataFrame(diag_rows).to_csv(
        os.path.join(outdir, "exp1_full_diagnostic.csv"), index=False)
    print(f"\n  [CSV] {outdir}/exp1_paper_table.csv  (LaTeX-ready)")
    print(f"  [CSV] {outdir}/exp1_full_diagnostic.csv  (all metrics)")

    # -- Significance table ------------------------------------------------
    sig_df, d_df = build_sig_table(summaries)
    print("\n[Pairwise significance (Bonferroni)]")
    print(sig_df.to_string())
    sig_df.to_csv(os.path.join(outdir, "exp1_significance.csv"))
    d_df.to_csv(os.path.join(outdir,   "exp1_cohens_d.csv"))

    # -- Boxplot -----------------------------------------------------------
    plot_data = [s["settling_success"] for s in summaries]
    plot_labels = [s["controller"] for s in summaries]

    fig, ax = plt.subplots(figsize=(8, 5))
    _boxplot_ax(ax, plot_data, plot_labels,
                )
    # Annotate success rate + IQR above each box
    ylim_top = ax.get_ylim()[1]
    for i, s in enumerate(summaries):
        ax.text(i + 1, ylim_top * 0.98,
                f"SR={s['success_rate']:.0f}%\nIQR={s['iqr_settling']:.1f}s",
                ha="center", va="top", fontsize=7, color="navy",
                linespacing=1.4)
    _save(fig, os.path.join(outdir, "exp1_boxplot"))

    # -- CI bar chart ------------------------------------------------------
    means = [s["mean_settling"] for s in summaries]
    errs = [[s["mean_settling"] - s["ci_lo"] for s in summaries],
            [s["ci_hi"] - s["mean_settling"] for s in summaries]]
    colors = [_ctrl_color(s["controller"]) for s in summaries]

    fig, ax = plt.subplots(figsize=(8, 4))
    bars = ax.bar(plot_labels, means, yerr=errs, capsize=5,
                  color=colors, alpha=0.8, edgecolor="black")
    # Annotate control effort below label
    for i, (bar, s) in enumerate(zip(bars, summaries)):
        if not np.isnan(s["mean_rms_u"]):
            ax.text(bar.get_x() + bar.get_width()/2,
                    0.5,
                    f"u={s['mean_rms_u']:.2f}",
                    ha="center", va="bottom", fontsize=8, color="darkred")
    ax.set_ylabel("Mean Settling Time (s)", fontsize=10)
    ax.set_title("Mean Settling Time with 95% Bootstrap CI\n"
                 "(annotations show RMS control effort)")
    ax.tick_params(axis="x", rotation=20)
    ax.grid(axis="y", alpha=0.3)
    _save(fig, os.path.join(outdir, "exp1_ci_bars"))

    # -- ADD ISE bar chart -------------------------------------------------
    _plot_ise_bars(summaries, outdir)

    # -- Significance heatmap ----------------------------------------------
    pmat, _, _, alpha_c = compute_stat_tests(summaries)
    names = [s["controller"] for s in summaries]
    sig_matrix = np.zeros((len(names), len(names)))
    for i in range(len(names)):
        for j in range(len(names)):
            v = pmat.iloc[i, j]
            sig_matrix[i, j] = -np.log10(v) if not np.isnan(v) else 0

    fig, ax = plt.subplots(figsize=(7, 5))
    im = ax.imshow(sig_matrix, cmap="YlOrRd", aspect="auto")
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=30, ha="right")
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names)
    for i in range(len(names)):
        for j in range(len(names)):
            if i != j and not np.isnan(pmat.iloc[i, j]):
                txt = _pval_stars(pmat.iloc[i, j], alpha_c)
                ax.text(j, i, txt, ha="center", va="center", fontsize=9,
                        color="black" if sig_matrix[i, j] < 4 else "white")
    plt.colorbar(im, ax=ax, label="-log₁₀(p)")
    ax.set_title(f"Pairwise Significance (Bonferroni α={alpha_c:.4f})")
    _save(fig, os.path.join(outdir, "exp1_significance_heatmap"))

    return summaries


# -----------------------------------------------------------------------------

#  EXPERIMENT 2 - SENSOR NOISE ROBUSTNESS

# -----------------------------------------------------------------------------

def run_exp2(files: list, labels: list, outdir: str, n_boot: int = 10_000):
    print("\n" + "="*60)
    print("EXPERIMENT 2 - SENSOR NOISE ROBUSTNESS")
    print("="*60)

    if not labels:
        labels = [f"{i*10}%" for i in range(len(files))]

    # Load all datasets; collect union of controller names
    datasets = [load_json(f) for f in files]
    all_ctrls = sort_controllers(list(set(
        c for d in datasets for c in d["controllers"].keys()
    )))

    # Build summary dict:  summaries[ctrl][condition] = summary
    summaries = {c: {} for c in all_ctrls}
    for lbl, dat in zip(labels, datasets):
        for c in all_ctrls:
            if c in dat["controllers"]:
                summaries[c][lbl] = controller_summary(
                    c, dat["controllers"][c], n_boot)

    # -- Table: paper Table (noise) ----------------------------------------
    print("\n[Noise robustness table]")
    rows = []
    for c in all_ctrls:
        row = {"Controller": c}
        baseline_mean = None
        for lbl in labels:
            if lbl in summaries[c]:
                s = summaries[c][lbl]
                row[lbl] = f"{s['mean_settling']:.1f} [{s['ci_lo']:.1f}, {s['ci_hi']:.1f}]"
                row[f"{lbl}_SR"] = f"{s['success_rate']:.0f}%"
                if baseline_mean is None:
                    baseline_mean = s["mean_settling"]
            else:
                row[lbl] = "-"
        # Degradation vs. 0% baseline
        last_lbl = labels[-1]
        if last_lbl in summaries[c] and baseline_mean is not None and baseline_mean > 0:
            last_mean = summaries[c][last_lbl]["mean_settling"]
            pct = 100 * (last_mean - baseline_mean) / baseline_mean
            row["Degradation"] = f"+{pct:.0f}%" if pct >= 0 else f"{pct:.0f}%"
        rows.append(row)

    df_noise = pd.DataFrame(rows)
    csv_path = os.path.join(outdir, "exp2_noise_table.csv")
    df_noise.to_csv(csv_path, index=False)
    print(df_noise.to_string(index=False))

    # -- Plot 1: Line plot - settling time vs noise level -----------------
    fig, ax = plt.subplots(figsize=(8, 5))
    for c in all_ctrls:
        xs, ys, ye_lo, ye_hi = [], [], [], []
        for i, lbl in enumerate(labels):
            if lbl in summaries[c]:
                s = summaries[c][lbl]
                xs.append(i)
                ys.append(s["mean_settling"])
                ye_lo.append(s["mean_settling"] - s["ci_lo"])
                ye_hi.append(s["ci_hi"] - s["mean_settling"])
        if xs:
            col = _ctrl_color(c)
            ax.errorbar(xs, ys, yerr=[ye_lo, ye_hi],
                        label=c, color=col, marker="o",
                        linewidth=2, capsize=4, markersize=6)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels)
    ax.set_xlabel("Sensor Noise Level", fontsize=11)
    ax.set_ylabel("Mean Settling Time (s)", fontsize=11)
    ax.legend(fontsize=9, loc="upper left")
    ax.grid(alpha=0.3)
    _save(fig, os.path.join(outdir, "exp2_noise_curves"))

    # -- Plot 2: Success rate vs noise level -------------------------------
    fig, ax = plt.subplots(figsize=(8, 4))
    for c in all_ctrls:
        xs, ys = [], []
        for i, lbl in enumerate(labels):
            if lbl in summaries[c]:
                xs.append(i)
                ys.append(summaries[c][lbl]["success_rate"])
        if xs:
            ax.plot(xs, ys, label=c, color=_ctrl_color(c),
                    marker="s", linewidth=2, markersize=6)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels)
    ax.axhline(70, color="red", linestyle="--", linewidth=0.8, alpha=0.6,
               label="70% threshold")
    ax.set_xlabel("Sensor Noise Level", fontsize=11)
    ax.set_ylabel("Success Rate (%)", fontsize=11)
    ax.set_title("Success Rate vs. Sensor Noise Level - Experiment 2")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    _save(fig, os.path.join(outdir, "exp2_success_curves"))

    # -- Plot 3: Boxplots per noise level (one subplot per condition) ------
    n_cond = len(labels)
    fig, axes = plt.subplots(1, n_cond, figsize=(4 * n_cond, 5), sharey=True)
    if n_cond == 1:
        axes = [axes]
    for ax, lbl, dat in zip(axes, labels, datasets):
        ctrls_here = sort_controllers(list(dat["controllers"].keys()))
        d_list = []
        l_list = []
        for c in ctrls_here:
            mets = [_trial_metrics(t) for t in dat["controllers"][c]]
            d_list.append([m["settling_time"] for m in mets])
            l_list.append(c)
        _boxplot_ax(ax, d_list, l_list,
                    title=f"Noise {lbl}", show_fail_line=False)
        ax.axhline(FAIL_TIME, color="red", linestyle="--",
                   linewidth=0.6, alpha=0.4)
    fig.suptitle("Settling Time Distributions by Noise Level - Experiment 2",
                 fontsize=12, y=1.01)
    plt.tight_layout()
    _save(fig, os.path.join(outdir, "exp2_boxplots_per_noise"))

    # -- Plot 4: Degradation bar chart -------------------------------------
    base_lbl = labels[0]
    last_lbl = labels[-1]
    deg_ctrls, deg_vals, deg_cols = [], [], []
    for c in all_ctrls:
        if base_lbl in summaries[c] and last_lbl in summaries[c]:
            bm = summaries[c][base_lbl]["mean_settling"]
            lm = summaries[c][last_lbl]["mean_settling"]
            if bm > 0:
                deg_ctrls.append(c)
                deg_vals.append(100 * (lm - bm) / bm)
                deg_cols.append(_ctrl_color(c))

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(deg_ctrls, deg_vals, color=deg_cols, alpha=0.8, edgecolor="black")
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_ylabel(f"Settling Time Degradation (%) vs. {base_lbl}", fontsize=10)
    ax.set_title(f"Performance Degradation at {last_lbl} Noise - Experiment 2")
    ax.tick_params(axis="x", rotation=20)
    ax.grid(axis="y", alpha=0.3)
    for i, (c, v) in enumerate(zip(deg_ctrls, deg_vals)):
        ax.text(i, v + 0.5, f"{v:+.0f}%", ha="center", va="bottom", fontsize=9)
    _save(fig, os.path.join(outdir, "exp2_degradation_bars"))

    return summaries


# -----------------------------------------------------------------------------

#  EXPERIMENT 3 - CONTROL FREQUENCY ROBUSTNESS

# -----------------------------------------------------------------------------

def run_exp3(files: list, labels: list, outdir: str, n_boot: int = 10_000):
    print("\n" + "="*60)
    print("EXPERIMENT 3 - CONTROL FREQUENCY ROBUSTNESS")
    print("="*60)

    if not labels:
        labels = ["5 Hz", "10 Hz", "20 Hz", "50 Hz"]

    datasets = [load_json(f) for f in files]

    all_ctrls = sort_controllers([
        c for c in set(c for d in datasets for c in d["controllers"].keys())
        if c in DESIGN_FREQ  # <- ADD THIS FILTER
    ])

    summaries = {c: {} for c in all_ctrls}
    for lbl, dat in zip(labels, datasets):
        for c in all_ctrls:
            if c in dat["controllers"]:
                summaries[c][lbl] = controller_summary(
                    c, dat["controllers"][c], n_boot)

    # -- parse Hz values for x-axis ----------------------------------------
    def _hz(lbl):
        try:
            return float(lbl.replace("Hz", "").replace(" ", ""))
        except:
            return float(labels.index(lbl))

    hz_vals = [_hz(l) for l in labels]

    # -- Table: paper Table (frequency) -----------------------------------
    print("\n[Frequency robustness table]")
    rows = []
    for c in all_ctrls:
        row = {"Controller": c, "Design Freq": f"{DESIGN_FREQ.get(c,'?')} Hz"}
        design_lbl = None
        # find the label closest to design freq
        for lbl in labels:
            if abs(_hz(lbl) - DESIGN_FREQ.get(c, 0)) < 5:
                design_lbl = lbl
                break
        for lbl in labels:
            if lbl in summaries[c]:
                s = summaries[c][lbl]
                val_str = f"{s['mean_settling']:.1f}"
                if design_lbl and lbl != design_lbl and design_lbl in summaries[c]:
                    ref = summaries[c][design_lbl]["mean_settling"]
                    delta = 100 * (s["mean_settling"] - ref) / \
                        ref if ref > 0 else 0
                    sign = "+" if delta >= 0 else ""
                    val_str += f" ({sign}{delta:.0f}%)"
                elif lbl == design_lbl:
                    val_str += " (ref)"
                row[lbl] = val_str
                row[f"{lbl}_SR"] = f"{s['success_rate']:.0f}%"
            else:
                row[lbl] = "-"
        rows.append(row)

    df_freq = pd.DataFrame(rows)
    csv_path = os.path.join(outdir, "exp3_freq_table.csv")
    df_freq.to_csv(csv_path, index=False)
    print(df_freq.to_string(index=False))

    # -- Plot 1: Line plot - settling time vs frequency --------------------
    fig, ax = plt.subplots(figsize=(8, 5))
    for c in all_ctrls:
        xs, ys, ye_lo, ye_hi = [], [], [], []
        for lbl, hz in zip(labels, hz_vals):
            if lbl in summaries[c]:
                s = summaries[c][lbl]
                xs.append(hz)
                ys.append(s["mean_settling"])
                ye_lo.append(s["mean_settling"] - s["ci_lo"])
                ye_hi.append(s["ci_hi"] - s["mean_settling"])
        if xs:
            ax.errorbar(xs, ys, yerr=[ye_lo, ye_hi],
                        label=c, color=_ctrl_color(c), marker="o",
                        linewidth=2, capsize=4, markersize=6)
    # Mark design frequencies with vertical dashed lines
    for hz_d in set(DESIGN_FREQ.values()):
        if hz_d in hz_vals:
            ax.axvline(hz_d, color="grey", linestyle=":",
                       linewidth=0.8, alpha=0.7)
    ax.set_xscale("log")
    ax.xaxis.set_major_formatter(mticker.ScalarFormatter())
    ax.set_xticks(hz_vals)
    ax.set_xticklabels(labels)
    ax.set_xlabel("Control Frequency (Hz)", fontsize=11)
    ax.set_ylabel("Mean Settling Time (s)", fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    _save(fig, os.path.join(outdir, "exp3_freq_curves"))

    # -- Plot 2: Success rate vs frequency --------------------------------
    fig, ax = plt.subplots(figsize=(8, 4))
    for c in all_ctrls:
        xs, ys = [], []
        for lbl, hz in zip(labels, hz_vals):
            if lbl in summaries[c]:
                xs.append(hz)
                ys.append(summaries[c][lbl]["success_rate"])
        if xs:
            ax.plot(xs, ys, label=c, color=_ctrl_color(c),
                    marker="s", linewidth=2, markersize=6)
    ax.set_xscale("log")
    ax.xaxis.set_major_formatter(mticker.ScalarFormatter())
    ax.set_xticks(hz_vals)
    ax.set_xticklabels(labels)
    ax.axhline(70, color="red", linestyle="--", linewidth=0.8, alpha=0.6)
    ax.set_xlabel("Control Frequency (Hz)", fontsize=11)
    ax.set_ylabel("Success Rate (%)", fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    _save(fig, os.path.join(outdir, "exp3_success_curves"))

    # -- Plot 3: Boxplots per frequency ------------------------------------
    n_cond = len(labels)
    fig, axes = plt.subplots(1, n_cond, figsize=(4 * n_cond, 5), sharey=True)
    if n_cond == 1:
        axes = [axes]
    for ax, lbl, dat in zip(axes, labels, datasets):
        ctrls_here = sort_controllers([
            c for c in dat["controllers"].keys()
            if c in DESIGN_FREQ  # <- ADD THIS FILTER
        ])
        d_list, l_list = [], []
        for c in ctrls_here:
            mets = [_trial_metrics(t) for t in dat["controllers"][c]]
            d_list.append([m["settling_time"] for m in mets])
            l_list.append(c)
        _boxplot_ax(ax, d_list, l_list, title=lbl, show_fail_line=False)
        ax.axhline(FAIL_TIME, color="red", linestyle="--",
                   linewidth=0.6, alpha=0.4)
    plt.tight_layout()
    _save(fig, os.path.join(outdir, "exp3_boxplots_per_freq"))

    # -- Plot 4: Spider / radar chart for overall flexibility -------------
    # (normalised settling time: lower = better -> invert for radar)
    categories = labels
    n_cat = len(categories)
    angles = np.linspace(0, 2 * np.pi, n_cat, endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(6, 6),
                           subplot_kw=dict(polar=True))
    for c in all_ctrls:
        vals = []
        for lbl in labels:
            if lbl in summaries[c]:
                v = summaries[c][lbl]["mean_settling"]
                # invert & normalise so larger radius = better
                vals.append(1.0 / max(v, 1e-3))
            else:
                vals.append(0.0)
        vals += vals[:1]
        ax.plot(angles, vals, label=c, color=_ctrl_color(c), linewidth=2)
        ax.fill(angles, vals, color=_ctrl_color(c), alpha=0.1)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, fontsize=9)
    ax.set_title("Frequency Flexibility (1/settling_time, higher=better)",
                 fontsize=10, pad=15)
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1), fontsize=8)
    _save(fig, os.path.join(outdir, "exp3_radar"))

    return summaries


# -----------------------------------------------------------------------------

#  CROSS-EXPERIMENT RANKING TABLE

# -----------------------------------------------------------------------------

def overall_ranking(s_exp1, s_exp2, s_exp3,
                    noise_ref_label="30%",
                    freq_ref_label="5 Hz",
                    outdir="."):
    """
    Build Table 5 (overall ranking) from the paper.
    s_exp1: list of summaries (Exp1)
    s_exp2: dict ctrl -> {label -> summary}
    s_exp3: dict ctrl -> {label -> summary}
    """
    print("\n" + "="*60)
    print("OVERALL RANKING TABLE")
    print("="*60)

    ctrls = sort_controllers(list({s["controller"] for s in s_exp1}))
    rows = []
    for c in ctrls:
        # Nominal speed rank (lower settling = better rank)
        e1 = next((s for s in s_exp1 if s["controller"] == c), None)
        nom_t = e1["mean_settling"] if e1 else np.nan

        # Noise robustness: % degradation at worst noise
        if c in s_exp2 and noise_ref_label in s_exp2[c]:
            base_lbl = sorted(s_exp2[c].keys())[0]
            base_m = s_exp2[c][base_lbl]["mean_settling"]
            last_m = s_exp2[c][noise_ref_label]["mean_settling"]
            noise_deg = 100 * (last_m - base_m) / max(base_m, 1e-3)
        else:
            noise_deg = np.nan

        # Freq robustness: % degradation at lowest tested frequency
        if c in s_exp3 and freq_ref_label in s_exp3[c]:
            ref_lbl = sorted(s_exp3[c].keys(),
                             key=lambda l: abs(float(l.replace("Hz", "").replace(" ", ""))
                                               - DESIGN_FREQ.get(c, 50)))

            ref_m = s_exp3[c][ref_lbl[0]
                              ]["mean_settling"] if ref_lbl else np.nan
            last_m_f = s_exp3[c][freq_ref_label]["mean_settling"]
            freq_deg = 100 * (last_m_f - ref_m) / max(ref_m, 1e-3)
        else:
            freq_deg = np.nan

        rows.append({
            "Controller":     c,
            "Nom. Settling":  f"{nom_t:.2f} s" if not np.isnan(nom_t) else "-",
            f"Noise Deg @{noise_ref_label}": f"{noise_deg:+.0f}%" if not np.isnan(noise_deg) else "-",
            f"Freq Deg @{freq_ref_label}":   f"{freq_deg:+.0f}%" if not np.isnan(freq_deg) else "-",
        })

    df = pd.DataFrame(rows)
    csv_path = os.path.join(outdir, "overall_ranking.csv")
    df.to_csv(csv_path, index=False)
    print(df.to_string(index=False))
    return df


# -----------------------------------------------------------------------------

#  THETA TRAJECTORY PLOTS  (optional, --traj flag)

# -----------------------------------------------------------------------------

def plot_trajectories(data: dict, outdir: str, max_trials: int = 3):
    """Plot representative theta(t) trajectories for each controller."""
    ctrlnames = sort_controllers(list(data["controllers"].keys()))
    fig, ax = plt.subplots(figsize=(10, 5))

    for c in ctrlnames:
        trials = data["controllers"][c]
        col = _ctrl_color(c)
        plotted = 0
        for trial in trials:
            if plotted >= max_trials:
                break
            states = trial.get("states", [])
            times = trial.get("timestamps", [])
            if not states or not times:
                continue
            thetas = np.array([s[2] for s in states])
            label = c if plotted == 0 else "_nolegend_"
            ax.plot(times, thetas, color=col, alpha=0.7,
                    linewidth=1.2, label=label)
            plotted += 1

    ax.axhline(0,           color="black", linewidth=0.8, linestyle="--")
    ax.axhline(SETTLING_BAND, color="green", linewidth=0.8,
               linestyle=":", alpha=0.6, label=f"±{SETTLING_BAND} rad band")
    ax.axhline(-SETTLING_BAND, color="green", linewidth=0.8,
               linestyle=":", alpha=0.6, label="_nolegend_")
    ax.set_xlabel("Time (s)", fontsize=11)
    ax.set_ylabel("Ball Angle θ (rad)", fontsize=11)
    ax.set_title("Representative θ(t) Trajectories - Experiment 1")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    _save(fig, os.path.join(outdir, "exp1_theta_trajectories"))


# -----------------------------------------------------------------------------

#  COMPUTE-TIME PLOT  (latency analysis)

# -----------------------------------------------------------------------------

def plot_compute_times(data: dict, outdir: str):
    ctrlnames = sort_controllers(list(data["controllers"].keys()))
    means, stds, labels = [], [], []
    for c in ctrlnames:
        trials = data["controllers"][c]
        all_ct = []
        for t in trials:
            ct = t.get("controller_time_ms", [])
            all_ct.extend(ct)
        if all_ct:
            means.append(np.mean(all_ct))
            stds.append(np.std(all_ct))
            labels.append(c)

    if not means:
        return

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(labels, means,
           yerr=stds, capsize=5,
           color=[_ctrl_color(l) for l in labels],
           alpha=0.8, edgecolor="black")
    ax.set_ylabel("Controller Compute Time (ms)", fontsize=10)
    ax.set_title("Mean Controller Compute Time ± 1 std")
    ax.tick_params(axis="x", rotation=20)
    ax.grid(axis="y", alpha=0.3)
    _save(fig, os.path.join(outdir, "exp1_compute_times"))


# -----------------------------------------------------------------------------

#  CLI

# -----------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="Ball-and-Arc benchmark analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--exp", choices=["1", "2", "3", "all"], required=True,
                   help="Which experiment to analyse.")
    p.add_argument("--files", nargs="+", default=[],
                   help="JSON file(s) for the selected experiment "
                        "(single file for exp1, 4 files for exp2/3).")
    p.add_argument("--noise_files", nargs="+", default=[],
                   help="4 JSON files for noise experiment (exp=all only).")
    p.add_argument("--freq_files",  nargs="+", default=[],
                   help="4 JSON files for frequency experiment (exp=all only).")
    p.add_argument("--labels", nargs="+", default=[],
                   help="Condition labels, e.g. '0%% 10%% 20%% 30%%'.")
    p.add_argument("--outdir", default="figures",
                   help="Output directory for figures and CSVs (default: figures/).")
    p.add_argument("--traj", action="store_true",
                   help="(Exp1) Also plot representative theta trajectories.")
    p.add_argument("--n_boot", type=int, default=10_000,
                   help="Bootstrap iterations (default 10000).")
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    if args.exp == "1":
        f = args.files[0] if args.files else str(DEFAULT_EXP1_FILE)
        data = load_json(f)
        s_exp1 = run_exp1(data, args.outdir, args.n_boot)
        if args.traj:
            plot_trajectories(data, args.outdir)
        plot_compute_times(data, args.outdir)

    elif args.exp == "2":
        if len(args.files) < 2:
            sys.exit("ERROR: provide at least 2 JSON files for exp 2")
        run_exp2(args.files, args.labels, args.outdir, args.n_boot)

    elif args.exp == "3":
        if len(args.files) < 2:
            sys.exit("ERROR: provide at least 2 JSON files for exp 3")
        run_exp3(args.files, args.labels, args.outdir, args.n_boot)

    elif args.exp == "all":
        # Experiment 1
        f_e1 = args.files[0] if args.files else str(DEFAULT_EXP1_FILE)
        data_e1 = load_json(f_e1)
        s_exp1 = run_exp1(data_e1, args.outdir, args.n_boot)
        if args.traj:
            plot_trajectories(data_e1, args.outdir)
        plot_compute_times(data_e1, args.outdir)

        # Experiment 2
        s_exp2 = {}
        if args.noise_files:
            noise_labels = args.labels if args.labels else \
                ["0%", "10%", "20%", "30%"]
            s_exp2 = run_exp2(args.noise_files, noise_labels,
                              args.outdir, args.n_boot)
        else:
            print("[WARN] No --noise_files provided, skipping Experiment 2.")

        # Experiment 3
        s_exp3 = {}
        if args.freq_files:
            freq_labels = ["5 Hz", "10 Hz", "20 Hz", "50 Hz"]
            s_exp3 = run_exp3(args.freq_files, freq_labels,
                              args.outdir, args.n_boot)
        else:
            print("[WARN] No --freq_files provided, skipping Experiment 3.")

        # Overall ranking (only if all three experiments ran)
        if s_exp2 and s_exp3:
            overall_ranking(s_exp1, s_exp2, s_exp3, outdir=args.outdir)

    print(f"\nDone. All outputs in: {os.path.abspath(args.outdir)}/")


if __name__ == "__main__":
    main()
