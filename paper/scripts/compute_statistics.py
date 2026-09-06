#!/usr/bin/env python3
"""Authoritative statistics for the Exp-1 hardware benchmark (Table I + Section V).

Single source of truth for every statistic reported in the main paper's
experimental protocol and results:

  * Success rate (SR) with 95% Wilson score intervals (correct at the
    100% boundary, where a bootstrap degenerates to [100, 100]).
  * Settling time: median [IQR] and mean over successful trials only.
  * RMS control effort over all trials.
  * Pairwise significance: Mann-Whitney U on the successful-trial settling
    times (a rank test suited to the skewed distributions), Bonferroni
    corrected across all C(13,2)=78 controller pairs.
  * Success-rate differences: Fisher's exact test, same Bonferroni family.
  * RMS-effort differences: Mann-Whitney U over all trials, its own
    Bonferroni family of 78 pairs (the axis on which the benchmark has
    power to separate controllers).
  * Cohen's d for settling gaps between the fast controllers.

Run:  python paper/scripts/compute_statistics.py
Writes paper/data/exp1_hardware/exp1_statistics.csv and
paper/data/exp1_hardware/exp1_effort_significance.csv, and prints the Table-I
rows and the significance summary quoted in Section V and Finding 1.
"""
import json
import os
import itertools
import numpy as np
from scipy import stats

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)  # paper/
DATA = os.path.join(_ROOT, "data", "exp1_hardware", "consolidated.json")

# Cart-velocity axis limit (m/s). The commanded velocity saturates here before
# it reaches the actuator (balancer/hardware/action_utils.encode_continuous_action,
# 900 mm/s), so control effort is the RMS of the *clipped* command actually sent.
VMAX = 0.9

# The 13 benchmarked controllers, display name -> consolidated.json key,
# in the Table-I sort order (SR desc, then mean settling).
T13 = [
    ("MPPI", "MPPI_v12"), ("PPO(WM)", "PPO_WM_v1"),
    ("PPO(BZ+DR)", "PPO_BZ_DR"), ("PD+WO", "PID_WO"),
    ("SMC+WO", "SMC_WO"), ("LQR+WO", "LQR_WO"), ("IQL", "IQL_OffRL"), ("TD3", "TD3_DR"),
    ("NMPC", "NMPC_N10"), ("TRPO", "TRPO_DR"), ("TQC", "TQC_DR"), ("SAC", "SAC_DR"),
    ("MPC+WO", "MPC_WO"),
]
N_PAIRS = len(T13) * (len(T13) - 1) // 2  # 78
ALPHA = 0.05


def wilson_ci(k, n, z=1.96):
    """95% Wilson score interval for a binomial proportion, in percent."""
    if n == 0:
        return float("nan"), float("nan")
    p = k / n
    den = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / den
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return 100 * (centre - half), 100 * (centre + half)


def cohens_d(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    pooled = np.sqrt((np.var(a, ddof=1) + np.var(b, ddof=1)) / 2.0)
    return float((np.mean(a) - np.mean(b)) / pooled) if pooled > 0 else 0.0


def load():
    ctr = json.load(open(DATA))["controllers"]
    out = {}
    for name, key in T13:
        tr = ctr[key]
        n = len(tr)
        succ = np.array([1 if t.get("success") else 0 for t in tr])
        settle = np.array([t["settling_time"] for t in tr if t.get("success")])
        eff = [np.sqrt(np.mean(np.clip(np.array(t["actions"]), -VMAX, VMAX) ** 2))
               for t in tr if t.get("actions")]
        out[name] = dict(n=n, k=int(succ.sum()), succ=succ, settle=settle,
                         eff=np.array(eff))
    return out


def main():
    R = load()
    # ---- Per-controller summary (Table I) ----
    lines = ["controller,SR_pct,wilson_lo,wilson_hi,mean_s,median_s,iqr_lo,iqr_hi,rms_effort"]
    print(f"{'controller':<12}{'SR%':>5}{'95% Wilson':>14}{'mean':>7}"
          f"{'median [IQR]':>20}{'effort':>8}")
    for name, _ in T13:
        r = R[name]
        sr = 100 * r["k"] / r["n"]
        lo, hi = wilson_ci(r["k"], r["n"])
        s = r["settle"]
        q1, q3 = np.percentile(s, [25, 75])
        eff = float(np.mean(r["eff"]))
        print(f"{name:<12}{sr:>5.0f}{f'[{lo:.0f}, {hi:.0f}]':>14}{s.mean():>7.2f}"
              f"{f'{np.median(s):.2f} [{q1:.2f}, {q3:.2f}]':>20}{eff:>8.3f}")
        lines.append(f"{name},{sr:.0f},{lo:.1f},{hi:.1f},{s.mean():.2f},"
                     f"{np.median(s):.2f},{q1:.2f},{q3:.2f},{eff:.3f}")

    # ---- Pairwise significance ----
    alpha_bonf = ALPHA / N_PAIRS
    settle_sig = 0
    sr_sig = 0
    effort_sig = 0
    mppi_faster = []
    mppi_ns = []
    eff_rows = []
    for a, b in itertools.combinations([n for n, _ in T13], 2):
        _, p_set = stats.mannwhitneyu(R[a]["settle"], R[b]["settle"],
                                      alternative="two-sided")
        settle_sig += p_set < alpha_bonf
        ca = [[R[a]["k"], R[a]["n"] - R[a]["k"]], [R[b]["k"], R[b]["n"] - R[b]["k"]]]
        _, p_sr = stats.fisher_exact(ca)
        sr_sig += p_sr < alpha_bonf
        _, p_eff = stats.mannwhitneyu(R[a]["eff"], R[b]["eff"],
                                      alternative="two-sided")
        effort_sig += p_eff < alpha_bonf
        eff_rows.append((a, b, float(np.median(R[a]["eff"])),
                         float(np.median(R[b]["eff"])), p_eff,
                         int(p_eff < alpha_bonf)))
        if a == "MPPI":
            (mppi_faster if p_set < alpha_bonf else mppi_ns).append((b, p_set))

    # For each controller, which others it uses significantly LESS effort than
    # (one-sided Mann-Whitney, same Bonferroni threshold). MPPI and IQL each beat
    # 11 of 12 and are statistically tied with each other, so we report the count
    # and the exception rather than demanding an unattainable 12/12.
    names = [n for n, _ in T13]
    gentle_lines = []
    for c in names:
        beaten = [o for o in names if o != c and
                  stats.mannwhitneyu(R[c]["eff"], R[o]["eff"],
                                     alternative="less")[1] < alpha_bonf]
        if len(beaten) >= len(names) - 2:
            ties = [o for o in names if o != c and o not in beaten]
            gentle_lines.append(
                f"  {c}: significantly gentler than {len(beaten)}/{len(names) - 1}"
                + (f"; not separable from {', '.join(ties)}" if ties else ""))

    print(f"\nBonferroni family: {N_PAIRS} pairs, alpha_corrected = {alpha_bonf:.2e}")
    print(f"Settling (Mann-Whitney, success-only): {settle_sig}/{N_PAIRS} pairs significant")
    print(f"Success rate (Fisher exact):           {sr_sig}/{N_PAIRS} pairs significant")
    print(f"RMS effort (Mann-Whitney, all trials): {effort_sig}/{N_PAIRS} pairs significant")
    print("Lowest-effort controllers (one-sided Mann-Whitney vs each other):")
    for line in gentle_lines:
        print(line)
    print("MPPI settling significantly faster than: "
          + ", ".join(n for n, _ in mppi_faster))
    print("MPPI settling NOT separable from:        "
          + ", ".join(n for n, _ in mppi_ns))
    for other in ("LQR+WO",):
        print(f"Cohen's d, MPPI vs {other} (settling): "
              f"{cohens_d(R['MPPI']['settle'], R[other]['settle']):.2f}")

    # ---- Per-stratum median settling (Finding 3 difficulty breakdown) ----
    # Strata by cart start; "at wall" combines the two extreme strata.
    strata = ["easy", "medium", "hard", "extreme_opposite", "extreme_same"]
    hdr = ["Center", "Mid", "NearWall", "Wall-opp", "Wall-same", "AtWall"]
    ctr = json.load(open(DATA))["controllers"]
    key_of = dict(T13)
    print("\nPer-stratum median settling (successful trials only; n in parens):")
    print(f"{'controller':<12}" + "".join(f"{h:>13}" for h in hdr))
    for name, _ in T13:
        tr = ctr[key_of[name]]

        def _med(pred):
            v = [t["settling_time"] for t in tr if t.get("success") and pred(t)]
            return f"{np.median(v):.2f}({len(v)})" if v else f"--({0})"
        cells = [_med(lambda t, s=s: t.get("difficulty") == s) for s in strata]
        cells.append(_med(lambda t: t.get("difficulty") in
                          ("extreme_opposite", "extreme_same")))
        print(f"{name:<12}" + "".join(f"{c:>13}" for c in cells))

    out_csv = os.path.join(_ROOT, "data", "exp1_hardware", "exp1_statistics.csv")
    with open(out_csv, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"\nWrote {out_csv}")

    eff_csv = os.path.join(_ROOT, "data", "exp1_hardware",
                           "exp1_effort_significance.csv")
    with open(eff_csv, "w") as f:
        f.write("controller_a,controller_b,median_effort_a,median_effort_b,"
                "p_value,significant\n")
        for a, b, ma, mb, p, s in eff_rows:
            f.write(f"{a},{b},{ma:.3f},{mb:.3f},{p:.3e},{s}\n")
    print(f"Wrote {eff_csv}")


if __name__ == "__main__":
    main()
