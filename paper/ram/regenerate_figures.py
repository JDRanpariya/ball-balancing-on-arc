#!/usr/bin/env python3
"""
Regenerate all supplementary figures from consolidated.json.
All 13 controllers, correct numbers.
Output: supplementary/figures/ and figures/
"""
import json, os, sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from collections import defaultdict

BASE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(BASE, '../data/exp1_hardware/consolidated.json')
SUPP_FIG = os.path.join(BASE, 'supplementary/figures')
MAIN_FIG = os.path.join(BASE, 'figures')
os.makedirs(SUPP_FIG, exist_ok=True)
os.makedirs(MAIN_FIG, exist_ok=True)

with open(DATA) as f:
    d = json.load(f)

SCHEDULE = d.get('schedule', [])

# schedule lives in source files, not consolidated
if not SCHEDULE:
    import os
    sched_src = os.path.join(BASE, '../data/exp1_hardware/sources/mppi_v12.json')
    if os.path.exists(sched_src):
        with open(sched_src) as f:
            s_data = json.load(f)
        SCHEDULE = s_data.get('schedule', [])

# ===== Controller display names and ordering =====
CTRL_ORDER = [
    ('MPPI_v12',   'MPPI',         'real-data',    '#1a6496'),
    ('PPO_WM_v1',  'PPO (WM)',      'real-data',    '#2196F3'),
    ('LQR_WO',     'LQR+WO',        'classical',    '#e67e22'),
    ('PPO_BZ_DR',  'PPO (BZ+DR)',   'sim-rl',       '#27ae60'),
    ('PID_WO',     'PD+WO',         'classical',    '#e74c3c'),
    ('SMC_WO',     'SMC+WO',        'classical',    '#f39c12'),
    ('TD3_DR',     'TD3 (DR)',      'sim-rl',       '#8e44ad'),
    ('IQL_OffRL',  'IQL',           'real-data',    '#16a085'),
    ('NMPC_N10',   'NMPC',          'predictive',   '#2c3e50'),
    ('TRPO_DR',    'TRPO (DR)',     'sim-rl',       '#7f8c8d'),
    ('TQC_DR',     'TQC (DR)',      'sim-rl',       '#3498db'),
    ('MPC_WO',     'MPC+WO',        'predictive',   '#95a5a6'),
    ('SAC_DR',     'SAC (DR)',      'sim-rl',       '#e91e63'),
]
PARADIGM_COLORS = {
    'real-data':  '#1a6496',
    'classical':  '#e67e22',
    'sim-rl':     '#27ae60',
    'predictive': '#2c3e50',
}

def get_stats(key):
    trials = d['controllers'].get(key, [])
    if not trials: return None
    n = len(trials)
    ok = [t for t in trials if t.get('success')]
    settles = sorted([t['settling_time'] for t in ok if t.get('settling_time')])
    sr = 100 * len(ok) / n
    all_s = [t['settling_time'] if t.get('success') and t.get('settling_time') else 30.0 for t in trials]
    return dict(sr=sr, n=n, n_ok=len(ok), settles=settles, all_s=all_s,
                mean=np.mean(settles) if settles else 0,
                median=np.median(settles) if settles else 0,
                q1=np.percentile(settles, 25) if settles else 0,
                q3=np.percentile(settles, 75) if settles else 0)

stats = {}
for key, label, paradigm, color in CTRL_ORDER:
    s = get_stats(key)
    if s:
        s.update(label=label, paradigm=paradigm, color=color)
        stats[key] = s

# ===== Figure 1: HW SR Ranking (hw_sr_ranking.png) =====
fig, ax = plt.subplots(figsize=(8, 5))
items = [(key, stats[key]) for key, _, _, _ in CTRL_ORDER if key in stats]
items_sorted = sorted(items, key=lambda x: (-x[1]['sr'], x[1]['mean']))
names = [s['label'] for _, s in items_sorted]
srs = [s['sr'] for _, s in items_sorted]
colors = [PARADIGM_COLORS[s['paradigm']] for _, s in items_sorted]
bars = ax.barh(range(len(names)), srs, color=colors, edgecolor='black', linewidth=0.4)
ax.set_yticks(range(len(names)))
ax.set_yticklabels(names, fontsize=10)
ax.set_xlabel('Hardware Success Rate (%)', fontsize=11)
ax.set_xlim(0, 108)
ax.invert_yaxis()
for i, (bar, sr) in enumerate(zip(bars, srs)):
    ax.text(sr + 0.5, i, f'{sr:.0f}%', va='center', fontsize=9)
legend_elements = [Patch(facecolor=PARADIGM_COLORS[p], label=lbl)
                   for p, lbl in [('real-data','Real-Data'),
                                  ('classical','Classical'),
                                  ('sim-rl','Sim RL'),
                                  ('predictive','Predictive')]]
ax.legend(handles=legend_elements, loc='lower right', fontsize=9)
ax.set_title('Hardware Success Rate (50 trials, stratified)', fontsize=11)
ax.grid(axis='x', alpha=0.3)
plt.tight_layout()
plt.savefig(f'{SUPP_FIG}/hw_sr_ranking.png', dpi=150)
plt.savefig(f'{MAIN_FIG}/hw_sr_ranking.png', dpi=150)
plt.close()
print('Saved hw_sr_ranking.png')

# ===== Figure 2: HW SR vs Settling Pareto (hw_sr_vs_settling.png) =====
def plot_sr_settling_pareto(metric='median', out_paths=()):
    """Failure rate vs settling time scatter for all 13 controllers.

    Both axes increase toward worse outcomes (up = more failures,
    right = slower settling), so the ideal corner is bottom-left.
    Settling statistics use successful trials only. Marker color and
    shape encode the controller family; every point is direct-labeled
    with a leader line.
    """
    from matplotlib.lines import Line2D
    fig, ax = plt.subplots(figsize=(7, 5))
    entries = [(s[metric], 100 - s['sr'], s['label'], s['paradigm'])
               for s in stats.values() if s['sr'] > 0 and s['n_ok'] > 0]
    xmax, ybot, ytop = 12, -1.5, 27
    MARKERS = {'classical': 'o', 'sim-rl': 's',
               'real-data': 'D', 'predictive': '^'}
    overrides = {'NMPC': (-0.3, 0.9), 'TRPO (DR)': (0.25, 0.9),
                 'SMC+WO': (-0.3, -1.3), 'TD3 (DR)': (0.25, -1.4),
                 'LQR+WO': (0.45, -1.7), 'PPO (WM)': (0.3, 1.1),
                 'IQL': (0.25, 0.9)}
    offsets = [(0.25, 0.9), (0.25, -1.0), (-0.3, 0.9), (0.25, 1.9),
               (-0.3, -1.0)]
    placed = []
    for x, y, label, par in sorted(entries, key=lambda e: (e[0], e[1])):
        ax.scatter(x, y, c=PARADIGM_COLORS[par], s=110,
                   marker=MARKERS.get(par, 'o'),
                   edgecolors='white', linewidths=0.8,
                   zorder=6 if par == 'predictive' else 5)
        if label in overrides:
            dx, dy = overrides[label]
        else:
            for dx, dy in offsets:
                lx = x + dx
                ly = min(max(y + dy, ybot + 0.9), ytop - 0.9)
                if all(abs(lx - px) > 2.3 or abs(ly - py) > 1.5
                       for px, py in placed):
                    break
        lx = x + dx
        ly = min(max(y + dy, ybot + 0.9), ytop - 0.9)
        ha = 'left' if dx > 0 else 'right'
        placed.append((lx, ly))
        ax.annotate(label, (x, y), xytext=(lx, ly), fontsize=10.5,
                    va='center', ha=ha,
                    arrowprops=dict(arrowstyle='-', lw=0.5, color='0.6',
                                    shrinkA=0, shrinkB=3))
    mname = 'Median' if metric == 'median' else 'Mean'
    ax.set_xlabel(f'{mname} Settling Time (s, successful trials)',
                  fontsize=11)
    ax.set_ylabel('Failure Rate (%)', fontsize=11)
    ax.set_xlim(0, xmax)
    ax.set_ylim(ybot, ytop)
    ax.grid(True, alpha=0.3)
    ax.legend(handles=[Line2D([0], [0], marker=MARKERS[p], color='none',
                              markerfacecolor=PARADIGM_COLORS[p],
                              markersize=9, label=lbl)
                       for p, lbl in [('real-data', 'Real-Data'),
                                      ('classical', 'Classical'),
                                      ('sim-rl', 'Sim RL'),
                                      ('predictive', 'Predictive')]],
              loc='upper left', fontsize=10.5)
    ax.set_title(f'Failure Rate vs. {mname} Settling Time '
                 '(Hardware, 50 trials)', fontsize=11)
    plt.tight_layout()
    for path in out_paths:
        plt.savefig(path, dpi=150)
        plt.savefig(path.replace('.png', '.eps'))
    plt.close()

plot_sr_settling_pareto('median', (f'{SUPP_FIG}/hw_sr_vs_settling.png',
                                   f'{MAIN_FIG}/hw_sr_vs_settling.png'))
print('Saved hw_sr_vs_settling.png')

# ===== Figure 3: Settling Time Boxplot (effort_boxplot.png proxy) =====
fig, ax = plt.subplots(figsize=(12, 5))
plot_keys = [key for key, _, _, _ in CTRL_ORDER if key in stats and stats[key]['n_ok'] > 0]
all_settles_list = [stats[k]['settles'] for k in plot_keys]
colors_list = [stats[k]['color'] for k in plot_keys]
labels_list = [stats[k]['label'] for k in plot_keys]

bp = ax.boxplot(all_settles_list, patch_artist=True, notch=False, showfliers=True)
for patch, color in zip(bp['boxes'], colors_list):
    patch.set_facecolor(color)
    patch.set_alpha(0.8)
for i, (key, settles) in enumerate(zip(plot_keys, all_settles_list)):
    ax.annotate(f'{stats[key]["sr"]:.0f}%', (i+1, max(settles)+0.5),
                fontsize=7, ha='center', va='bottom')
ax.set_xticks(range(1, len(labels_list)+1))
ax.set_xticklabels(labels_list, rotation=45, ha='right', fontsize=9)
ax.set_ylabel('Settling Time (s, successful trials)', fontsize=11)
ax.set_title('Settling Time Distribution by Controller (annotations show hardware SR)', fontsize=11)
ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig(f'{SUPP_FIG}/settling_time_boxplot_hw.png', dpi=150)
plt.close()
print('Saved settling_time_boxplot_hw.png')

# ===== Figure 4: Median Settling by Stratum (hw_difficulty_breakdown.png) =====
if SCHEDULE and len(SCHEDULE) == 50:
    STRATA_MAP = {'easy': 'Center', 'medium': 'Mid-rail',
                  'hard': 'Near-wall', 'extreme_same': 'At-wall-same',
                  'extreme_opposite': 'At-wall-opp'}
    strata_order = ['easy', 'medium', 'hard', 'extreme_same', 'extreme_opposite']
    strata_labels = [STRATA_MAP[s] for s in strata_order]

    def get_settling_by_stratum(key):
        trials = d['controllers'].get(key, [])
        if len(trials) != 50: return {}
        by_diff = defaultdict(list)
        for trial, sched in zip(trials, SCHEDULE):
            by_diff[sched.get('difficulty', 'unknown')].append(trial)
        result = {}
        for diff, ts in by_diff.items():
            settles = [t['settling_time'] for t in ts if t.get('success') and t.get('settling_time')]
            result[diff] = np.median(settles) if settles else None
        return result

    top13 = [key for key, _, _, _ in CTRL_ORDER if key in stats]
    n_ctrl = len(top13)
    fig, ax = plt.subplots(figsize=(13, 5))
    bar_w = 0.06
    x = np.arange(len(strata_order))
    for i, key in enumerate(top13):
        med_by_s = get_settling_by_stratum(key)
        vals = [med_by_s.get(s) for s in strata_order]
        vals_plot = [v if v is not None else 0 for v in vals]
        offset = (i - n_ctrl/2 + 0.5) * bar_w
        ax.bar(x + offset, vals_plot, bar_w, label=stats[key]['label'],
               color=stats[key]['color'], alpha=0.85)
    ax.set_xlabel('Stratum (by cart position)', fontsize=11)
    ax.set_ylabel('Median Settling Time (s, successful)', fontsize=11)
    ax.set_xticks(x)
    ax.set_xticklabels(strata_labels, fontsize=10)
    ax.legend(loc='upper left', fontsize=7, ncol=3)
    ax.grid(axis='y', alpha=0.3)
    ax.set_title('Per-Stratum Median Settling Time (All 13 Controllers)', fontsize=11)
    plt.tight_layout()
    plt.savefig(f'{MAIN_FIG}/hw_difficulty_breakdown.png', dpi=150)
    plt.savefig(f'{SUPP_FIG}/hw_difficulty_breakdown.png', dpi=150)
    plt.close()
    print('Saved hw_difficulty_breakdown.png')

    # ===== Figure 5: SR by Stratum (settling_by_stratum.png, supplementary only) =====
    def get_sr_by_stratum(key):
        trials = d['controllers'].get(key, [])
        if len(trials) != 50: return {}
        by_diff = defaultdict(list)
        for trial, sched in zip(trials, SCHEDULE):
            by_diff[sched.get('difficulty', 'unknown')].append(trial)
        return {diff: 100*sum(1 for t in ts if t.get('success'))/len(ts)
                for diff, ts in by_diff.items()}

    fig, ax = plt.subplots(figsize=(12, 5))
    bar_w = 0.06
    for i, key in enumerate(top13):
        sr_by_s = get_sr_by_stratum(key)
        vals = [sr_by_s.get(s, 0) for s in strata_order]
        offset = (i - n_ctrl/2 + 0.5) * bar_w
        ax.bar(x + offset, vals, bar_w, label=stats[key]['label'],
               color=stats[key]['color'], alpha=0.85)
    ax.set_xlabel('Stratum (by cart position)', fontsize=11)
    ax.set_ylabel('Success Rate (%)', fontsize=11)
    ax.set_xticks(x)
    ax.set_xticklabels(strata_labels, fontsize=10)
    ax.set_ylim(0, 105)
    # Legend above the axes: at 'upper right' it sat on top of the
    # at-wall-opposite bars, which all reach 100%.
    ax.legend(loc='lower center', bbox_to_anchor=(0.5, 1.02), fontsize=7,
              ncol=5, frameon=False)
    ax.set_title('Per-Stratum Hardware SR (All 13 Controllers)', fontsize=11,
                 pad=54)
    plt.tight_layout()
    # The supplement includes .eps; the .png is kept for quick inspection.
    plt.savefig(f'{SUPP_FIG}/settling_by_stratum.eps')
    plt.savefig(f'{SUPP_FIG}/settling_by_stratum.png', dpi=150)
    plt.close()
    print('Saved settling_by_stratum.eps + .png')

print('All figures regenerated.')
