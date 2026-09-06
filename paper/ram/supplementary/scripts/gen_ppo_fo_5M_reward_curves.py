#!/usr/bin/env python3
"""Regenerate the first-order 5M PPO training-reward curves (Appendix V, Fig. 6).

Reads the three Stable-Baselines3 ``evaluations.npz`` eval logs committed under
``paper/data/rl_5m_ablation/training_runs/`` and plots the mean eval reward
(20 episodes per checkpoint) against training step.

Smoothing is a centred rolling mean computed with ``min_periods`` semantics, so
the first and last points average over however many samples exist rather than
being divided by the full window. A window that runs off the end and still
divides by its full width drags the tail toward zero and invents a collapse
that is not in the logs: all three variants finish above 340.
"""
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

HERE = Path(__file__).resolve().parent
RUNS = HERE.parent.parent.parent / 'data' / 'rl_5m_ablation' / 'training_runs'
OUT = HERE.parent / 'figures' / 'ppo_fo_5M_reward_curves.png'

THEORETICAL_MAX = 360
WINDOW = 21  # centred, in evaluation points

VARIANTS = [
    ('ppo_fo_5M_base',  'PPO base (no DR, no BZ)', '#1f9ae0'),
    ('ppo_fo_5M_dr',    'PPO + DR',                '#2ca02c'),
    ('ppo_fo_5M_dr_bz', 'PPO + BZ + DR',           '#e0245e'),
]


def rolling_mean(y, window):
    """Centred rolling mean that shrinks the window at both edges."""
    half = window // 2
    out = np.empty_like(y, dtype=float)
    for i in range(len(y)):
        lo, hi = max(0, i - half), min(len(y), i + half + 1)
        out[i] = y[lo:hi].mean()
    return out


fig, ax = plt.subplots(figsize=(7.0, 3.9))

for run, label, color in VARIANTS:
    d = np.load(RUNS / run / 'eval_logs' / 'evaluations.npz')
    steps = d['timesteps'] / 1e6
    reward = d['results'].mean(axis=1)
    ax.plot(steps, rolling_mean(reward, WINDOW), color=color, lw=1.4, label=label)

ax.axhline(THEORETICAL_MAX, color='0.55', ls='--', lw=0.9, zorder=1)
ax.annotate(f'theoretical max $\\approx$ {THEORETICAL_MAX}',
            xy=(0.015, THEORETICAL_MAX), xycoords=('axes fraction', 'data'),
            va='top', ha='left', fontsize=8, color='0.35',
            xytext=(0, -4), textcoords='offset points')

ax.set_xlabel('Training steps (M)')
ax.set_ylabel('Mean eval reward (20 episodes)')
ax.set_title('PPO training reward, first-order motor, 5M steps')
ax.set_xlim(0, 5)
ax.set_ylim(top=THEORETICAL_MAX + 45)
ax.grid(alpha=0.3)
ax.legend(loc='lower right', framealpha=0.95)

fig.tight_layout()
# The supplement includes .eps; the .png is kept for quick inspection.
fig.savefig(OUT.with_suffix('.eps'))
fig.savefig(OUT, dpi=200)
print(f'wrote {OUT.with_suffix(".eps")}')
print(f'wrote {OUT}')

for run, label, _ in VARIANTS:
    d = np.load(RUNS / run / 'eval_logs' / 'evaluations.npz')
    r = d['results'].mean(axis=1)
    print(f'  {label:24} final {r[-1]:6.1f}   min over last 50 {r[-50:].min():6.1f}')
