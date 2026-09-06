#!/usr/bin/env python3
"""Regenerate wall-override dependence figure (Appendix IV, Fig. 1).

Only includes the four classical controllers that use WO (PD, LQR, SMC, MPC).
NMPC is excluded because it does not use a wall-override mechanism.
"""
import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

HERE = Path(__file__).resolve().parent
DATA = HERE.parent.parent.parent / 'data' / 'exp1_hardware' / 'consolidated.json'
OUT = HERE.parent / 'figures' / 'wall_override_dependence.png'

WO_THRESHOLD = 0.66  # cart |x| threshold for override zone

CONTROLLERS = [
    ('PD+WO', 'PID_WO'),
    ('LQR+WO', 'LQR_WO'),
    ('SMC+WO', 'SMC_WO'),
    ('MPC+WO', 'MPC_WO'),
]

COLORS = ['#FF8A80', '#FFB74D', '#CE93D8', '#80CBC4']

with open(DATA) as f:
    data = json.load(f)

labels = []
fire_rates = []
active_fracs = []
end_positions = []

for label, key in CONTROLLERS:
    trials = data['controllers'][key]
    n_fired = 0
    step_fracs = []
    ends = []
    for t in trials:
        if not t['success']:
            continue
        cart = np.abs(np.array(t['states'])[:, 0])
        in_zone = cart > WO_THRESHOLD
        if np.any(in_zone):
            n_fired += 1
            step_fracs.append(np.mean(in_zone))
        ends.append(t['states'][-1][0])  # signed end position

    n_success = sum(1 for t in trials if t['success'])
    labels.append(label)
    fire_rates.append(n_fired / n_success * 100 if n_success else 0)
    active_fracs.append(np.mean(step_fracs) * 100 if step_fracs else 0)
    end_positions.append(ends)

# --- Figure (2 panels: a + b) ---
fig, axes = plt.subplots(2, 1, figsize=(7, 7), gridspec_kw={'height_ratios': [1, 0.8]})
fig.suptitle(
    'Wall-override dependence across the four WO classical controllers\n'
    f'override fires in {min(fire_rates):.0f}\u2013{max(fire_rates):.0f}% of successful trials',
    fontsize=11, fontweight='bold'
)

# (a) Override firing rate + active fraction
ax = axes[0]
x = np.arange(len(labels))
width = 0.35
bars1 = ax.bar(x - width/2, fire_rates, width, color=[c for c in COLORS], label='trials where override fired (%)')
bars2 = ax.bar(x + width/2, active_fracs, width, color=[c for c in COLORS], alpha=0.5,
               hatch='//', label='mean % of steps active (when fired)')
ax.set_ylabel('Percent (%)')
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.set_ylim([0, 110])
ax.set_title('(a) Override firing rate and active-step fraction by controller')
ax.legend(loc='upper left', fontsize=8)
for i, v in enumerate(fire_rates):
    ax.text(i - width/2, v + 2, f'{v:.0f}', ha='center', fontsize=9, fontweight='bold')

# (b) End-of-trial cart position
ax = axes[1]
bins = np.linspace(-0.8, 0.8, 30)
for i, (label, ends) in enumerate(zip(labels, end_positions)):
    ax.hist(ends, bins=bins, alpha=0.5, color=COLORS[i], label=label)
ax.axvline(-WO_THRESHOLD, color='red', linestyle='--', alpha=0.5)
ax.axvline(WO_THRESHOLD, color='red', linestyle='--', alpha=0.5)
ax.set_xlabel('End-of-trial cart position (m)')
ax.set_ylabel('Trials')
ax.set_title('(b) End-of-trial cart position; red = override trigger zone')
ax.legend(loc='upper left', fontsize=8)

plt.tight_layout()
plt.savefig(OUT, dpi=150, bbox_inches='tight')
print(f'Saved: {OUT}')
plt.close()
