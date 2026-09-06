#!/usr/bin/env python3
"""Generate PPO ablation checkpoint figure (s-curve eval)."""
import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

HERE = Path(__file__).resolve().parent
FIG_DIR = HERE.parent.parent / 'ram' / 'supplementary' / 'figures'

with open(HERE / 'ppo_checkpoint_sweep.json') as f:
    data = json.load(f)

variants = {
    'ppo_balanced':    {'label': 'Balanced', 'color': '#2196F3', 'marker': 'o'},
    'ppo_gaussian':    {'label': 'Gaussian', 'color': '#FF9800', 'marker': 's'},
    'ppo_balanced_dr': {'label': 'Balanced + DR', 'color': '#4CAF50', 'marker': 'D'},
}

checkpoints = ['500000', '1000000', '1500000', '2000000', '2500000', '3000000']
ckpt_labels = ['0.5M', '1M', '1.5M', '2M', '2.5M', '3M']

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

for variant_key, props in variants.items():
    srs = []
    settles = []
    vdata = data['results'][variant_key]
    for ckpt in checkpoints:
        r = vdata.get(ckpt, {})
        srs.append(r.get('sr', 0) * 100)
        settles.append(r.get('settle', np.nan))

    ax1.plot(range(len(checkpoints)), srs, label=props['label'],
             color=props['color'], marker=props['marker'], linewidth=2, markersize=7)
    ax2.plot(range(len(checkpoints)), settles, label=props['label'],
             color=props['color'], marker=props['marker'], linewidth=2, markersize=7)

ax1.set_xticks(range(len(checkpoints)))
ax1.set_xticklabels(ckpt_labels)
ax1.set_xlabel('Training Steps')
ax1.set_ylabel('Success Rate (%)')
ax1.set_ylim([30, 105])
ax1.axhline(100, color='gray', linestyle='--', alpha=0.3)
ax1.legend(loc='lower right', fontsize=9)
ax1.set_title('(a) Success Rate vs. Training Steps')
ax1.grid(True, alpha=0.3)

ax2.set_xticks(range(len(checkpoints)))
ax2.set_xticklabels(ckpt_labels)
ax2.set_xlabel('Training Steps')
ax2.set_ylabel('Mean Settling Time (s)')
ax2.set_ylim([0, 3.5])
ax2.legend(loc='upper right', fontsize=9)
ax2.set_title('(b) Settling Time vs. Training Steps')
ax2.grid(True, alpha=0.3)

plt.tight_layout()
out_path = FIG_DIR / 'ppo_ablation_checkpoints.png'
plt.savefig(out_path, dpi=200, bbox_inches='tight')
print(f'Saved: {out_path}')
plt.close()
