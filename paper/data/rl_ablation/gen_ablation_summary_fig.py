#!/usr/bin/env python3
"""Generate complete ablation summary figure for Appendix V (Fig. 5)."""
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

HERE = Path(__file__).resolve().parent
FIG_DIR = HERE.parent.parent / 'ram' / 'supplementary' / 'figures'

# All PPO ablation data (sim + hardware)
variants = [
    'Balanced',
    'Balanced\n+ DR',
    'Gaussian',
    'Balanced\n+ BZ',
    'Balanced\n+ BZ+DR',
]

sim_sr = [100, 100, 95, 96, 100]
sim_settle = [1.72, 1.49, 2.22, 2.37, 1.67]

hw_sr = [100, 100, 30, 60, 100]
hw_settle = [5.26, 6.94, 15.62, 2.72, 4.98]

x = np.arange(len(variants))
width = 0.35

fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

# --- Success Rate ---
ax = axes[0]
bars1 = ax.bar(x - width/2, sim_sr, width, label='Simulation', color='#2196F3', alpha=0.8)
bars2 = ax.bar(x + width/2, hw_sr, width, label='Hardware', color='#FF5722', alpha=0.8)
ax.set_ylabel('Success Rate (%)')
ax.set_xticks(x)
ax.set_xticklabels(variants, fontsize=9)
ax.set_ylim([0, 110])
ax.axhline(100, color='gray', linestyle='--', alpha=0.3)
ax.legend(loc='lower left', fontsize=9)
ax.set_title('(a) Success Rate')
ax.grid(True, axis='y', alpha=0.3)

# Annotate failed variants
for i, (s, h) in enumerate(zip(sim_sr, hw_sr)):
    if h < 100:
        ax.annotate(f'{h}%', (i + width/2, h + 2), ha='center', fontsize=8, color='red')

# --- Settling Time ---
ax = axes[1]
bars1 = ax.bar(x - width/2, sim_settle, width, label='Simulation', color='#2196F3', alpha=0.8)
bars2 = ax.bar(x + width/2, hw_settle, width, label='Hardware', color='#FF5722', alpha=0.8)
ax.set_ylabel('Mean Settling Time (s)')
ax.set_xticks(x)
ax.set_xticklabels(variants, fontsize=9)
ax.set_ylim([0, 18])
ax.legend(loc='upper right', fontsize=9)
ax.set_title('(b) Settling Time (successful trials only)')
ax.grid(True, axis='y', alpha=0.3)

# Add sim-to-real ratio annotations
for i, (s, h) in enumerate(zip(sim_settle, hw_settle)):
    if hw_sr[i] >= 60:  # only annotate if meaningful
        ratio = h / s
        ax.annotate(f'{ratio:.1f}×', (i + width/2, h + 0.5),
                    ha='center', fontsize=7, color='#555', style='italic')

plt.tight_layout()
out_path = FIG_DIR / 'ppo_ablation_summary.png'
plt.savefig(out_path, dpi=200, bbox_inches='tight')
print(f'Saved: {out_path}')
plt.close()
