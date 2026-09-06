#!/usr/bin/env python3
"""Reproduce the original curated exp1_boxplot.png style from consolidated.json (FO data)."""
import json, os, sys, numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt

BASE=os.path.dirname(os.path.abspath(__file__))
d=json.load(open(os.path.join(BASE,'../data/exp1_hardware/consolidated.json')))['controllers']

# key, label, family-color-group  (matches the original figure's palette)
CTRL=[
 ('MPPI_v12','MPPI','red'), ('PPO_WM_v1','PPO (WM)','red'),
 ('LQR_WO','LQR+WO','blue'), ('PPO_BZ_DR','PPO (BZ+DR)','green'),
 ('PID_WO','PD+WO','blue'), ('SMC_WO','SMC+WO','blue'),
 ('TD3_DR','TD3 (DR)','green'), ('IQL_OffRL','IQL','orange'),
 ('NMPC_N10','NMPC','blue'), ('TRPO_DR','TRPO (DR)','green'),
 ('TQC_DR','TQC (DR)','green'), ('SAC_DR','SAC (DR)','green'),
 ('MPC_WO','MPC+WO','blue'),
]
COLORS={'blue':'#4e79a7','green':'#59a14f','red':'#e15759','orange':'#f28e2b'}

def settles(key):
    tr=d.get(key,[])
    return sorted(t['settling_time'] for t in tr if t.get('success') and t.get('settling_time'))
def sr(key):
    tr=d.get(key,[]); return 100*sum(1 for t in tr if t.get('success'))/len(tr) if tr else 0

rows=[(k,lab,c,settles(k),sr(k)) for k,lab,c in CTRL]
rows.sort(key=lambda r:(-r[4], np.mean(r[3]) if r[3] else 0))   # SR desc, then mean settling
data=[r[3] for r in rows]; labels=[r[1] for r in rows]; cols=[COLORS[r[2]] for r in rows]

fig,ax=plt.subplots(figsize=(7,4.4))
bp=ax.boxplot(data, patch_artist=True, showfliers=True, widths=0.6,
    flierprops=dict(marker='o', markerfacecolor='none', markeredgecolor='0.45', markersize=3.5, alpha=0.8),
    medianprops=dict(color='#e67e22', linewidth=1.5),
    whiskerprops=dict(color='0.2', linewidth=1.1),
    capprops=dict(color='0.2', linewidth=1.1),
    boxprops=dict(edgecolor='0.2', linewidth=1.0))
for patch,c in zip(bp['boxes'],cols):
    patch.set_facecolor(c); patch.set_alpha(0.85)
ax.axhline(30, ls='--', color='#f4a6a6', lw=1.8, label='Timeout')
ax.set_ylabel('Settling Time (s)', fontsize=12)
ax.set_xticks(range(1,len(labels)+1))
ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=10.5)
ax.set_ylim(-1.2, 31)
ax.set_yticks(range(0,31,5))
ax.tick_params(axis='y', labelsize=10.5)
ax.legend(loc='upper left', fontsize=10.5, frameon=True)
for spine in ('top','right'): ax.spines[spine].set_visible(True)
plt.tight_layout()
out=sys.argv[1] if len(sys.argv)>1 else os.path.join(BASE,'figures/exp1_boxplot.png')
plt.savefig(out, dpi=170)
plt.savefig(os.path.splitext(out)[0] + '.eps')
print('saved', out)
print('order:', labels)
