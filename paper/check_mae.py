import numpy as np, os, glob
from pathlib import Path
# wm_data_sweep lives at paper/data/wm_data_sweep/ - resolve relative to this
# file so the script runs from any cwd.
_ROOT = Path(__file__).resolve().parent
base = str(_ROOT / 'data' / 'wm_data_sweep')
sizes = sorted([d for d in os.listdir(base) if os.path.isdir(os.path.join(base,d))], key=lambda x: int(x) if x.isdigit() else 999999)
print(f"{'size':<10}{'bare_h30_traj':<22}{'wm_h30_traj':<22}{'bare-vs-wm':<16}")
for sz in sizes:
    p = os.path.join(base, sz, 'exp1_results.npz')
    if not os.path.exists(p):
        print(f"{sz:<10}no npz"); continue
    d = np.load(p, allow_pickle=True)
    bare = wm = None
    for k in d.files:
        if 'Bare' in k and 'tau=0.06' in k and 'h30_traj_mae' in k:
            bare = d[k]
        if 'WorldModel_LSTM' in k and 'h30_traj_mae' in k:
            wm = d[k]
    if bare is None or wm is None:
        # try other key patterns
        for k in d.files:
            if 'Bare' in k and 'h30_traj_mae' in k and 'tau=0.06' in k:
                bare = d[k]
            if 'world' in k.lower() and 'lstm' in k.lower() and 'h30_traj_mae' in k:
                wm = d[k]
    if bare is not None and wm is not None:
        bt = float(np.mean(bare)); wt = float(np.mean(wm))
        pct = (bt - wt) / wt * 100
        print(f"{sz:<10}{bt:<22.5f}{wt:<22.5f}{pct:+.1f}%  (bare {bt/wt:.2f}x wm)")
    else:
        print(f"{sz:<10}keys: {[k for k in d.files if 'h30_traj' in k][:4]}")
