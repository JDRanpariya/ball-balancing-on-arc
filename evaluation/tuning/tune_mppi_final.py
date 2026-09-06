"""Final MPPI refinement - combine best dimensions."""
import sys
import time as _time
import json
import os

sys.path.insert(0, '..')

import numpy as np
import torch
from simulate import (
    simulate_controller, get_tuning_initial_conditions,
    compute_metrics_ise, CART_LIMIT, FAIL_TIME,
)
from balancer.world_model import WorldModelPredictor
from balancer.controllers.mppi import MPPIController


device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Device: {device}')

predictor_path = '../models/world_model/v1/best_model.pth'
ics = get_tuning_initial_conditions()

# Combine best findings:
# - Q[2]=80 is optimal
# - H=15 is optimal
# - λ=0.5 best settle speed, λ=0.2 best ISE
# - R=0.05 also gets 5/7
# - N=800 gets 4/7 (more samples = better coverage)
# Now combine:
configs = [
    # Baseline best
    ("baseline",       [0.0, 1.0, 80.0, 1.0],  0.01,  0.5, 15, 500),
    # Higher R + more samples
    ("R05_N800",       [0.0, 1.0, 80.0, 1.0],  0.05,  0.5, 15, 800),
    ("R05_N500",       [0.0, 1.0, 80.0, 1.0],  0.05,  0.5, 15, 500),
    ("R03_N800",       [0.0, 1.0, 80.0, 1.0],  0.03,  0.5, 15, 800),
    # Lower lambda + more samples  
    ("lam02_N800",     [0.0, 1.0, 80.0, 1.0],  0.01,  0.2, 15, 800),
    ("lam03_N800",     [0.0, 1.0, 80.0, 1.0],  0.01,  0.3, 15, 800),
    ("lam04_N800",     [0.0, 1.0, 80.0, 1.0],  0.01,  0.4, 15, 800),
    # Combine R + lambda
    ("R03_lam03",      [0.0, 1.0, 80.0, 1.0],  0.03,  0.3, 15, 800),
    ("R03_lam04",      [0.0, 1.0, 80.0, 1.0],  0.03,  0.4, 15, 800),
    ("R05_lam03",      [0.0, 1.0, 80.0, 1.0],  0.05,  0.3, 15, 800),
    # Slightly higher theta
    ("q90_R03_lam04",  [0.0, 1.0, 90.0, 1.0],  0.03,  0.4, 15, 800),
    ("q90_R03_N500",   [0.0, 1.0, 90.0, 1.0],  0.03,  0.5, 15, 500),
]

SIM_T = 30.0

print(f'\nFinal sweep: {len(configs)} configs, {len(ics)} ICs x {SIM_T}s')
print(f'{"Config":>18s}  {"Settled":>7s}  {"AvgSettle":>10s}  {"ISE_theta":>10s}  {"Time":>6s}')
print('=' * 70)

results = []

for name, Q_diag, R, lam, horizon, n_samples in configs:
    predictor = WorldModelPredictor(predictor_path, device=device)
    ctrl = MPPIController(
        predictor=predictor, horizon=horizon, n_samples=n_samples,
        lambda_=lam, device=device, Q_diag=Q_diag, R=R,
    )

    t0 = _time.time()
    settled_count = 0
    total_settle = 0.0
    total_ise_theta = 0.0

    for s0 in ics:
        ctrl.reset()
        time_arr, theta_arr, u_arr = simulate_controller(
            ctrl, s0, Ts=0.05, sim_T=SIM_T, action_type='cont')
        m = compute_metrics_ise(time_arr, theta_arr, u_arr)
        if m['settling_time'] < FAIL_TIME:
            settled_count += 1
        total_settle += m['settling_time']
        total_ise_theta += m['ise_theta']

    elapsed = _time.time() - t0
    n = len(ics)
    avg_settle = total_settle / n
    avg_ise = total_ise_theta / n

    print(f'{name:>18s}  {settled_count:>3d}/{n:<3d}  {avg_settle:10.2f}  '
          f'{avg_ise:10.6f}  {elapsed:5.0f}s')

    results.append({
        'name': name, 'Q_diag': Q_diag, 'R': R, 'lambda_': lam,
        'horizon': horizon, 'n_samples': n_samples,
        'settled': settled_count, 'total_ics': n,
        'avg_settling_time': avg_settle, 'avg_ise_theta': avg_ise,
    })

results.sort(key=lambda r: (-r['settled'], r['avg_settling_time']))

print(f'\n{"=" * 70}')
print('TOP 5:')
for i, r in enumerate(results[:5]):
    print(f"  #{i+1} {r['name']}: {r['settled']}/{r['total_ics']}, "
          f"settle={r['avg_settling_time']:.2f}s, ISE_θ={r['avg_ise_theta']:.6f}")
    print(f"      Q={r['Q_diag']}, R={r['R']}, λ={r['lambda_']}, "
          f"H={r['horizon']}, N={r['n_samples']}")

# Save best
best = results[0]
outdir = 'tuning/mppi'
os.makedirs(outdir, exist_ok=True)
param_dict = {
    'Q_diag': best['Q_diag'], 'R': best['R'], 'lambda_': best['lambda_'],
    'horizon': best['horizon'], 'n_samples': best['n_samples'],
    'checkpoint': 'models/world_model/v1/best_model.pth',
    'Ts': 0.05, 'tuned_on': 'cont', 'method': 'final_sweep',
    'settled': best['settled'],
    'avg_settling_time': best['avg_settling_time'],
}
with open(os.path.join(outdir, 'best_params.json'), 'w') as f:
    json.dump(param_dict, f, indent=2)
print(f'\n  Saved -> {outdir}/best_params.json')
