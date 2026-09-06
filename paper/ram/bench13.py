"""Shared controller-naming constants for the exp1 benchmark.

Import-safe (no side effects): used by regenerate_extra_figures.py and
paper/verify_table1.py so the verifier does not trigger figure
regeneration on import.
"""

# -- Paper-consistent naming ------------------------------------------------
NAME_MAP = {
    'PID_vanilla':      'PD',
    'PID_WO':           'PD+WO',
    'LQR_basic':        'LQR',
    'LQR_Qcenter':      'LQR (Q-center)',
    'LQR_WO':           'LQR+WO',
    'SMC_basic':        'SMC',
    'SMC_full':         'SMC+WO (full)',
    'SMC_WO':           'SMC+WO',
    'MPC_basic_constr': 'MPC',
    'MPC_WO':           'MPC+WO',
    'NMPC_N10':         'NMPC',
    'PPO_base':         'PPO',
    'PPO_DR':           'PPO(DR)',
    'PPO_BZ_DR':        'PPO(BZ+DR)',
    'TD3_DR':           'TD3',
    'SAC_DR':           'SAC',
    'TQC_DR':           'TQC',
    'TRPO_DR':          'TRPO',
    'PPO_WM_v1':        'PPO(WM)',
    'IQL_OffRL':        'IQL',
    'MPPI_v12':         'MPPI',
}

# The 13 deployed benchmark controllers (PPO_base and PPO_DR are excluded
# ablations, reported separately in the failed-variants table).
BENCH13 = ['PID_WO', 'LQR_WO', 'SMC_WO', 'MPC_WO', 'NMPC_N10',
           'PPO_BZ_DR', 'TD3_DR', 'SAC_DR',
           'TQC_DR', 'PPO_WM_v1', 'IQL_OffRL', 'MPPI_v12', 'TRPO_DR']
