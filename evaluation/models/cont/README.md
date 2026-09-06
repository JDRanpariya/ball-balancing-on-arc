# Continuous-action RL checkpoints (`*.zip`)

`eval_sim.py --exp 1 --action-type cont` loads the 5 deployed RL checkpoints
(PPO, SAC, TD3, TQC, TRPO) from this directory via
`discover_rl_models("models", "cont")`. The simulation benchmark loads them
automatically; their results populate Table I's RL rows and the `rl_new4`
ablation.

Files (Stable-Baselines3 `.zip`, LFS-tracked via `.gitattributes` - run
`git lfs pull` to materialise):

    ppo.zip       # PPO (BZ+DR) - blind-zone + domain randomization
    sac.zip       # SAC (DR)
    td3.zip       # TD3 (DR)
    tqc.zip       # TQC (DR)
    trpo.zip      # TRPO (DR)
    ppo_wm_v1.zip # PPO-WM - policy trained in the learned world model (v1)

Trained with `training/scripts/train_all_cont.py` (see
`reproducibility_guide.md` Stage 6.2). PPO-WM is trained inside the world
model via `training/offline_rl/` (see supplementary G).
