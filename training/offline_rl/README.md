# Offline RL - ArcBall Control

Offline reinforcement learning for the ArcBall cart-control task using
d3rlpy (https://d3rlpy.readthedocs.io). Supports both discrete and
continuous action spaces.

d3rlpy is included in the main project environment (`env.yaml` at the
repository root). No separate environment is needed.

## Environment

    conda env create -f ../../env.yaml   # if not already created
    conda activate balancer
    pip install -e "../../balancer/[all]"

## Scripts

| Script | Purpose |
|--------|--------|
| offline_train_arcball.py | Discrete offline RL (DiscreteCQL / DiscreteBCQ) |
| offline_train_cont.py | Continuous offline RL (CQL / IQL / TD3+BC / BCQ) |
| eval_offline_sim.py | Simulation evaluation for trained d3rlpy models |

## Continuous Training

Dataset: arcball_real_demonstrations/arcball_post_recalib_flat.h5 (~333k transitions from
1,415 hardware trials, actions in [-1, 1]). Not expert data: 83% successful / 17% failed episodes.

Key features:
- Dense reward recomputation: balanced_reward + dip_weight * dip_reward, clipped [-1,1]
- Proper episode segmentation with timeouts (not false terminals)
- Observation and action scalers (StandardObservationScaler, MinMaxActionScaler)
- Saves .d3 format compatible with OfflineRLController

### Algorithms

| --algo | Class | Key hyperparams |
|--------|-------|------------------|
| cql | CQL | --cql-weight (conservative_weight) |
| iql | IQL | --iql-expectile, --iql-weight-temp |
| td3bc | TD3PlusBC | --td3bc-alpha |
| bcq | BCQ | --bcq-lam, --bcq-action-flexibility |

### Quick start

    # CQL (default) with dense reward recomputation
    python offline_train_cont.py --data-path ../../data/dataset/arcball_real_demonstrations/arcball_post_recalib_flat.h5

    # IQL on GPU with W&B
    python offline_train_cont.py --algo iql --n-steps 500000 --gpu --wandb

    # TD3+BC with higher dip bonus
    python offline_train_cont.py --algo td3bc --dip-weight 0.3

    # Use stored (sparse) reward instead of recomputing
    python offline_train_cont.py --use-stored-reward

    # Dry run (validate data loading only)
    python offline_train_cont.py --dry-run

Full CLI: python offline_train_cont.py --help

### Key defaults

| Flag | Default | Purpose |
|------|---------|--------|
| --algo | cql | Algorithm |
| --n-steps | 500000 | Total gradient updates |
| --batch-size | 256 | Mini-batch size |
| --gamma | 0.99 | Discount factor |
| --actor-lr | 3e-4 | Actor learning rate |
| --critic-lr | 3e-4 | Critic learning rate |
| --n-critics | 2 | Q-function ensemble size |
| --tau | 0.005 | Soft target update rate |
| --episode-length | 600 | Steps per episode (30s at 20Hz) |
| --dip-weight | 0.2 | Binary dip reward bonus weight |
| --use-scalers | True | Observation/action normalization |
| --eval-interval | 10000 | Steps between evaluations |

## Evaluation

    # Evaluate a trained model in simulation
    python eval_offline_sim.py --model-path ./cont_cql_model.d3

    # Multiple models, 100 trials
    python eval_offline_sim.py --model-path ./cont_cql_model.d3 ./cont_iql_model.d3 --num-trials 100

Outputs: results/<model_name>_eval.json (one JSON per model: trial-by-trial success/failure
and summary stats, matching the schema used across evaluation/results/).

## Integration with eval_sim.py

To use trained models in the full evaluation pipeline:
1. Copy .d3 files to evaluation/models/offline_rl/
2. Run (from the repo root): make eval-sim
3. discover_offline_rl_models() will auto-detect them
4. balancer.controllers.OfflineRLController wraps them as BaseController

## Model format

- Continuous offline RL saves .d3 files (via algo.save())
- Load with: d3rlpy.load_learnable(path)
- Or via: OfflineRLController(path) for step-by-step inference
- NOT compatible with SB3 .zip files (those use RLController)

## Discrete Training (Legacy)

Dataset: arcball_discrete_3M.h5 (archived, not included in this release)
See: python offline_train_arcball.py --help

## Cross-references

- Data README: ../../data/README.md
- Evaluation README: ../../evaluation/README.md
- OfflineRLController: ../../balancer/balancer/controllers/offline_rl.py
- Online RL pipeline: ../README.md
