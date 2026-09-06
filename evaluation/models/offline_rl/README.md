# Offline-RL checkpoints (d3rlpy `.d3`)

`evaluation/utils/load_controllers.py`'s `discover_offline_rl_models()`
discovers the continuous-action IQL checkpoint here (d3rlpy `.d3` format,
nested one level deep).

    iql_best_demos/
      model_280000.d3    # IQL, 280k steps, expectile=0.7

LFS-tracked via `.gitattributes` - run `git lfs pull` to materialise. The
sim benchmark loads it automatically; its results populate Table I's IQL row.
