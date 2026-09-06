"""Stable-Baselines3 training driver for the ball-on-arc RL policies.

Trains PPO/SAC/TD3/TRPO/TQC and related algorithms via Stable-Baselines3
(Raffin et al., JMLR 2021) and its SB3-Contrib extensions, configured with
Hydra.
"""

import logging
import os
from pathlib import Path
try:
    import wandb
except ImportError:
    wandb = None
import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf
import numpy as np

import stable_baselines3
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.callbacks import (
    EvalCallback,
    StopTrainingOnRewardThreshold,
)
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.logger import make_output_format
from stable_baselines3.common.vec_env import (
    DummyVecEnv,
    SubprocVecEnv,
    VecVideoRecorder,
)

from balancer.utils.utils import (
    CheckpointCallbackLimited,
    CustomLogger,
    PerformanceUpdateFixedIntervalCallback,
    PerStepStateRecorderCallback,
    seed_everything,
)
from balancer.envs.real import BalancerReal

_CONFIG_PATH = str(Path(__file__).resolve().parent.parent / "configs")


@hydra.main(version_base="1.3", config_path=_CONFIG_PATH, config_name="train.yaml")
def main(cfg: DictConfig):

    logging_level = logging.DEBUG if cfg.debug else logging.INFO
    logging.basicConfig(format="%(levelname)s: %(message)s",
                        level=logging_level)

    print("=== ENV TARGET ===")
    print(cfg.env)

    # Hydra-managed run directories
    run_dir = Path(HydraConfig.get().runtime.output_dir)
    checkpoint_dir = run_dir / "checkpoints"
    model_dir = run_dir / "models"
    video_dir = run_dir / "videos"

    # setup wandb run (skipped when WANDB_MODE=disabled or wandb not installed)
    run = None
    if os.environ.get("WANDB_MODE", "").lower() != "disabled" and wandb is not None:
        run = wandb.init(
            config=OmegaConf.to_container(
                cfg, resolve=True, throw_on_missing=True),
            project=cfg.wandb.project,
            entity=cfg.wandb.entity,
            sync_tensorboard=True,
            monitor_gym=cfg.capture_video,
            save_code=True,
            dir=str(run_dir)
        )

    def make_env(render_mode='rgb_array', performance_eval=None):
        env = hydra.utils.instantiate(
            cfg.env, render_mode=render_mode, _recursive_=True)

        check_env(env)

        seed_everything(env, cfg.seed, cfg.cudnn_deterministic)

        for wrapper in cfg.wrappers:
            env = hydra.utils.instantiate(wrapper, env=env)

        if isinstance(env.unwrapped, BalancerReal):
            assert cfg.n_envs == 1, "n_envs > 1 but using real robot"

        print(f"action_space = {env.action_space}")
        if hasattr(env.unwrapped, 'dyn'):
            print(f"dyn.action_type = {env.unwrapped.dyn.action_type}")

        return env

    def make_eval_env(render_mode='rgb_array'):
        """Eval env: fixed nominal params for stable evaluation."""
        eval_cfg = OmegaConf.to_container(cfg.env, resolve=True)
        eval_cfg['fixed_param'] = True       # <- override DR for eval
        env = hydra.utils.instantiate(eval_cfg, render_mode=render_mode, _recursive_=True)

        check_env(env)
        seed_everything(env, cfg.seed + 1000, cfg.cudnn_deterministic)  # different seed

        for wrapper in cfg.wrappers:
            env = hydra.utils.instantiate(wrapper, env=env)

        return env

    vec_env = make_vec_env(lambda: make_env(), n_envs=cfg.n_envs)

    # setup algo/model
    verbose = 2 if cfg.debug else 0
    model = hydra.utils.instantiate(
        cfg.algo,
        env=vec_env,
        tensorboard_log=str(run_dir / "tb"),
        verbose=verbose,
        _convert_="all",
        _recursive_=True,
    )

    folder = str(run_dir / "tb")
    format_strings = ["tensorboard"]
    output_formats = [make_output_format(
        _format=f, log_dir=folder) for f in format_strings]
    model.set_logger(CustomLogger(
        folder=folder, output_formats=output_formats))

    if cfg.model_path:
        model_path = Path(cfg.model_path)
        assert model_path.exists(), f"Model path {model_path} does not exist"
        model = model.load(model_path, env=vec_env)
        print(model)

        # Override hyperparameters from config after loading pretrained model
        # (model.load restores saved hyperparams; we want FT-specific values)
        # Only override LR and entropy - buffer-related params (n_steps, batch_size)
        # are baked into the rollout buffer at init and cannot be changed post-load.
        if hasattr(cfg.algo, 'learning_rate'):
            lr = cfg.algo.learning_rate
            model.learning_rate = lr
            model.lr_schedule = lambda _: lr
        if hasattr(cfg.algo, 'ent_coef') and hasattr(model, 'ent_coef'):
            model.ent_coef = cfg.algo.ent_coef
        if hasattr(cfg.algo, 'batch_size') and hasattr(model, 'batch_size'):
            model.batch_size = cfg.algo.batch_size
        if hasattr(cfg.algo, 'n_epochs') and hasattr(model, 'n_epochs'):
            model.n_epochs = cfg.algo.n_epochs

        # Re-set logger after loading a model
        folder = str(run_dir / "tb")
        format_strings = ["tensorboard"]
        output_formats = [make_output_format(
            _format=f, log_dir=folder) for f in format_strings]
        model.set_logger(CustomLogger(
            folder=folder, output_formats=output_formats))

    # Stop training when the model reaches the reward threshold
    eval_callback = None

    # Detect real robot to disable the eval env
    is_real_robot = "BalancerReal" in cfg.env.get("_target_", "")

    if cfg.evaluation.eval_freq is not None:

        # Real robot: skip the separate eval environment
        if is_real_robot:
            print("Real Robot detected: Disabling separate evaluation environment.")
            eval_env = None
        else:
            eval_env = make_vec_env(make_eval_env, n_envs=cfg.n_envs)

            if cfg.evaluation.early_stopping_reward_threshold is not None:
                callback_on_best = StopTrainingOnRewardThreshold(
                    reward_threshold=cfg.evaluation.early_stopping_reward_threshold,
                    verbose=1
                )
            else:
                callback_on_best = None

            # account for vec env, see sb3 doc
            eval_freq = max(cfg.evaluation.eval_freq // cfg.n_envs, 1)
            eval_callback = EvalCallback(
                eval_env,
                best_model_save_path=str(run_dir / "best_model"),
                log_path=str(run_dir / "eval_logs"),
                deterministic=cfg.evaluation.deterministic,
                n_eval_episodes=cfg.evaluation.n_eval_episodes,
                eval_freq=eval_freq,
                callback_on_new_best=callback_on_best,
                verbose=1,
            )

    # Save checkpoints every n steps, keep at most x checkpoints
    checkpoint_callback = CheckpointCallbackLimited(
        save_freq=cfg.checkpoint.save_freq,
        save_path=str(checkpoint_dir),
        max_checkpoints=cfg.checkpoint.max_checkpoints,
        verbose=1
    )

    # Filter None entries from callbacks (real-robot case)
    raw_callbacks = [eval_callback, checkpoint_callback]
    callback_list = [c for c in raw_callbacks if c is not None]

    try:
        logging.info("Starting to train")

        if isinstance(model, stable_baselines3.TD3):
            n_actions = vec_env.action_space.shape[0]
            model.action_noise = NormalActionNoise(
                mean=np.zeros(n_actions),
                sigma=0.1 * np.ones(n_actions)
            )

        model.learn(
            total_timesteps=cfg.total_timesteps,
            callback=callback_list,
            progress_bar=cfg.progress_bar,
            reset_num_timesteps=False,
        )
    except KeyboardInterrupt:
        logging.info("Interupting training")

    # only save last video
    if cfg.capture_video:
        video_length = 500
        env = DummyVecEnv([make_env])
        env = VecVideoRecorder(
            env,
            str(video_dir),
            record_video_trigger=lambda x: x == 0,
            video_length=video_length,
        )

        obs = env.reset()
        for _ in range(video_length + 1):
            action, _ = model.predict(obs, deterministic=False)
            obs, _, done, _ = env.step(action)
            if done:
                break
        env.close()
        # Keep videos under the run's video_dir
        if run is not None:
            wandb.log({"video": wandb.Video(
                str(video_dir / "rl-video-step-0-to-step-500.mp4"),
                format="mp4"
            )})

    if cfg.save_model:
        logging.info("Saving model to artifacts")
        model_dir.mkdir(parents=True, exist_ok=True)    # ensure dir exists
        model_save_path = str(model_dir / "model.zip")
        model.save(model_save_path)

    if run is not None:
        run.finish()


if __name__ == "__main__":
    main()
