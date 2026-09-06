import os
import random
from pathlib import Path
from typing import Optional, SupportsFloat, Tuple

import torch
import numpy as np
import torch.nn as nn

from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.logger import Logger, KVWriter, DISABLED


class CheckpointCallbackLimited(BaseCallback):
    def __init__(self, save_freq, save_path, max_checkpoints=5, verbose=0):
        super().__init__(verbose)
        self.save_freq = save_freq
        self.save_path = Path(save_path)
        self.max_checkpoints = max_checkpoints
        self.checkpoints = []

    def _init_callback(self):
        # Ensure save directory exists
        self.save_path.mkdir(parents=True, exist_ok=True)

    def _on_step(self) -> bool:
        if self.num_timesteps % self.save_freq == 0:
            # Define checkpoint file path
            checkpoint_file = self.save_path / f"checkpoint_step_{self.num_timesteps}.zip"
            self.model.save(checkpoint_file)

            # Track saved checkpoints
            self.checkpoints.append(checkpoint_file)

            # Remove old checkpoints if over the limit
            if len(self.checkpoints) > self.max_checkpoints:
                oldest_checkpoint = self.checkpoints.pop(0)
                if oldest_checkpoint.exists():
                    os.remove(oldest_checkpoint)

            if self.verbose > 0:
                print(f"Checkpoint saved at step {self.num_timesteps}")

        return True

class ParamEncoder(nn.Module):
    def __init__(self, param_dim, latent_dim):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(param_dim, 32),
            nn.ReLU(),
            nn.Linear(32, latent_dim),
        )

    def forward(self, params):
        return self.encoder(params)

class CustomLogger(Logger):
    def __init__(self, folder: Optional[str], output_formats: list[KVWriter]):
        super(CustomLogger, self).__init__(folder, output_formats)

    def dump(self, step: int = 0) -> None:
        """
        Write all of the diagnostics from the current iteration without clearing the stored values.
        """
        if self.level == DISABLED:
            return
        for _format in self.output_formats:
            if isinstance(_format, KVWriter):
                _format.write(self.name_to_value, self.name_to_excluded, step)

class PerStepStateRecorderCallback(BaseCallback):
    """
    Records the individual state components as scalars at EVERY STEP 
    and forces the logger to dump the data immediately.
    """
    
    def __init__(self, state_names: list[str], verbose: int = 0):
        super().__init__(verbose)
        self.state_names = state_names
        
    def _on_step(self) -> bool:
        # self.locals['new_obs'] holds the observation after env.step().
        # Assuming a single environment (VecEnv with n_envs=1), we take the first element.
        state = self.locals['new_obs'][0]
        reward = self.locals['rewards'][0]
        
        for name, value in zip(self.state_names, state):
            # Record the key-value pair. We use the 'state_per_step/' prefix
            # to make them easy to find in W&B/TensorBoard.
            self.logger.record(f"state_per_step/{name}", value.item())
            
        self.logger.record(f"state_per_step/reward", reward)
        # Flush every step: SB3 only dumps at the end of a rollout by
        # default, which is too coarse for per-step state logging.
        if self.logger:
            # self.num_timesteps is the global timestep counter
            self.logger.dump(self.num_timesteps) 

        return True

class PerformanceUpdateFixedIntervalCallback(BaseCallback):

    def __init__(self, total_timesteps):
        super().__init__()
        self.total_timesteps = total_timesteps

    def _on_step(self):

        if self.n_calls % (self.total_timesteps // 5) == 0:
           self.training_env.env_method('update_noise')
        return True

class PerformanceUpdateCallback(BaseCallback):

    def __init__(self, update_freq=2000, reward_threshold=480, wait_steps=12000):
        super().__init__()
        self.update_freq = update_freq
        self.max_rew = reward_threshold
        self.wait_steps = wait_steps
        self.steps_since_high_perf = 0  # Counter to track waiting steps
        self.high_perf_threshold = 80  # Performance threshold
        self.first_exceed = True  # Flag to check if we are still sending values before exceeding threshold

    def _on_step(self):
        if self.n_calls % self.update_freq == 0:
            ep_rew_mean = self.logger.name_to_value['rollout/ep_rew_mean']
            if ep_rew_mean > 0:
                perf = ep_rew_mean / self.max_rew * 100

                if self.first_exceed:
                    # Always send performance until it exceeds the threshold
                    self.training_env.env_method('set_perf', perf)

                    # Check if we have exceeded the threshold
                    if perf > self.high_perf_threshold:
                        self.first_exceed = False  # Stop sending continuously
                        self.steps_since_high_perf = 0  # Reset the waiting counter

                else:
                    # We have exceeded the threshold, now we wait
                    self.steps_since_high_perf += self.update_freq

                    # Allow performance update after waiting period
                    if self.steps_since_high_perf >= self.wait_steps:
                        # Reset back to sending performance values
                        self.training_env.env_method('set_perf', perf)
                        self.steps_since_high_perf = 0  # Reset the waiting counter
                        self.first_exceed = True  # Go back to the first state

        return True


def seed_everything(env, seed, cudnn_deterministic):
    # got this from cleanrl
    env.action_space.seed(seed)
    env.observation_space.seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = cudnn_deterministic
    env.reset(seed=seed)


# def download_artifact_file(artifact_alias, filename):
#     """Download artifact and returns path to filename.
# 
#     :param artifact_name: wandb artifact alias
#     :param filename: filename in the artifact
#     """
#     logging.info(f"loading {filename} from {artifact_alias}")
# 
#     artifact = wandb.use_artifact(artifact_alias)
#     artifact_dir = Path(artifact.download())
#     filepath = artifact_dir / filename
# 
#     assert filepath.is_file(), f"{artifact_alias} doesn't contain {filename}"
# 
#     return filepath
# 
# 
# def upload_file_to_artifacts(pth, artifact_name, artifact_type):
#     logging.info(f"Saving {pth} to {artifact_name}")
#     if not isinstance(pth, Path):
#         pth = Path(pth)
# 
#     assert os.path.isfile(pth), f"{pth} is not a file"
# 
#     artifact = wandb.Artifact(artifact_name, type=artifact_type)
#     artifact.add_file(pth)
#     wandb.log_artifact(artifact)


"""
Utility functions used for classic control environments.
"""



def verify_number_and_cast(x: SupportsFloat) -> float:
    """Verify parameter is a single number and cast to a float."""
    try:
        x = float(x)
    except (ValueError, TypeError) as e:
        raise ValueError(f"An option ({x}) could not be converted to a float.") from e
    return x


def maybe_parse_reset_bounds(
    options: Optional[dict], default_low: float, default_high: float
) -> Tuple[float, float]:
    """
    This function can be called during a reset() to customize the sampling
    ranges for setting the initial state distributions.

    Args:
      options: Options passed in to reset().
      default_low: Default lower limit to use, if none specified in options.
      default_high: Default upper limit to use, if none specified in options.

    Returns:
      Tuple of the lower and upper limits.
    """
    if options is None:
        return default_low, default_high

    low = options.get("low") if "low" in options else default_low
    high = options.get("high") if "high" in options else default_high

    # We expect only numerical inputs.
    low = verify_number_and_cast(low)
    high = verify_number_and_cast(high)
    if low > high:
        raise ValueError(
            f"Lower bound ({low}) must be lower than higher bound ({high})."
        )

    return low, high