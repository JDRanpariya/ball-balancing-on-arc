from gymnasium.envs.registration import register

register(
    id="BalancerSim-v0",
    entry_point="balancer.envs.sim:BalancerSim",
    max_episode_steps=200,        # legacy - short episodes
    reward_threshold=175.0,       # legacy
)

register(
    id="BalancerSim-v1",
    entry_point="balancer.envs.sim:BalancerSim",
    max_episode_steps=600,        # 30s at 20Hz
    reward_threshold=475.0,       # for gaussian/regular rewards
)

register(
    id="BalancerSim-v2",
    entry_point="balancer.envs.sim:BalancerSim",
    max_episode_steps=1200,       # 60s at 20Hz (2× the 30 s FAIL_TIME used for hardware trials)
    reward_threshold=950.0,       # for gaussian/regular (60s × ~0.95)
)
