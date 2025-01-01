import gymnasium as gym
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv
from typing import Callable


def make_env(env_fn: Callable, rank: int, seed: int = 0) -> Callable:
    """
    Create a function that will create and wrap an environment with the correct rank.
    """

    def _init() -> gym.Env:
        env = env_fn()
        env.reset(seed=seed + rank)
        return env

    return _init


def create_vec_env(
    env_fn: Callable, n_envs: int, seed: int = 0, use_subprocess: bool = True
) -> SubprocVecEnv:
    """
    Create a vectorized environment that runs multiple environments in parallel.

    Args:
        env_fn: Function that creates a single environment
        n_envs: Number of parallel environments to create
        seed: Base seed for the environments
        use_subprocess: If True, use SubprocVecEnv (parallel), if False use DummyVecEnv (serial)
    """
    env_fns = [make_env(env_fn, i, seed) for i in range(n_envs)]

    if use_subprocess:
        return SubprocVecEnv(env_fns)  # Parallel execution
    else:
        return DummyVecEnv(env_fns)  # Serial execution (for debugging)
