import gymnasium
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecNormalize
from stable_baselines3.common.env_checker import check_env

from env import GridWorldEnv

env = GridWorldEnv(m = 6, n = 6, k = 5, render_mode="human")

model = PPO(
    policy="MultiInputPolicy",
    env=env,
    learning_rate=3e-4,
    n_steps=512,
    batch_size=64,
    n_epochs=3,
    gamma=0.99,
    gae_lambda=0.95,
    clip_range=0.2,
    ent_coef=0.01,
    verbose=0,
)

model.learn(total_timesteps=20_000)