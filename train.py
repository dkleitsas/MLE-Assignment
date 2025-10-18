import numpy as np
import random

from stable_baselines3 import PPO, DQN
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback

from env import GridWorldEnv

SEED = 42
random.seed(SEED)
np.random.seed(SEED)

model_type = "DQN"  # DQN or PPO


class EarlyStoppingCallback(BaseCallback):
    def __init__(self, patience: int = 5, verbose: int = 1):
        super().__init__(verbose)
        self.patience = patience
        self.best_mean_reward = -np.inf
        self.evaluations_without_improvement = 0

    def _on_step(self) -> bool:
        continue_training = True

        current_mean_reward = float(self.parent.last_mean_reward)

        if current_mean_reward > self.best_mean_reward:
            if self.verbose > 0:
                print(
                    f"New best mean reward: {current_mean_reward:.2f} (previous: {self.best_mean_reward:.2f})"
                )
            self.best_mean_reward = current_mean_reward
            self.evaluations_without_improvement = 0
        else:
            self.evaluations_without_improvement += 1
            if self.verbose > 0:
                print(
                    f"No improvement for {self.evaluations_without_improvement}/{self.patience} evaluations"
                )

            if self.evaluations_without_improvement >= self.patience:
                if self.verbose > 0:
                    print(
                        f"Stopping training: no improvement for {self.patience} evaluations"
                    )
                continue_training = False

        return continue_training


env = make_vec_env(lambda: GridWorldEnv(m=6, n=6, k=5, render_mode=None, max_steps=30), n_envs=8, seed=SEED)
eval_env = Monitor(GridWorldEnv(m=6, n=6, k=5, render_mode=None, max_steps=30))
eval_env.reset(seed=SEED)

early_stop_callback = EarlyStoppingCallback(patience=10, verbose=1)

eval_callback = EvalCallback(
    eval_env,
    callback_after_eval=early_stop_callback,
    eval_freq=50_000,
    n_eval_episodes=1000,
    log_path="./logs",
    best_model_save_path=f"./models/{model_type.lower()}_best_model",
    verbose=1,
)


if model_type == "PPO":

    model = PPO(
        policy="MultiInputPolicy",
        env=env,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=256,
        n_epochs=4,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        verbose=0,
        seed=SEED,
    )

    model.learn(total_timesteps=20_000_000, callback=eval_callback)

elif model_type == "DQN":

    model = DQN(
        policy="MultiInputPolicy",
        env=env,
        learning_rate=3e-4,
        buffer_size=100000,
        learning_starts=10000,
        batch_size=256,
        gamma=0.99,
        train_freq=4,
        target_update_interval=1000,
        exploration_fraction=0.1,
        exploration_final_eps=0.05,
        verbose=0,
        seed=SEED,
    )

    model.learn(total_timesteps=20_000_000, callback=eval_callback)