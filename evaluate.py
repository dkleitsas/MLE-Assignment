import numpy as np
from stable_baselines3 import DQN, PPO
from env import GridWorldEnv

N_EPISODES = 100
SEED = 42
MODEL_TYPES = ["DQN", "PPO"]

for model_type in MODEL_TYPES:
    model_path = f"./models/{model_type.lower()}_best_model/best_model.zip"
    
    print(f"Evaluating {model_type} Model")
    
    if model_type == "DQN":
        model = DQN.load(model_path)
    elif model_type == "PPO":
        model = PPO.load(model_path)
    
    env = GridWorldEnv(m=6, n=6, k=5, render_mode=None, max_steps=30)
    
    episode_steps = []
    episode_rewards = []
    successes = 0
    
    print(f"\nEvaluating over {N_EPISODES} episodes...")
    
    for episode in range(N_EPISODES):
        obs, info = env.reset(seed=SEED + episode)
        terminated = False
        truncated = False
        steps = 0
        total_reward = 0.0
        
        while not (terminated or truncated):
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(int(action))
            total_reward += reward
            steps += 1
        
        episode_steps.append(steps)
        episode_rewards.append(total_reward)
        
        if terminated and not truncated and reward == 1.0:
            successes += 1
        
        if (episode + 1) % 10 == 0:
            print(f"  Episodes completed: {episode + 1}/{N_EPISODES}")
    
    avg_steps = np.mean(episode_steps)
    avg_reward = np.mean(episode_rewards)
    success_rate = (successes / N_EPISODES) * 100
    
    print(f"Average Steps per Episode:   {avg_steps:.2f}")
    print(f"Average Reward per Episode:  {avg_reward:.4f}")
    print(f"Success Rate:                {success_rate:.2f}%")
    print()

