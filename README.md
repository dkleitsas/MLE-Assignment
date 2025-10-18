# Reinforcement Learning GridWorld Navigation

A reinforcement learning project implementing and comparing on-policy (PPO) and off-policy (DQN) algorithms to solve a custom GridWorld navigation task with obstacles.

## Project Overview

This project implements a custom GridWorld environment where an agent must navigate from a starting position to a target location while avoiding obstacles. The environment ensures that every generated grid is solvable by verifying path existence using BFS. Two reinforcement learning algorithms are trained and compared:

- **PPO (Proximal Policy Optimization)** - On-policy algorithm
- **DQN (Deep Q-Network)** - Off-policy algorithm

## Default Environment Specifications

- **Grid Size**: 6x6
- **Obstacles**: 5 randomly placed obstacles per episode
- **Max Steps**: 30 steps per episode
- **Actions**: 4 discrete actions (Up, Down, Left, Right)
- **State Space**: Multi-input observation containing agent position, target position, obstacle positions, and edge detection flags
- **Reward Structure**:
  - `+1.0` for reaching the target
  - `-1.0` for hitting an obstacle or wall, or exceeding max steps
  - `+0.1 * (previous_distance - current_distance) - 0.01` for intermediate steps


## Setup Instructions

1. **Clone the repository**
   ```bash
   git clone https://github.com/dkleitsas/MLE-Assignment
   cd MLE-Assignment
   ```

2. **Create and activate a virtual environment**
   ```bash
   python -m venv .venv
   
   # On Windows
   .venv\Scripts\activate
   
   # On macOS/Linux
   source .venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

## Training the Agent

To train both PPO and DQN models:

```bash
python train.py
```

**Training Details**:
- Both algorithms are trained sequentially
- Training runs for up to 20 million timesteps per algorithm
- Uses 8 parallel environments for faster training
- Early stopping with patience of 10 evaluations (no improvement)
- Evaluations every 50,000 steps with 1,000 episodes
- Best models are automatically saved to `models/` directory
- Training logs are saved to `logs/` directory


## Evaluating Trained Models

To evaluate both trained models:

```bash
python evaluate.py
```

This script:
- Loads the best saved models for both DQN and PPO
- Runs 100 evaluation episodes per model with deterministic policy
- Reports average steps, average reward, and success rate


---

## Docker Setup (Optional)

As an alternative to local installation, you can use Docker to run the project in a containerized environment.


### Installation

1. **Clone the repository** (if you haven't already)
   ```bash
   git clone <repository-url>
   cd MLE-Assignment
   ```

2. **Build the Docker image**
   ```bash
   docker-compose build
   ```

### Training with Docker

To train models using Docker:

```bash
docker-compose run --rm mle-assignment python train.py
```

### Evaluation with Docker

To evaluate models using Docker:

```bash
docker-compose run --rm mle-assignment python evaluate.py
```
