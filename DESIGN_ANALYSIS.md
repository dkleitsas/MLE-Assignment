# Design Analysis: GridWorld Reinforcement Learning

This document provides a detailed analysis of the design choices, algorithm comparison, and challenges encountered during the development of this reinforcement learning project.


## Environment Design

### State Representation

The environment uses a **multi-input observation space** (`gym.spaces.Dict`) with four components:

#### 1. Agent Position (`agent`)
- **Type**: 2D integer coordinates `[x, y]`
- **Range**: `[0, m-1] × [0, n-1]` where m=6, n=6
- **Use**: Provides the agent's current location in the grid. Essential for the agent to understand its position relative to the target and obstacles.

#### 2. Target Position (`target`)
- **Type**: 2D integer coordinates `[x, y]`
- **Range**: `[0, m-1] × [0, n-1]`
- **Use**: Enables the agent to learn goal-directed behavior. With both agent and target positions, the agent can compute implicit direction and distance to the goal.

#### 3. Obstacle Positions (`obstacles`)
- **Type**: k×2 integer array
- **Range**: Each obstacle at `[0, m-1] × [0, n-1]`
- **Use**: Provides complete environmental awareness. The agent knows obstacle locations and can learn to plan paths that avoid them.

#### 4. Edge Detection Flags (`at_edge`)
- **Type**: 4D binary vector `[at_right, at_top, at_left, at_bottom]`
- **Range**: Each element in {0, 1}
- **Use**: Helps the agent quickly identify when it's at grid boundaries without requiring additional computation. This feature engineering reduces the learning burden by making boundary conditions explicit.



### Reward Function


#### Terminal Rewards
```python
+1.0  # Successfully reaching the target
-1.0  # Collision with obstacle/wall OR exceeding max steps
```

#### Intermediate Rewards (Dense Reward Shaping)
```python
reward = (previous_distance - current_distance) * 0.1 - 0.01
```

### Reward Function Analysis

**Positive Components**:
1. **Goal Achievement Reward (+1.0)**: Strong positive signal for task completion
2. **Distance-Based Shaping (+0.1 × Δdistance)**: Encourages progress toward the goal
   - Uses BFS-computed shortest path distance (considers obstacles)
   - Magnitude of 0.1 balances guidance without overwhelming terminal reward

**Negative Components**:
1. **Collision Penalty (-1.0)**: Strong disincentive for invalid moves
2. **Time Step Penalty (-0.01)**: Encourages efficiency, prevents infinite exploration

### How Reward Design Affects Learning

#### Advantages

1. **Faster Convergence**: Dense rewards (distance-based shaping) provide learning signal at every step, not just at episode end. This is crucial for a relatively large state space (6×6 grid with moving obstacles).

2. **Exploration Efficiency**: The distance-based component guides exploration toward the goal rather than relying purely on random exploration with sparse rewards.

3. **Path Optimization**: The small time penalty (-0.01) encourages the agent to find efficient paths, not just successful ones.

4. **Obstacle Avoidance**: Strong collision penalty (-1.0) quickly teaches the agent to avoid obstacles.

### Additional Environment Guarantees

**Solvability Check**: Every episode is guaranteed to have a valid solution path
```python
def _is_grid_solvable(self, agent_pos, target_pos, obstacles):
    # BFS to verify path exists
```

**Impact**: 
- Eliminates impossible episodes that could confuse learning
- Ensures reward function is always meaningful

---

## Algorithm Comparison

### Overview

Two algorithms were trained and compared:
- **PPO (Proximal Policy Optimization)**: On-policy, policy gradient method
- **DQN (Deep Q-Network)**: Off-policy, value-based method

### Performance Results

Based on evaluation across 100 episodes per model:

| Metric | DQN (Off-Policy) | PPO (On-Policy) |
|--------|------------------|-----------------|
| **Success Rate** | ~83% | ~88% |
| **Learning Stability** | More variable | Smoother convergence |
| **Plateauing Behavior** | Plateaus earlier | Continues improving longer |
| **Sample Efficiency** | Better early learning | Moderate early learning |
| **Final Performance** | Good | Better |


### Why PPO Performs Better

PPO performs better than DQN in this environment because its policy-gradient approach is more naturally equipped to handle complex, multi-input observations and frequent reward feedback. Since PPO is an on-policy method, it learns directly from its most recent experiences, updating its stochastic policy in a smooth and stable way through clipped adjustments. This lets it make effective use of the distance-based and time-step rewards, learning faster and assigning credit to actions more accurately. As a result, PPO converges more reliably and adapts more efficiently to the structured challenges of the environment.


### Sample Efficiency Comparison

**Early Training**:
- **DQN**: Shows faster initial learning due to experience replay (reusing samples)
- **PPO**: Slower initial progress as it only learns from fresh samples

**Mid Training**:
- Both algorithms show steady improvement
- PPO begins catching up due to more stable learning

**Late Training**:
- **PPO**: Continues improving, achieving higher final performance
- **DQN**: Improvement slows significantly, plateaus around 15M steps


---

## Challenges and Solutions

### Challenge 1: Ensuring Grid Solvability

**Problem**: 
Random obstacle placement could create unsolvable grids where no path exists from agent to target.


**Solution**:
Implemented BFS-based solvability check in `reset()`:
```python
def _is_grid_solvable(self, agent_pos, target_pos, obstacles):
    # Perform BFS to check if path exists
    # Retry grid generation until solvable configuration found
```

---

### Challenge 2: Sparse Rewards Leading to Slow Learning

**Initial Approach**:
Sparse rewards with only terminal feedback (+1 for goal, -1 for collision)

**Problem**:
Slow convergence as the agent needed to by pure luck stumble onto the correct square 

**Solution**:
Implemented reward shaping with BFS-computed distance:
```python
current_distance = self._compute_path_distance(agent, target)
distance_reward = (previous_distance - current_distance) * 0.1
reward = distance_reward - 0.01
```

### Challenge 3: Early Stopping Implementation

**Problem**:
Training for full 20M steps often wasteful as algoritmhs often converged earlier

**Solution**:
Implemented early stopping callback:
```python
class EarlyStoppingCallback:
    patience=10  # Stop if no improvement for 10 evaluations
```


