import gymnasium as gym
import numpy as np
from collections import deque

class GridWorldEnv(gym.Env):

    def __init__(self, m: int = 6, n: int = 6, k: int = 5, render_mode: str = None, max_steps: int = 100):

        self.grid_dimensions = (m, n)
        self.num_obstacles = k
        self.render_mode = render_mode
        self.max_steps = max_steps
        self.step_count = 0

        self._agent_location = np.array([-1, -1], dtype=np.int32)
        self._target_location = np.array([-1, -1], dtype=np.int32)
        self._obstacle_locations = np.array([[-1, -1]] * k, dtype=np.int32)
        self._previous_distance = 0

        self.observation_space = gym.spaces.Dict(
            {
                "agent": gym.spaces.Box(
                    low=np.array([0, 0]),
                    high=np.array([m - 1, n - 1]),
                    shape=(2,),
                    dtype=np.int32
                ),
                "target": gym.spaces.Box(
                    low=np.array([0, 0]),
                    high=np.array([m - 1, n - 1]),
                    shape=(2,),
                    dtype=np.int32
                ),
                "obstacles": gym.spaces.Box(
                    low=np.array([[0, 0]] * k),
                    high=np.array([[m - 1, n - 1]] * k),
                    shape=(k, 2),
                    dtype=np.int32
                ),
                "at_edge": gym.spaces.Box(
                    low=0,
                    high=1,
                    shape=(4,),
                    dtype=np.int32
                ),
            }
        )

        self.action_space = gym.spaces.Discrete(4)

        self._action_to_direction = {
            0: np.array([1, 0]),   # RIGHT
            1: np.array([0, 1]),   # UP
            2: np.array([-1, 0]),  # LEFT
            3: np.array([0, -1]),  # DOWN
        }


    def _get_obs(self):
        x, y = self._agent_location
        m, n = self.grid_dimensions
        at_edge = np.array([
            int(x == m - 1),
            int(y == n - 1),
            int(x == 0),
            int(y == 0)
        ], dtype=np.int32)
        
        return {
            "agent": self._agent_location, 
            "target": self._target_location, 
            "obstacles": self._obstacle_locations,
            "at_edge": at_edge
        }
    

    def _is_grid_solvable(self, agent_pos, target_pos, obstacles):
        m, n = self.grid_dimensions
        grid = np.zeros((m, n), dtype=int)
        for ox, oy in obstacles:
            grid[ox, oy] = 1

        queue = deque([tuple(agent_pos)])
        visited = set(queue)

        directions = [(1,0), (-1,0), (0,1), (0,-1)]
        while queue:
            x, y = queue.popleft()
            if (x, y) == tuple(target_pos):
                return True
            for dx, dy in directions:
                nx, ny = x + dx, y + dy
                if 0 <= nx < m and 0 <= ny < n and grid[nx, ny] == 0 and (nx, ny) not in visited:
                    visited.add((nx, ny))
                    queue.append((nx, ny))
        return False


    def _compute_path_distance(self, agent_pos, target_pos):
        m, n = self.grid_dimensions
        grid = np.zeros((m, n), dtype=int)
        for ox, oy in self._obstacle_locations:
            grid[ox, oy] = 1

        queue = deque([(tuple(agent_pos), 0)])
        visited = set([tuple(agent_pos)])

        directions = [(1,0), (-1,0), (0,1), (0,-1)]
        while queue:
            (x, y), dist = queue.popleft()
            if (x, y) == tuple(target_pos):
                return dist
            for dx, dy in directions:
                nx, ny = x + dx, y + dy
                if 0 <= nx < m and 0 <= ny < n and grid[nx, ny] == 0 and (nx, ny) not in visited:
                    visited.add((nx, ny))
                    queue.append(((nx, ny), dist + 1))
        return float('inf')


    def reset(self, seed: int = None, options: dict = None):
        if seed is not None:
            self.np_random = np.random.default_rng(seed)

        self.step_count = 0
        occupied_positions = []

        def random_empty_pos():
            while True:
                pos = tuple(self.np_random.integers([0, 0], [self.grid_dimensions[0], self.grid_dimensions[1]], size=2, dtype=int))
                if pos not in occupied_positions:
                    occupied_positions.append(pos)
                    return np.array(pos, dtype=int)
        
        while True:
            self._agent_location = random_empty_pos()

            self._target_location = random_empty_pos()

            obstacle_positions = [random_empty_pos() for _ in range(self.num_obstacles)]
            self._obstacle_locations = np.array(obstacle_positions, dtype=int)

            if self._is_grid_solvable(self._agent_location, self._target_location, self._obstacle_locations):
                break

        self._previous_distance = self._compute_path_distance(self._agent_location, self._target_location)

        obs = self._get_obs()
        info = {}
        return obs, info


    def step(self, action: int):
        self.step_count += 1
        direction = self._action_to_direction[action]
        terminated = False
        reward = 0.0

        def detect_collision(new_pos):
            if any(np.array_equal(new_pos, obs) for obs in self._obstacle_locations):
                return True
            if not (0 <= new_pos[0] < self.grid_dimensions[0]) or not (0 <= new_pos[1] < self.grid_dimensions[1]):
                return True
            return False

        if not detect_collision(self._agent_location + direction):
            self._agent_location += direction
        else:
            terminated = True   

        truncated = self.step_count >= self.max_steps

        if terminated or truncated:
            reward = -1.0
        elif np.array_equal(self._agent_location, self._target_location):
            reward = 1.0
            terminated = True
        else:
            current_distance = self._compute_path_distance(self._agent_location, self._target_location)
            distance_reward = (self._previous_distance - current_distance) * 0.1
            reward = distance_reward - 0.01
            self._previous_distance = current_distance

        observation = self._get_obs()
        info = {}

        return observation, reward, terminated, truncated, info

   
    def render(self):
        if self.render_mode == "human":
            for y in range(self.grid_dimensions[1] - 1, -1, -1):
                row = ""
                for x in range(self.grid_dimensions[0]):
                    if np.array_equal([x, y], self._agent_location):
                        row += "A "
                    elif np.array_equal([x, y], self._target_location):
                        row += "T "
                    elif np.any(np.all(self._obstacle_locations == [x, y], axis=1)):
                        row += "x "
                    else:
                        row += "O "
                print(row)
            print()
