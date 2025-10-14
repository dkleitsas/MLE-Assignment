import gymnasium as gym
import numpy as np

class GridWorldEnv(gym.Env):

    def __init__(self, m: int = 6, n: int = 6, k: int = 5, render_mode: str = None):

        self.grid_dimensions = (m, n)
        self.num_obstacles = k
        self.render_mode = render_mode

        self._agent_location = np.array([-1, -1], dtype=np.int32)
        self._target_location = np.array([-1, -1], dtype=np.int32)
        self._obstacle_locations = np.array([[-1, -1]] * k, dtype=np.int32)

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
        return {"agent": self._agent_location, "target": self._target_location, "obstacles": self._obstacle_locations}
    

    def reset(self, seed: int = None, options: dict = None):
        if seed is not None:
            self.np_random = np.random.default_rng(seed)

        occupied_positions = []

        def random_empty_pos():
            while True:
                pos = tuple(self.np_random.integers([0, 0], [self.grid_dimensions[0], self.grid_dimensions[1]], size=2, dtype=int))
                if pos not in occupied_positions:
                    occupied_positions.append(pos)
                    return np.array(pos, dtype=int)

        self._agent_location = random_empty_pos()

        self._target_location = random_empty_pos()

        obstacle_positions = [random_empty_pos() for _ in range(self.num_obstacles)]
        self._obstacle_locations = np.array(obstacle_positions, dtype=int)

        obs = self._get_obs()
        info = {}
        return obs, info


    def step(self, action: int):
        direction = self._action_to_direction[action]
        failed = False
        terminated = False

        def detect_collision(new_pos):
            if any(np.array_equal(new_pos, obs) for obs in self._obstacle_locations):
                return True
            if not (0 <= new_pos[0] < self.grid_dimensions[0]) or not (0 <= new_pos[1] < self.grid_dimensions[1]):
                return True
            return False

        if not detect_collision(self._agent_location + direction):
            self._agent_location += direction
        else:
            failed = True   

        if failed:
            reward = -1.0
            terminated = True
        elif np.array_equal(self._agent_location, self._target_location):
            reward = 1.0
            terminated = True

        truncated = False

        if not terminated:
            reward = -0.02

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
