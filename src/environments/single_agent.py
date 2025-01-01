import gymnasium as gym
import numpy as np
from gymnasium import spaces
from typing import Tuple, Dict, Any, Optional


class SnakeEnv(gym.Env):
    """Single-agent Snake environment."""

    metadata = {"render_modes": ["console"]}

    def __init__(self, grid_size: int = 10, max_steps: int = 100):
        super().__init__()

        self.grid_size = grid_size
        self.max_steps = max_steps

        # Action space: 0: up, 1: right, 2: down, 3: left
        self.action_space = spaces.Discrete(4)

        # Observation space: grid_size x grid_size with channels
        # Channel 1: Snake body (1 for snake body, 0 otherwise)
        # Channel 2: Snake head (1 for head, 0 otherwise)
        # Channel 3: Food (1 for food, 0 otherwise)
        self.observation_space = spaces.Box(
            low=0, high=1, shape=(3, grid_size, grid_size), dtype=np.float32
        )

        # Initialize other attributes
        self.snake = []
        self.food = None
        self.steps = 0

    def reset(self, seed: Optional[int] = None) -> Tuple[np.ndarray, Dict[str, Any]]:
        super().reset(seed=seed)

        # Initialize snake at the center
        center = self.grid_size // 2
        self.snake = [(center, center)]

        # Place food randomly
        self._place_food()

        # Reset steps
        self.steps = 0

        return self._get_obs(), {}

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """Execute one step in the environment.

        Takes an action and advances the environment by one timestep. Handles snake movement,
        collision detection, food collection, and termination conditions.

        Args:
            action (int): The action to take (0: up, 1: right, 2: down, 3: left)

        Returns:
            Tuple containing:
                - np.ndarray: The new observation after taking the action
                - float: The reward received (-1.0 for collision, 1.0 for food, 0.0 otherwise)
                - bool: Whether the episode has terminated
                - bool: Whether the episode was truncated (always False for this env)
                - Dict[str,Any]: Additional info (contains 'reason' on collision or 'steps')
        """
        self.steps += 1

        # Get new head position
        old_head = self.snake[0]
        new_head = self._get_new_head(old_head, action)

        # Check if game is over (wall collision or self collision)
        if self._is_collision(new_head):
            return self._get_obs(), -5.0, True, False, {"reason": "collision"}

        # Move snake
        self.snake.insert(0, new_head)

        # Check if food was eaten
        reward = 0.0
        if new_head == self.food:
            reward = 1.0
            self._place_food()
        else:
            self.snake.pop()

        # Check if max steps reached
        done = self.steps >= self.max_steps

        return self._get_obs(), reward, done, False, {"steps": self.steps}

    def render(self):
        """Simple console rendering."""
        grid = np.full((self.grid_size, self.grid_size), ".", dtype=str)

        # Render snake body
        for segment in self.snake[1:]:
            grid[segment[0], segment[1]] = "o"

        # Render snake head
        head = self.snake[0]
        grid[head[0], head[1]] = "H"

        # Render food
        grid[self.food[0], self.food[1]] = "F"

        # Print grid
        for row in grid:
            print(" ".join(row))
        print("\n")

    def _get_new_head(self, old_head: Tuple[int, int], action: int) -> Tuple[int, int]:
        """Calculate new head position based on action."""
        if action == 0:  # up
            return (old_head[0] - 1, old_head[1])
        elif action == 1:  # right
            return (old_head[0], old_head[1] + 1)
        elif action == 2:  # down
            return (old_head[0] + 1, old_head[1])
        else:  # left
            return (old_head[0], old_head[1] - 1)

    def _is_collision(self, position: Tuple[int, int]) -> bool:
        """Check if position collides with wall or snake body."""
        return (
            position[0] < 0
            or position[0] >= self.grid_size
            or position[1] < 0
            or position[1] >= self.grid_size
            or position in self.snake
        )

    def _place_food(self) -> None:
        """Place food in random empty position."""
        while True:
            food = (
                self.np_random.integers(0, self.grid_size),
                self.np_random.integers(0, self.grid_size),
            )
            if food not in self.snake:
                self.food = food
                break

    def _get_obs(self) -> np.ndarray:
        """Convert game state to observation."""
        obs = np.zeros((3, self.grid_size, self.grid_size), dtype=np.float32)

        # Place snake body
        for segment in self.snake[1:]:
            obs[0, segment[0], segment[1]] = 1

        # Place snake head
        head = self.snake[0]
        obs[1, head[0], head[1]] = 1

        # Place food
        obs[2, self.food[0], self.food[1]] = 1

        return obs
