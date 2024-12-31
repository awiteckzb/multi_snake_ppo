import numpy as np
from typing import List, Tuple
import torch


class PPOMemory:
    """Memory buffer for storing a single trajectory segment of length T"""

    def __init__(self, n_actors: int, timesteps_per_actor: int):
        """Initialize empty memory to store NxT timesteps."""
        self.n_actors = n_actors
        self.timesteps_per_actor = timesteps_per_actor
        self.max_size = n_actors * timesteps_per_actor
        self.clear()

    def clear(self) -> None:
        """Clear memory."""
        self.states: List[np.ndarray] = []
        self.actions: List[int] = []
        self.rewards: List[float] = []
        self.values: List[float] = []
        self.logprobs: List[float] = []
        self.actor_ids: List[int] = []  # Track which actor each transition came from
        self.is_full = False

    def store(
        self,
        actor_id: int,
        state: np.ndarray,
        action: int,
        reward: float,
        value: float,
        logprob: float,
    ) -> bool:
        """
        Store a transition. Return True if buffer becomes full.
        """
        if len(self.states) >= self.max_size:
            return True

        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.values.append(value)
        self.logprobs.append(logprob)
        self.actor_ids.append(actor_id)
        self.is_full = len(self.states) >= self.max_size
        return self.is_full

    def compute_advantages(
        self, gamma: float = 0.99, gae_lambda: float = 0.95
    ) -> np.ndarray:
        """
        Compute advantages separately for each actor's trajectory.
        """
        advantages = np.zeros(self.max_size)

        # Compute advantages separately for each actor
        for actor in range(self.n_actors):
            # Get indices for this actor's transitions
            actor_indices = np.array(
                [i for i, aid in enumerate(self.actor_ids) if aid == actor]
            )

            if len(actor_indices) != self.timesteps_per_actor:
                raise ValueError(
                    f"Actor {actor} has {len(actor_indices)} transitions "
                    f"but expected {self.timesteps_per_actor}"
                )

            # Get this actor's data
            rewards = np.array([self.rewards[i] for i in actor_indices])
            values = np.array([self.values[i] for i in actor_indices])

            # Calculate advantages for this actor
            deltas = rewards[:-1] + gamma * values[1:] - values[:-1]

            # Compute GAE
            gae = 0
            for t in reversed(range(len(deltas))):
                gae = deltas[t] + gamma * gae_lambda * gae
                advantages[actor_indices[t]] = gae

            # Handle last timestep for this actor
            advantages[actor_indices[-1]] = rewards[-1] - values[-1]

        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        return advantages

    def get_batch(self, device: torch.device) -> Tuple:
        """Convert entire trajectory to torch tensors for training.

        This method processes the stored trajectory data and converts it to PyTorch tensors
        for PPO training. It computes advantages using GAE, converts numpy arrays to tensors,
        and moves them to the specified device.

        Args:
            device (torch.device): The device (CPU/GPU) to move the tensors to

        Returns:
            Tuple containing:
                - states (torch.FloatTensor): Batch of environment states
                - actions (torch.LongTensor): Batch of actions taken
                - old_logprobs (torch.FloatTensor): Log probabilities of actions under old policy
                - advantages (torch.FloatTensor): Computed advantages using GAE
                - returns (torch.FloatTensor): Computed returns (advantages + value estimates)
        """
        advantages = self.compute_advantages()

        states = torch.FloatTensor(np.array(self.states)).to(device)
        actions = torch.LongTensor(np.array(self.actions)).to(device)
        old_logprobs = torch.FloatTensor(np.array(self.logprobs)).to(device)
        advantages = torch.FloatTensor(advantages).to(device)

        # Returns = advantages + values
        returns = advantages + torch.FloatTensor(self.values).to(device)

        return states, actions, old_logprobs, advantages, returns

    def __len__(self) -> int:
        """Return current size of memory."""
        return len(self.states)
