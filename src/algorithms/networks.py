import torch
import torch.nn as nn
from torch.distributions import Categorical
from typing import List, Tuple
import numpy as np


class CNNBase(nn.Module):
    """Base CNN for processing the grid. Output is then fed into actor and critic networks."""

    def __init__(
        self, input_channels: int, grid_size: int, cnn_channels: List[int] = [32, 64]
    ):
        super().__init__()

        layers = []
        in_channels = input_channels

        for out_channels in cnn_channels:
            layers.extend(
                [
                    nn.Conv2d(
                        in_channels, out_channels, kernel_size=3, stride=1, padding=1
                    ),
                    nn.ReLU(),
                    nn.BatchNorm2d(out_channels),
                ]
            )
            in_channels = out_channels

        self.cnn = nn.Sequential(*layers)

        # Calculate output size
        test_input = torch.zeros(1, input_channels, grid_size, grid_size)
        test_output = self.cnn(test_input)
        self.output_size = int(np.prod(test_output.shape[1:]))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.cnn(x)
        # x.size(0): Represents the batch size
        return x.view(x.size(0), -1)  # Flatten the output


class ActorCritic(nn.Module):
    """Combined actor and critic network."""

    def __init__(
        self, input_channels: int, grid_size: int, n_actions: int, hidden_size: int = 64
    ):
        super().__init__()

        self.cnn_base = CNNBase(input_channels, grid_size)

        # Actor is a simple MLP
        # Takes embedded state and outputs a distribution over actions
        # Logits in this case, will process them thru softmax later
        self.actor = nn.Sequential(
            nn.Linear(self.cnn_base.output_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, n_actions),
        )

        # Also a simple MLP
        # Takes embedded state and outputs a single value
        self.critic = nn.Sequential(
            nn.Linear(self.cnn_base.output_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1),
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        features = self.cnn_base(x)

        action_logits = self.actor(features)
        state_value = self.critic(features)

        # Process action logits with softmax
        action_probs = torch.softmax(action_logits, dim=-1)

        return action_probs, state_value

    def evaluate_actions(
        self, states: torch.Tensor, actions: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Evaluate actions for PPO training by computing log probabilities, values and entropy.

        This method is used during PPO training to evaluate previously taken actions and compute
        the components needed for the PPO loss function.

        Args:
            states (torch.Tensor): Batch of states from the collected trajectory
            actions (torch.Tensor): Batch of actions that were taken in those states

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: A tuple containing:
                - action_log_probs: Log probabilities of the actions under current policy
                - state_values: Value estimates for the states from critic
                - dist_entropy: Entropy of the action distribution for exploration bonus
        """
        action_probs, state_values = self.forward(states)
        dist = Categorical(action_probs)

        # Effectively calculating probability of taking the action given current state. Log prob is used mainly for stability and PPO ratio calculation. Instead of calculating prob directly (i.e. new_prob/old_prob) we use exp(log_new_prob - log_old_prob).
        action_log_probs = dist.log_prob(actions)

        # Entropy is used as a bonus in the PPO objective function, mainly to encourage exploration and prevent premature convergence.
        dist_entropy = dist.entropy()

        return action_log_probs, state_values, dist_entropy

    def act(self, state: torch.Tensor) -> Tuple[int, float, float]:
        """Choose an action based on the current state during trajectory collection.

        This method is used during the rollout phase of PPO to collect trajectories.
        It samples actions from the policy network and returns necessary values for
        PPO training.

        Args:
            state (torch.Tensor): The current environment state tensor

        Returns:
            Tuple[int, float, float]: A tuple containing:
                - action (int): The sampled action to take in the environment
                - value (float): The critic's value estimate for the current state
                - action_log_prob (float): Log probability of the sampled action,
                  used for computing PPO's probability ratio during training
        """

        with torch.no_grad():
            # Add batch dimension if not present
            if state.dim() == 3:
                state = state.unsqueeze(0)  # Add batch dimension

            action_probs, value = self(state)

        # Sample an actual action to take in the environment
        dist = Categorical(action_probs)
        action = dist.sample()

        # Store the log probability of that action (we'll need this for PPO's ratio). See equation
        action_log_prob = dist.log_prob(action)

        return (action.item(), value.item(), action_log_prob.item())

    def act_vectorized(
        self, states: torch.Tensor
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Choose actions for multiple states at once."""
        # print(
        #     f"States shape in act_vectorized: {states.shape}"
        # )  # Should be (n_envs, channels, height, width)

        # Get action probabilities and values
        action_probs, values = self(states)
        # print(
        #     f"Action probs shape: {action_probs.shape}"
        # )  # Should be (n_envs, n_actions)
        # print(f"Values shape: {values.shape}")  # Should be (n_envs, 1)

        # Sample actions from the distributions
        dist = Categorical(action_probs)
        actions = dist.sample()
        # print(f"Actions shape: {actions.shape}")  # Should be (n_envs,)
        log_probs = dist.log_prob(actions)
        # print(f"Log probs shape: {log_probs.shape}")  # Should be (n_envs,)

        return (
            actions.cpu().numpy(),
            values.squeeze().cpu().numpy(),
            log_probs.cpu().numpy(),
        )
