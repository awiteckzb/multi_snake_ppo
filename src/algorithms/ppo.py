import torch
from torch.optim import Adam
from typing import Tuple, Dict
import numpy as np
import gymnasium as gym
from src.algorithms.networks import ActorCritic
from src.algorithms.memory import PPOMemory
from src.common.config import ConfigManager


class PPO:
    def __init__(
        self, env: gym.Env, config_manager: ConfigManager, device: str = "cpu"
    ):
        self.env = env
        self.config = config_manager
        self.device = device

        # Load PPO params from config
        self.learning_rate = self.config.get("ppo.learning_rate")
        self.gamma = self.config.get("ppo.gamma")
        self.gae_lambda = self.config.get("ppo.gae_lambda")
        self.clip_epsilon = self.config.get("ppo.clip_epsilon")
        self.c1 = self.config.get("ppo.c1")  # Value loss coefficient
        self.c2 = self.config.get("ppo.c2")  # Entropy coefficient
        self.max_grad_norm = self.config.get("ppo.max_grad_norm")
        self.n_epochs = self.config.get("ppo.n_epochs")
        self.trajectory_length = self.config.get("ppo.n_steps")
        self.total_iterations = self.config.get("training.total_iterations")
        self.n_actors = self.config.get("ppo.n_actors", 8)  # N in the paper
        self.timesteps_per_actor = self.config.get("ppo.n_steps")  # T in the paper

        # Initialize actor-critic network
        self.actor_critic = ActorCritic(
            input_channels=env.observation_space.shape[0],
            grid_size=env.observation_space.shape[1],
            n_actions=env.action_space.n,
            hidden_size=self.config.get("network.hidden_size"),
        ).to(self.device)

        # Initialize optimizer
        self.optimizer = Adam(self.actor_critic.parameters(), lr=self.learning_rate)

        # Initialize memory
        self.memory = PPOMemory(
            n_actors=self.n_actors, timesteps_per_actor=self.timesteps_per_actor
        )

    def choose_action(self, state: np.ndarray) -> Tuple[int, float, float]:
        """Choose an action using the actor critic network based on the current state."""
        state = torch.tensor(state, dtype=torch.float32).to(self.device)
        action, value, action_log_prob = self.actor_critic.act(state)
        return action, value, action_log_prob

    def update(self) -> Dict[str, float]:
        """Update the actor-critic network using collected trajectory. Returns dict with loss metrics."""

        # Get all data from memory
        states, actions, old_logprobs, advantages, returns = self.memory.get_batch(
            self.device
        )

        # Track metrics
        metrics = {
            "policy_loss": 0.0,
            "value_loss": 0.0,
            "entropy": 0.0,
        }

        # Perform multiple epochs of updates
        for _ in range(self.n_epochs):
            # Get new action probabilities and values
            new_logprobs, state_values, entropy = self.actor_critic.evaluate_actions(
                states, actions
            )

            # Calculate the probability ratio. Use logprobs to avoid numerical instability
            ratios = torch.exp(new_logprobs - old_logprobs)

            # Calculate the surrogate objective (Equation 7)
            surr1 = ratios * advantages
            surr2 = (
                torch.clamp(ratios, 1 - self.clip_epsilon, 1 + self.clip_epsilon)
                * advantages
            )

            # Calculate losses
            # Policy loss (negative because we're doing gradient ascent)
            policy_loss = -torch.min(surr1, surr2).mean()

            # Value loss (MSE). The purpose of this loss is to ensure that the critic's
            # value estimates match the true returns. Otherwise the advantage estimates will
            # be inaccurate.
            value_loss = 0.5 * (returns - state_values.squeeze()).pow(2).mean()

            # Entropy loss (for exploration)
            entropy_loss = -entropy.mean()

            # Combine losses
            total_loss = policy_loss + self.c1 * value_loss + self.c2 * entropy_loss

            # Update networks
            self.optimizer.zero_grad()
            total_loss.backward()

            # Clip gradients
            torch.nn.utils.clip_grad_norm_(
                self.actor_critic.parameters(), self.max_grad_norm
            )

            self.optimizer.step()

            # Update metrics
            metrics["value_loss"] += value_loss.item() / self.n_epochs
            metrics["policy_loss"] += policy_loss.item() / self.n_epochs
            metrics["entropy"] += entropy.mean().item() / self.n_epochs

        # Clear memory after update
        self.memory.clear()

        return metrics

    def train(self) -> None:
        """
        PPO training loop following the paper's algorithm exactly:
        - Run N parallel actors for T timesteps each
        - Compute advantage estimates
        - Optimize surrogate L for K epochs with minibatch size M
        """
        # Initialize N environments
        envs = [
            self.env for _ in range(self.n_actors)
        ]  # In practice, might want to create separate env instances
        states = [env.reset()[0] for env in envs]

        # Initialize memory for NT timesteps
        for iteration in range(self.total_iterations):
            # Clear memory at the start of each iteration
            self.memory.clear()

            # Step 1: Run policy π_{θ old} in environment for T timesteps
            for t in range(self.timesteps_per_actor):
                # Collect step from each actor
                for actor_id in range(self.n_actors):
                    # Get action from current policy
                    state = states[actor_id]
                    action, value, log_prob = self.choose_action(state)

                    # Execute action in environment
                    next_state, reward, done, truncated, _ = envs[actor_id].step(action)

                    # Store transition
                    self.memory.store(
                        actor_id=actor_id,
                        state=state,
                        action=action,
                        reward=reward,
                        value=value,
                        logprob=log_prob,
                    )

                    # Update state (reset if necessary)
                    if done or truncated:
                        states[actor_id] = envs[actor_id].reset()[0]
                    else:
                        states[actor_id] = next_state

            # Step 3: Update policy for K epochs
            metrics = self.update()

            print(f"Iteration {iteration}")
            print(f"Policy Loss: {metrics['policy_loss']:.3f}")
            print(f"Value Loss: {metrics['value_loss']:.3f}")
            print(f"Entropy: {metrics['entropy']:.3f}")
            print("-" * 50)

    def save_checkpoint(self, path: str, metrics: Dict[str, float]) -> None:
        """Save model checkpoint and metrics."""
        checkpoint = {
            "model_state_dict": self.actor_critic.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "metrics": metrics,
            "config": self.config.config,  # Save full config
        }
        torch.save(checkpoint, path)

    def load_checkpoint(self, path: str) -> Dict[str, float]:
        """Load model checkpoint and return metrics."""
        checkpoint = torch.load(path, map_location=self.device)
        self.actor_critic.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        return checkpoint["metrics"]
