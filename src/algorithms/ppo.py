import torch
from torch.optim import Adam
from typing import Tuple, Dict, Callable
import numpy as np
import gymnasium as gym
import time

from src.algorithms.networks import ActorCritic
from src.algorithms.memory import PPOMemory
from src.common.config import ConfigManager
from src.environments.utils import create_vec_env


class PPO:
    def __init__(
        self, env_fn: Callable, config_manager: ConfigManager, device: str = "cpu"
    ):
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
        self.total_iterations = self.config.get("training.total_iterations")
        self.n_actors = self.config.get("ppo.n_actors", 8)  # N in the paper
        self.timesteps_per_actor = self.config.get("ppo.n_steps")  # T in the paper

        # Create vectorized environment
        self.env = create_vec_env(env_fn, self.n_actors)

        # Get a sample environment to initialize network
        sample_env = env_fn()
        self.actor_critic = ActorCritic(
            input_channels=sample_env.observation_space.shape[0],
            grid_size=sample_env.observation_space.shape[1],
            n_actions=sample_env.action_space.n,
            hidden_size=self.config.get("network.hidden_size"),
        ).to(self.device)
        sample_env.close()

        # Initialize optimizer
        self.optimizer = Adam(self.actor_critic.parameters(), lr=self.learning_rate)

        # Initialize memory
        self.memory = PPOMemory(
            n_actors=self.n_actors, timesteps_per_actor=self.timesteps_per_actor
        )

    def choose_action(self, state: np.ndarray) -> Tuple[int, float, float]:
        """Choose an action for a single state."""
        state = torch.FloatTensor(state).to(self.device)
        with torch.no_grad():
            action, value, log_prob = self.actor_critic.act(state)
        return action, value, log_prob

    def choose_actions(
        self, states: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Choose actions for all environments at once."""
        # print(f"States shape in choose_actions before conversion: {states.shape}")
        states = torch.FloatTensor(states).to(self.device)
        # print(f"States shape in choose_actions after conversion: {states.shape}")
        with torch.no_grad():
            actions, values, log_probs = self.actor_critic.act_vectorized(states)
        return actions, values, log_probs

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

        all_metrics = []
        # print("n_actors:", self.n_actors)
        # print("Number of envs in the vectorized env:", self.env.num_envs)
        obs_info = self.env.reset()
        # print("Raw from env.reset():", obs_info)
        # print("Type(obs_info) =", type(obs_info))
        # if isinstance(obs_info, (list, tuple)):
        #     print("obs_info[0].shape =", getattr(obs_info[0], "shape", None))

        # if obs_info is a numpy array, print its shape
        # if isinstance(obs_info, np.ndarray):
        #     print("Shape of obs_info:", obs_info.shape)

        # print("n_actors:", self.n_actors)
        # print("Number of envs in the vectorized env:", self.env.num_envs)
        states = obs_info

        # Initialize memory for NT timesteps
        for iteration in range(self.total_iterations):
            iteration_start_time = time.time()

            # Clear memory at the start of each iteration
            self.memory.clear()

            # Metrics for this iteration
            episode_rewards = np.zeros(self.n_actors)
            episode_lengths = np.zeros(self.n_actors)
            episode_counts = np.zeros(self.n_actors)
            action_counts = np.zeros((self.n_actors, 4))

            # Step 1: Run policy π_{θ old} in environment for T timesteps
            for t in range(self.timesteps_per_actor):

                # Get actions for all environments at once
                actions, values, log_probs = self.choose_actions(states)

                # Track action distribution
                for i, action in enumerate(actions):
                    action_counts[i][action] += 1

                # Step all environments at once
                next_states, rewards, dones, _ = self.env.step(actions)

                # Update metrics
                episode_rewards += rewards
                episode_lengths += 1

                # Store transitions
                for i in range(self.n_actors):
                    self.memory.store(
                        actor_id=i,
                        state=states[i],
                        action=actions[i],
                        reward=rewards[i],
                        value=values[i],
                        logprob=log_probs[i],
                    )

                # Handle episode completion
                done_envs = dones
                episode_counts += done_envs

                # Update states
                states = next_states

                # # Collect step from each actor
                # for actor_id in range(self.n_actors):
                #     # Get action from current policy
                #     state = states[actor_id]
                #     action, value, log_prob = self.choose_action(state)

                #     # Track action distribution
                #     action_counts[actor_id][action] += 1

                #     # Execute action in environment
                #     next_state, reward, done, truncated, _ = envs[actor_id].step(action)

                #     # Update metrics
                #     episode_rewards[actor_id] += reward
                #     episode_lengths[actor_id] += 1

                #     # Store transition
                #     self.memory.store(
                #         actor_id=actor_id,
                #         state=state,
                #         action=action,
                #         reward=reward,
                #         value=value,
                #         logprob=log_prob,
                #     )

                #     # Handle episode completion
                #     if done or truncated:
                #         episode_counts[actor_id] += 1
                #         states[actor_id] = envs[actor_id].reset()[0]
                #     else:
                #         states[actor_id] = next_state

            # Step 3: Update policy for K epochs
            # Update policy
            update_metrics = self.update()

            # Calculate metrics
            iteration_metrics = {
                "iteration": iteration,
                "time": time.time() - iteration_start_time,
                "mean_reward": np.mean(episode_rewards),
                "max_reward": np.max(episode_rewards),
                "mean_length": np.mean(episode_lengths),
                "total_episodes": np.sum(episode_counts),
                **update_metrics,
            }

            for action in range(4):
                freqs = action_counts[:, action] / np.maximum(
                    np.sum(action_counts, axis=1), 1
                )
                iteration_metrics[f"action_{action}_freq"] = np.mean(freqs)

            all_metrics.append(iteration_metrics)

            # Print progress
            if iteration % 1 == 0:  # Print every iteration
                print(f"\nIteration {iteration} ({iteration_metrics['time']:.2f}s)")
                print(
                    f"Mean/Max Reward: {iteration_metrics['mean_reward']:.2f}/{iteration_metrics['max_reward']:.2f}"
                )
                print(f"Mean Episode Length: {iteration_metrics['mean_length']:.2f}")
                print(
                    f"Episodes Completed (across all actors): {iteration_metrics['total_episodes']}"
                )
                print(f"Action Frequencies: ", end="")
                for action in range(4):
                    print(
                        f"{action}:{iteration_metrics[f'action_{action}_freq']:.2f} ",
                        end="",
                    )
                print(
                    f"\nLosses - Policy: {iteration_metrics['policy_loss']:.3f}, ",
                    f"Value: {iteration_metrics['value_loss']:.3f}, ",
                    f"Entropy: {iteration_metrics['entropy']:.3f}",
                )
                print("-" * 80)
        return all_metrics

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

    def close(self) -> None:
        """Close the environment and release resources."""
        self.env.close()
