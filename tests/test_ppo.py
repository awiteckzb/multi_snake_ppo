import yaml
from pathlib import Path

from src.common.config import ConfigManager
from src.environments.single_agent import SnakeEnv
from src.algorithms.ppo import PPO


def main():
    # Load config
    config = ConfigManager("configs/default.yaml")

    # Create environment
    env = SnakeEnv(
        grid_size=config.get("environment.grid_size"),
        max_steps=config.get("environment.max_steps"),
    )

    # Initialize PPO agent with minimal number of actors for testing
    ppo = PPO(env=env, config_manager=config, device="cpu")

    # Test parameters (smaller than what we'll use in production)
    config.config["ppo"]["n_actors"] = 2  # Just 2 actors for testing
    config.config["ppo"]["n_steps"] = 128  # Smaller number of steps
    config.config["training"]["total_episodes"] = 5  # Just a few iterations for testing

    try:
        # Run training
        ppo.train()
        print("Training completed successfully!")

        # Test the trained policy
        print("\nTesting trained policy...")
        state, _ = env.reset()
        total_reward = 0
        done = False

        while not done:
            # Use the policy (no need to store values/logprobs during testing)
            action, _, _ = ppo.choose_action(state)
            state, reward, done, truncated, _ = env.step(action)
            done = done or truncated
            total_reward += reward
            env.render()  # Show the game

        print(f"Test episode completed with reward: {total_reward}")

    except Exception as e:
        print(f"Error during training: {str(e)}")
        raise e


if __name__ == "__main__":
    main()
