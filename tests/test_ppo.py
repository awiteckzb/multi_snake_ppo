import yaml
from pathlib import Path

from src.common.config import ConfigManager
from src.environments.single_agent import SnakeEnv
from src.algorithms.ppo import PPO


def make_env(config):
    """Function that creates a single environment instance."""
    return lambda: SnakeEnv(
        grid_size=config.get("environment.grid_size"),
        max_steps=config.get("environment.max_steps"),
    )


def main():
    # Load config
    config = ConfigManager("configs/default.yaml")

    # Test parameters (smaller than what we'll use in production)
    config.config["ppo"]["n_actors"] = 10  # Just 2 actors for testing
    config.config["ppo"]["n_steps"] = 256  # Smaller number of steps
    config.config["training"][
        "total_iterations"
    ] = 100  # Just a few iterations for testing

    try:
        # Initialize PPO with environment creation function
        ppo = PPO(
            env_fn=make_env(config),  # Pass function instead of environment
            config_manager=config,
            device="cpu",
        )

        # Run training
        ppo.train()
        print("Training completed successfully!")

        # Create a single environment for testing
        test_env = SnakeEnv(
            grid_size=config.get("environment.grid_size"),
            max_steps=config.get("environment.max_steps"),
        )

        # Test the trained policy
        print("\nTesting trained policy...")
        state, _ = test_env.reset()
        total_reward = 0
        done = False

        while not done:
            action, _, _ = ppo.choose_action(state)
            state, reward, done, truncated, _ = test_env.step(action)
            done = done or truncated
            total_reward += reward
            test_env.render()  # Show the game

        print(f"Test episode completed with reward: {total_reward}")

        # Clean up
        ppo.close()
        test_env.close()

    except Exception as e:
        print(f"Error during training: {str(e)}")
        raise e


if __name__ == "__main__":
    main()
