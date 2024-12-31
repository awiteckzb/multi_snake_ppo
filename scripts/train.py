import argparse
from pathlib import Path
import sys

# Add src to python path
sys.path.append(str(Path(__file__).parent.parent))

from src.common.config import ConfigManager
from src.environments.single_agent import SnakeEnv
from src.algorithms.ppo import PPO
from src.common.logging import Logger


def parse_args():
    parser = argparse.ArgumentParser(description="Train PPO agent on Snake environment")
    parser.add_argument(
        "--config", type=str, default="configs/default.yaml", help="Path to config file"
    )
    parser.add_argument(
        "--checkpoint", type=str, default=None, help="Path to checkpoint to resume from"
    )
    parser.add_argument(
        "--overrides", type=str, default=None, help="JSON string of config overrides"
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # Load config
    config = ConfigManager(args.config)

    # Initialize environment
    env = SnakeEnv(
        grid_size=config.get("environment.grid_size"),
        max_steps=config.get("environment.max_steps"),
    )

    # Initialize agent
    agent = PPO(
        env=env,
        learning_rate=config.get("ppo.learning_rate"),
        gamma=config.get("ppo.gamma"),
        gae_lambda=config.get("ppo.gae_lambda"),
        clip_epsilon=config.get("ppo.clip_epsilon"),
        c1=config.get("ppo.c1"),
        c2=config.get("ppo.c2"),
        batch_size=config.get("ppo.batch_size"),
        n_epochs=config.get("ppo.n_epochs"),
        max_grad_norm=config.get("ppo.max_grad_norm"),
    )

    # Load checkpoint if provided
    if args.checkpoint:
        agent.load_checkpoint(args.checkpoint)

    # Initialize logger
    logger = Logger(config.get("training.log_dir"))

    # Training loop
    total_episodes = config.get("training.total_episodes")
    save_freq = config.get("training.save_frequency")
    checkpoint_dir = Path(config.get("training.checkpoint_dir"))
    checkpoint_dir.mkdir(exist_ok=True)

    for episode in range(total_episodes):
        # Train one episode
        episode_info = agent.train_episode()

        # Log metrics
        logger.log_metrics(episode_info, episode)

        # Save checkpoint
        if episode % save_freq == 0:
            checkpoint_path = checkpoint_dir / f"checkpoint_{episode}.pt"
            agent.save_checkpoint(checkpoint_path)

            # Also save config
            config.save(checkpoint_dir / f"config_{episode}.yaml")


if __name__ == "__main__":
    main()
