import yaml
from pathlib import Path
from typing import Dict, Any, Optional


class ConfigManager:
    def __init__(self, config_path: str, overrides: Optional[Dict[str, Any]] = None):
        self.config_path = Path(config_path)
        self.config = self._load_config()

        if overrides:
            self._update_recursive(self.config, overrides)

    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from YAML file."""
        with open(self.config_path, "r") as f:
            return yaml.safe_load(f)

    def _update_recursive(self, d: Dict[str, Any], u: Dict[str, Any]) -> None:
        """Recursively update nested dictionary."""
        for k, v in u.items():
            if isinstance(v, dict) and k in d:
                self._update_recursive(d[k], v)
            else:
                d[k] = v

    def get(self, key: str, default: Any = None) -> Any:
        """Get value from nested config using dot notation."""
        keys = key.split(".")
        value = self.config

        for k in keys:
            if isinstance(value, dict):
                value = value.get(k)
            else:
                return default

            if value is None:
                return default

        return value

    def save(self, path: str) -> None:
        """Save current configuration to file."""
        with open(path, "w") as f:
            yaml.dump(self.config, f, default_flow_style=False)


# Usage example:
if __name__ == "__main__":
    config = ConfigManager("configs/default.yaml")

    # Access nested config values
    lr = config.get("ppo.learning_rate")
    grid_size = config.get("environment.grid_size")

    # Override values
    overrides = {"ppo": {"learning_rate": 1e-4}, "environment": {"grid_size": 15}}

    config = ConfigManager("configs/default.yaml", overrides)
