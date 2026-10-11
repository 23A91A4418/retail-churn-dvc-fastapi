import os
import yaml
from typing import Any, Dict

def load_config(config_path: str = "params.yaml") -> Dict[str, Any]:
    """Loads YAML configuration file."""
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found at: {config_path}")
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    return config
