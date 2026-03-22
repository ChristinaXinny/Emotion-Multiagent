"""Configuration management module."""

import yaml
from pathlib import Path
from typing import Dict, Any


def load_config(config_path: str = None) -> Dict[str, Any]:
    """
    Load configuration from YAML file.

    Args:
        config_path: Path to config file. Defaults to config/config.yaml

    Returns:
        Configuration dictionary
    """
    if config_path is None:
        # Default to config/config.yaml in project root
        project_root = Path(__file__).parent.parent.parent
        config_path = project_root / "config" / "config.yaml"

    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    return config
