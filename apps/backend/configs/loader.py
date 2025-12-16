import os
import yaml
from typing import Any, Dict


def load_yaml(path: str) -> Dict[str, Any]:
    """Load a YAML file relative to the configs directory."""
    base_dir = os.path.dirname(__file__)  # path to src/configs
    full_path = os.path.join(base_dir, path)

    if not os.path.exists(full_path):
        raise FileNotFoundError(f"Config file not found: {full_path}")

    with open(full_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)



def merge_dicts(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """
    Recursively merge two dictionaries.
    Values in 'override' take precedence.
    """
    result = base.copy()

    for key, value in override.items():
        if (
            key in result
            and isinstance(result[key], dict)
            and isinstance(value, dict)
        ):
            result[key] = merge_dicts(result[key], value)
        else:
            result[key] = value

    return result


def load_config(path: str, extra: Dict[str, Any] = None) -> Dict[str, Any]:
    """
    Load configuration from YAML and optionally merge with extra overrides.

    Args:
        path: Path to YAML file
        extra: Optional dict to override values

    Returns:
        Final configuration dictionary
    """
    cfg = load_yaml(path)

    if extra:
        cfg = merge_dicts(cfg, extra)

    return cfg
