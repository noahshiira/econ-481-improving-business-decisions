import yaml
from pathlib import Path

def load_config(path: str = "configs/default.yaml") -> dict:
    """Load YAML configuration file."""
    with open(Path(path), "r") as f:
        cfg = yaml.safe_load(f)
    return cfg
