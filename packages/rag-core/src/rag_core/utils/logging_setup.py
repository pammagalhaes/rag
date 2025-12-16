import logging
import logging.config
import yaml
from pathlib import Path


def init_logging(config_path: str = "configs/logging.yaml"):
    path = Path(config_path)
    if path.exists():
        with open(path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        logging.config.dictConfig(cfg)
    else:
        logging.basicConfig(level=logging.INFO)