import yaml
from pathlib import Path


TEMPLATES_PATH = Path("configs/prompt_templates.yaml")


def load_templates():
    with open(TEMPLATES_PATH, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)