from pathlib import Path

import yaml


def load_config(path):
    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)
