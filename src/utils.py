from typing import Dict

import yaml
from yaml import Loader


def load_config(cfg_path: str) -> Dict:
    """Load YAML config file."""

    with open(cfg_path, "r") as ymlfile:
        cfg = yaml.load(ymlfile, Loader)

    return cfg

def check_input_dim(cfg: Dict, input_dim: int) -> bool:
    """Check input dimension entered in config."""

    if cfg["input_layer"][0][1] == cfg["output_layer"][0][-1] == input_dim:
        return True
    return False