import os
import random
from pathlib import Path

from hydra.core.hydra_config import HydraConfig


def get_hydra_output_dir() -> Path:
    return Path(HydraConfig.get().runtime.output_dir)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
