import os
import random
from collections.abc import Callable
from functools import partial
from pathlib import Path
from typing import Any

from hydra.core.hydra_config import HydraConfig
from hydra_zen import builds

# `builds` shortcuts for partial configs: `pbuilds` leaves the target partially
# applied (zen_partial), `pbuilds_full` also populates the full signature.
pbuilds: Callable[..., Any] = partial(builds, zen_partial=True)
pbuilds_full: Callable[..., Any] = partial(builds, zen_partial=True, populate_full_signature=True)


def get_hydra_output_dir() -> Path:
    return Path(HydraConfig.get().runtime.output_dir)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
