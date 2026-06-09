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


def get_cache_dir() -> Path:
    """Shared cache root, kept off `$HOME`. Override with the `CACHE_DIR` env var
    (e.g. node-local scratch on a cluster). Defaults to `<cwd>/.cache`.
    """
    return Path(os.environ.get("CACHE_DIR", Path.cwd() / ".cache"))


def redirect_caches() -> None:
    """Point cache/config dirs of tools that otherwise pollute `$HOME` at the
    shared `CACHE_DIR`. Uses `setdefault` so an explicit env override always
    wins. Extend this for any other tool you add (e.g. `HF_HOME`,
    `MPLCONFIGDIR`, `XDG_CACHE_HOME`).
    """
    cache_dir = get_cache_dir()
    os.environ.setdefault("WANDB_CACHE_DIR", str(cache_dir / "wandb"))
    os.environ.setdefault("WANDB_CONFIG_DIR", str(cache_dir / "wandb"))


def seed_everything(seed: int) -> None:
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
