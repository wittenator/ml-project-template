from dataclasses import dataclass

from src.common.logging.wandb import WandBRun
from src.runs.base.job import Job


@dataclass
class Run:
    """Configures a basic run."""

    seed: int
    wandb: WandBRun | None
    job: Job | None
