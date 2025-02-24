from pathlib import Path
from dataclasses import dataclass, field
from typing import Literal
import socket
from hydra_zen import builds, store

from lib.utils.wandb import WandBRun
from lib.utils.job import Job, SweepJob, SlurmConfig


@dataclass
class RuntimeInfo:
    # device: str = field(default_factory=lambda: "cuda" if th.cuda.is_available() else "cpu")
    out_dir: Path | None = None
    # n_gpu: int = field(default_factory=th.cuda.device_count)
    node_hostname: str = field(default_factory=socket.gethostname)


@dataclass
class BaseConfig:
    seed: int = 42
    runtime: RuntimeInfo = field(default_factory=RuntimeInfo)
    loglevel: Literal["debug", "info"] = "debug"
    wandb: WandBRun | None = None
    job: Job | None = None


BaseSlurmConfig = builds(
    SlurmConfig,
    partition="gpu",
    cpus_per_task=2,
    gpus_per_task=1,
    memory_gb=8,
    nodes=1,
    constraint="",
)

# get main script path
sif_path = Path(__file__).resolve().parent.parent.parent / "container.sif"

BaseJobConfig = builds(
    Job,
    image=sif_path,
    kwargs={},
    slurm_config=BaseSlurmConfig,
)

BaseSweepConfig = builds(
    SweepJob,
    num_workers=2,
    sweep_id="new_md17_exp",
    metric_name="loss",
    parameters={"bar": {"values": [42, 43, 44]}},
    method="grid",
    builds_bases=(BaseJobConfig,),
)

BaseWandBConfig = builds(WandBRun, group=None, mode="online")


def configure_main(extra_defaults: list[dict] | None = None):
    wandb_config_store = store(group="cfg/wandb")
    wandb_config_store(BaseWandBConfig, name="log")

    job_config_store = store(group="cfg/job")
    job_config_store(BaseJobConfig, name="run")
    job_config_store(BaseSweepConfig, name="sweep")

    run_config = builds(BaseConfig, populate_full_signature=True)
    base_defaults = ["_self_", {"cfg/wandb": None}, {"cfg/job": None}]
    if extra_defaults is not None:
        base_defaults.extend(extra_defaults)

    def decorator(main_func):
        main_func_store = store(
            main_func,
            name="root",
            cfg=run_config,
            hydra_defaults=base_defaults,
        )

        return main_func_store

    return decorator
