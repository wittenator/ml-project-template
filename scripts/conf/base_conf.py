import socket
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from conf.cluster_profile import ClusterProfile
from conf.clusters import register_clusters
from conf.resolvers import register_resolvers
from hydra.conf import HydraConf, RunDir
from hydra_zen import builds, store
from lib.utils.job import Job, SlurmConfig, SweepJob
from lib.utils.wandb import WandBRun


@dataclass
class RuntimeInfo:
    # device: str = field(default_factory=lambda: "cuda" if th.cuda.is_available() else "cpu")
    out_dir: Path | None = None
    # n_gpu: int = field(default_factory=th.cuda.device_count)
    node_hostname: str = field(default_factory=socket.gethostname)


@dataclass
class BaseConfig:
    seed: int = 42
    debug: bool = False
    runtime: RuntimeInfo = field(default_factory=RuntimeInfo)
    loglevel: Literal["warning", "info", "debug"] = "info"
    wandb: WandBRun | None = None
    job: Job | None = None
    cluster: ClusterProfile | None = None


# SlurmConfig only carries per-job *overrides* — every field left None falls
# back to the selected cluster profile (see SlurmConfig.to_submitit_params).
BaseSlurmConfig = builds(
    SlurmConfig,
    nodes=1,
)

# get main script path
sif_path = Path(__file__).resolve().parent.parent.parent / "container.sif"

BaseJobConfig = builds(
    Job,
    image=sif_path,
    kwargs={},
    slurm_config=BaseSlurmConfig,
    cluster_profile="${cfg.cluster}",
    # `oc.select` falls back to "slurm" for profiles that don't set `executor`;
    # LocalProfile sets it to "local" so `cfg/cluster=local` flips automatically.
    cluster="${oc.select:cfg.cluster.executor,slurm}",
)

BaseSweepConfig = builds(
    SweepJob,
    num_workers=2,
    sweep_id="example_sweep",
    metric_name="loss",
    parameters={"bar": {"values": [42, 43, 44]}},
    method="grid",
    builds_bases=(BaseJobConfig,),
)

BaseWandBConfig = builds(WandBRun, group=None, mode="online", name="${wandb_name:}")

# Sentinel guarding the one-time store registration (see clusters/__init__.py).
_registered: list[bool] = []


def configure_main(extra_defaults: list[dict] | None = None):
    register_resolvers()
    register_clusters()

    if not _registered:
        # Override Hydra's default `outputs/YYYY-MM-DD/HH-MM-SS/` run dir with a
        # `${run_id:}`-keyed layout (stable across a job's SLURM/local id and
        # nicer for sweeps). `OUTPUT_DIR` lets the job dispatcher redirect
        # worker output to fast local scratch; off-cluster it falls back to
        # `./outputs`.
        store(
            HydraConf(run=RunDir(dir="${oc.env:OUTPUT_DIR,./outputs}/${now:%Y-%m-%d}/${run_id:}")),
            name="config",
            group="hydra",
            provider="ml-project-template",
        )

        wandb_config_store = store(group="cfg/wandb")
        wandb_config_store(BaseWandBConfig, name="log")

        job_config_store = store(group="cfg/job")
        job_config_store(BaseJobConfig, name="run")
        job_config_store(BaseSweepConfig, name="sweep")
        _registered.append(True)

    run_config = builds(BaseConfig, populate_full_signature=True)
    base_defaults = [
        "_self_",
        {"cfg/wandb": None},
        {"cfg/job": None},
        {"cfg/cluster": "local"},
    ]
    if extra_defaults is not None:
        base_defaults.extend(extra_defaults)

    def decorator(main_func):
        # Derive a unique store name from the module so several entry points
        # (train.py, sample.py, ...) can register in the same process.
        module_name = main_func.__module__.split(".")[-1]
        store_name = "root" if module_name == "__main__" else f"{module_name}_main"

        main_func_store = store(
            main_func,
            name=store_name,
            cfg=run_config,
            hydra_defaults=base_defaults,
        )

        return main_func_store

    return decorator
