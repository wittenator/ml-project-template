from hydra_zen import builds, store

from lib.utils.wandb import WandBRun
from lib.utils.run_utils import Run, Job, SweepJob

BaseRunConfig = builds(Run, seed=42, wandb=None, job=None)

BaseJobConfig = builds(
    Job,
    partition="cpu-2h",
    image="docker://ghcr.io/marvinsxtr/ml-project-template:main",
    cluster="slurm",
    kwargs={},
)

BaseSweepConfig = builds(
    SweepJob,
    num_workers=2,
    parameters={"cfg.seed": [42, 1337]},
    builds_bases=(BaseJobConfig,),
)

BaseWandBConfig = builds(WandBRun, group=None, mode="online")


run_config_store = store(
    group="cfg", hydra_defaults=["_self_", {"wandb": None}, {"job": None}]
)
run_config_store(BaseRunConfig, name="base")

wandb_config_store = store(group="cfg/wandb")
wandb_config_store(BaseWandBConfig, name="base")

job_config_store = store(group="cfg/job")
job_config_store(BaseJobConfig, name="base")
job_config_store(BaseSweepConfig, name="sweep")
