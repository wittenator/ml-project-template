from hydra_zen import builds

from src.runs.base.job import Job, SweepJob
from src.runs.base.run import Run

BaseRunConfig = builds(Run, seed=42, wandb=None, job=None)

BaseJobConfig = builds(
    Job, partition="cpu-2h", image="docker://ghcr.io/marvinsxtr/ml-project-template:main", cluster="slurm", kwargs={}
)

BaseSweepConfig = builds(SweepJob, num_workers=2, parameters={"cfg.seed": [42, 1337]}, builds_bases=(BaseJobConfig,))
