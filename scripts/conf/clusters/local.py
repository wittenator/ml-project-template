"""Local-host cluster profile for smoke-testing the submitit path.

Selects submitit's `LocalExecutor` (via `executor="local"`), which spawns a
host subprocess per submitted job — no apptainer, no sbatch, no SLURM env vars.
Useful for verifying the training/sweep wiring on a dev machine before going to
a real cluster.

GPU allocation: each submitted task gets a distinct slice of the host's visible
GPUs (see `_detect_local_visible_gpus` and the local branches in
`lib/utils/job.py`). For sweeps the dispatcher caps `num_workers` to
`len(visible_gpus) // gpus_per_task` so concurrent workers don't clash on the
same device — the local executor has no array-parallelism throttle.

Override knobs:
- `cfg.job.slurm_config.gpus_per_task=N` — request N GPUs per task.
- `CUDA_VISIBLE_DEVICES=0,2` (env) — restrict the host's GPU pool.
"""

from conf.cluster_profile import ClusterProfile
from hydra_zen import builds

LocalProfile = builds(
    ClusterProfile,
    name="local",
    executor="local",
    binds=[],
    local_job_dir="${oc.env:TMPDIR,/tmp}/ml-local",
    local_job_dir_setup=[],
    partition="local",
    cpus_per_task=4,
    gpus_per_task=1,
    memory_gb=None,
    constraint=None,
    partition_time_limits={"local": 1},
    default_time_hours=1,
)
