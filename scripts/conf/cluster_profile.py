"""Per-cluster knobs that the rest of the code reads at submission time.

A `ClusterProfile` bundles everything that differs between the machines you
submit to: the submitit executor backend, apptainer binds, node-local scratch,
the resource defaults, and the per-partition wall-clock limits.

Adding a new cluster = one new file under `scripts/conf/clusters/` that builds a
`ClusterProfile` and registers it in `clusters/__init__.py`. No `Job` or shebang
edits, and no second copy of the resource fields — `SlurmConfig` only carries
the per-job *overrides* and falls back to the values here (see
`SlurmConfig.to_submitit_params` in `lib/utils/job.py`).
"""

from dataclasses import dataclass, field
from typing import Literal


@dataclass
class ClusterProfile:
    name: str

    # Submitit executor backend. "slurm" = a real cluster (apptainer + sbatch);
    # "local" = host subprocesses, used to smoke-test the submission path on a
    # dev machine. `BaseJobConfig.cluster` interpolates from this so
    # `cfg/cluster=local` flips the executor automatically.
    executor: Literal["slurm", "local"] = "slurm"

    # Apptainer binds in "src:dst" form. Use bash's no-brace `$VAR` syntax for
    # shell variables (e.g. `$LOCAL_JOB_DIR:/cache`) so OmegaConf leaves them
    # alone and bash expands them on the compute node — the brace form
    # `${VAR}` collides with OmegaConf's interpolation grammar.
    binds: list[str] = field(default_factory=list)

    # Node-local scratch directory (a shell expression evaluated on the compute
    # node) and the commands that create it. Emitted as the slurm `setup`
    # preamble before the worker starts.
    local_job_dir: str = "${TMPDIR:-/tmp}/${SLURM_JOB_ID}"
    local_job_dir_setup: list[str] = field(default_factory=list)

    # Resource defaults — used whenever the per-job `SlurmConfig` leaves the
    # matching field unset (None).
    partition: str = "gpu"
    cpus_per_task: int = 12
    gpus_per_task: int | None = 1
    memory_gb: int | None = 32
    constraint: str | None = None

    # Per-partition wall-clock limit, in hours. Submitit defaults `timeout_min`
    # to 5 minutes if unset, which kills any real job — so each partition you
    # actually submit to needs an entry here. Partitions not listed fall back
    # to `default_time_hours`.
    partition_time_limits: dict[str, int] = field(default_factory=dict)
    default_time_hours: int = 24
