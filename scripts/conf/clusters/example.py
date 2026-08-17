"""Example Slurm cluster profile — copy this to add your own cluster.

Replace the binds, partitions, and time limits with the ones your cluster
actually uses, then register the new profile in `clusters/__init__.py`. Submit
to it with `cfg/cluster=example cfg/job=run`.

Use bash's no-brace `$VAR` syntax for shell variables in `binds` /
`local_job_dir` — the brace form `${VAR}` collides with OmegaConf's
interpolation grammar inside `builds(...)`.
"""

from conf.cluster_profile import ClusterProfile
from hydra_zen import builds

ExampleProfile = builds(
    ClusterProfile,
    name="example",
    executor="slurm",
    # Bind your project/data volumes 1:1 so absolute paths match between the
    # login node and compute nodes. `$LOCAL_JOB_DIR:/cache` gives the worker a
    # writable scratch mount on the compute node.
    binds=[
        "$LOCAL_JOB_DIR:/cache",
        "/data/$USER:/data/$USER",
    ],
    local_job_dir="/tmp/$SLURM_JOB_ID",
    local_job_dir_setup=[
        'export LOCAL_JOB_DIR="/tmp/$SLURM_JOB_ID"',
        'mkdir -p "$LOCAL_JOB_DIR/run_output"',
    ],
    partition="gpu",
    cpus_per_task=12,
    gpus_per_task=1,
    memory_gb=32,
    constraint=None,
    partition_time_limits={
        "gpu": 24,
        "gpu-test": 1,
    },
    default_time_hours=24,
)
