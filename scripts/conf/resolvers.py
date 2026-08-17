"""OmegaConf resolvers used by the run-directory / wandb-name layout.

Registered once via `register_resolvers()` from `configure_main` (rather than at
import time) so module imports stay at the top of `base_conf.py`. `_RUN_UUID` is
computed once per process, so every `${uuid:}` / `${run_id:}` reference in the
same config render returns the same value.
"""

import os
import uuid

from omegaconf import OmegaConf

_RUN_UUID = uuid.uuid4().hex[:8]


def _uuid_resolver() -> str:
    return _RUN_UUID


def run_id() -> str:
    """Resolve the leaf run directory / wandb run name.

    Priority:
      1. `SLURM_ARRAY_JOB_ID`/`SLURM_JOB_ID` — set on a SLURM worker.
      2. `SUBMITIT_LOCAL_JOB_ID` — set by submitit's local executor (used when
         running `cfg/cluster=local`).
      3. `_RUN_UUID` — bare 8-hex uuid for plain off-cluster runs.
    """
    if jobid := os.environ.get("SLURM_JOB_ID"):
        if "SLURM_ARRAY_JOB_ID" in os.environ:
            return f"{os.environ['SLURM_ARRAY_JOB_ID']}_{os.environ.get('SLURM_ARRAY_TASK_ID', '0')}"
        return jobid
    if local_jobid := os.environ.get("SUBMITIT_LOCAL_JOB_ID"):
        return local_jobid
    return _RUN_UUID


def wandb_name() -> str | None:
    """Wandb run name. `None` lets wandb auto-assign its fun two-word name.

    Inside a sweep — detected via `WANDB_SWEEP_ID` set by `wandb agent` — pin
    the run name to the worker's jobid so it matches the on-disk run dir.
    """
    if os.environ.get("WANDB_SWEEP_ID"):
        return run_id()
    return None


def register_resolvers() -> None:
    OmegaConf.register_new_resolver("uuid", _uuid_resolver, replace=True)
    OmegaConf.register_new_resolver("run_id", run_id, replace=True)
    OmegaConf.register_new_resolver("wandb_name", wandb_name, replace=True)
