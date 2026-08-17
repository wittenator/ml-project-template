import os
import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from conf.cluster_profile import ClusterProfile
from lib.utils.helpers import get_hydra_output_dir
from lib.utils.wandb import WandBConfig
from loguru import logger
from omegaconf import OmegaConf
from submitit import AutoExecutor, LocalExecutor
from submitit.helpers import CommandFunction

MINS_IN_H = 60


def _detect_local_visible_gpus() -> list[int]:
    """Inspect the host's visible GPUs for `cfg/cluster=local` runs.

    Honors `CUDA_VISIBLE_DEVICES` if set (so users can scope a local run to a
    subset), else falls back to `nvidia-smi -L`. Returns `[]` if no GPUs are
    found — callers downgrade to CPU mode.

    Note: submitit's `LocalExecutor` overwrites `CUDA_VISIBLE_DEVICES` in the
    worker env, so to give the worker any GPU we must pass `visible_gpus`
    explicitly to `update_parameters` — inherited env alone is clobbered.
    """
    cvd = os.environ.get("CUDA_VISIBLE_DEVICES")
    if cvd is not None:
        return [int(x) for x in cvd.split(",") if x.strip().isdigit()] if cvd.strip() else []
    try:
        out = subprocess.run(["nvidia-smi", "-L"], capture_output=True, text=True, check=True).stdout
    except (FileNotFoundError, subprocess.CalledProcessError):
        return []
    return list(range(sum(1 for line in out.splitlines() if line.startswith("GPU "))))


@dataclass
class SlurmConfig:
    """Per-job SLURM overrides.

    Every resource field defaults to `None`, meaning "use the cluster
    profile's value". Set one to override it for a single job/sweep, e.g.
    `cfg.job.slurm_config.gpus_per_task=4`.
    """

    partition: str | None = None
    cpus_per_task: int | None = None
    gpus_per_task: int | None = None
    memory_gb: int | None = None
    constraint: str | None = None
    exclude: str | None = None
    time_hours: int | None = None
    nodes: int | None = None
    tasks_per_node: int | None = None

    def _timeout_min(self, partition: str, cluster_profile: ClusterProfile) -> int:
        """Wall-clock `timeout_min`. An explicit `time_hours` wins; otherwise
        the partition's entry in `partition_time_limits`, else
        `default_time_hours`. Submitit defaults to 5 min if unset, which kills
        any real job — so we always set it.
        """
        hours = self.time_hours
        if hours is None:
            hours = cluster_profile.partition_time_limits.get(partition, cluster_profile.default_time_hours)
        return int(hours * MINS_IN_H)

    def to_submitit_params(self, cluster_profile: ClusterProfile) -> dict:
        """Resolve overrides against the cluster profile into submitit params."""
        partition = self.partition or cluster_profile.partition
        cpus_per_task = self.cpus_per_task or cluster_profile.cpus_per_task
        gpus_per_task = self.gpus_per_task if self.gpus_per_task is not None else cluster_profile.gpus_per_task
        memory_gb = self.memory_gb if self.memory_gb is not None else cluster_profile.memory_gb
        constraint = self.constraint or cluster_profile.constraint

        params: dict[str, Any] = {
            "slurm_partition": partition,
            "timeout_min": self._timeout_min(partition, cluster_profile),
        }
        if cpus_per_task:
            params["cpus_per_task"] = cpus_per_task
        if gpus_per_task:
            params["slurm_gpus_per_node"] = gpus_per_task
        if memory_gb:
            params["mem_gb"] = memory_gb
        if constraint:
            params["slurm_constraint"] = constraint
        if self.exclude:
            params["slurm_exclude"] = self.exclude
        if self.nodes:
            params["nodes"] = self.nodes
        if self.tasks_per_node:
            params["tasks_per_node"] = self.tasks_per_node
        return params


@dataclass
class Job:
    """Job to run code on a cluster using apptainer (or locally via submitit)."""

    image: str
    cluster_profile: ClusterProfile
    cluster: str = "slurm"
    slurm_config: SlurmConfig = field(default_factory=SlurmConfig)
    kwargs: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.run()
        sys.exit(0)

    def get_absolute_program_path(self, prog_arg: str | Path) -> str:
        """Get the absolute path of the program."""
        program_path = Path(prog_arg)
        if not program_path.is_absolute():
            program_path = program_path.resolve()
        return str(program_path)

    def filter_args(self, args: list[str]) -> list[str]:
        """Filter args to prevent recursive jobs on the cluster.

        Drops both `cfg/job=…` (defaults-list selection — would make the worker
        re-dispatch the job) and `cfg.job.<field>=…` overrides (dispatcher-only
        knobs; the worker's `cfg.job` is the bare `run` config and rejects them
        as unknown keys).
        """
        return [
            arg
            for arg in args
            if "cfg/job" not in arg and not arg.lstrip("+~").split("=", 1)[0].startswith("cfg.job.")
        ]

    @property
    def python_command(self) -> str:
        """Python command submitit's slurm worker uses to enter the runtime:
        `apptainer exec --nv ${binds} ${image} uv run --no-dev python`.
        (Unused on the local path, which spawns its worker via `sys.executable`.)
        """
        binds = " ".join(f"--bind {b}" for b in self.cluster_profile.binds)
        return f"apptainer exec --nv {binds} {self.image} uv run --no-dev python"

    @property
    def _slurm_setup(self) -> list[str]:
        # Worker shell preamble: create the node-local scratch dir the binds
        # reference. Note: shell-side `export FOO=…` here does NOT reach the
        # python script — submitit's CommandFunction runs Popen(env=…,
        # shell=False), which replaces env. Anything the script reads goes into
        # `exec_env` (built in `copy_project_files`).
        return [
            *self.cluster_profile.local_job_dir_setup,
            f'mkdir -p "{self.cluster_profile.local_job_dir}/run_output"',
        ]

    def _local_timeout_min(self) -> int:
        return self.slurm_config._timeout_min(
            self.slurm_config.partition or self.cluster_profile.partition, self.cluster_profile
        )

    def _submit_local(self, function: CommandFunction, snapshot_dir: Path, *args: Any) -> Any:
        """Submit one task to a fresh `LocalExecutor` with an explicit GPU pin.

        Goes direct rather than through `AutoExecutor` because the latter drops
        the slurm-prefixed GPU params under `cluster="local"`, leaving the
        worker with no visible device. `args` are forwarded to `submit` (used
        by sweep dispatch to pass the wandb sweep id to `wandb agent`).
        """
        visible = _detect_local_visible_gpus()
        n_gpus = self.slurm_config.gpus_per_task
        if n_gpus is None:
            n_gpus = self.cluster_profile.gpus_per_task or 0
        if n_gpus and not visible:
            logger.warning(f"gpus_per_task={n_gpus} but no GPUs detected on host; running CPU-only.")
            n_gpus = 0
        executor = LocalExecutor(snapshot_dir)
        executor.update_parameters(
            timeout_min=self._local_timeout_min(),
            gpus_per_node=n_gpus,
            visible_gpus=visible[:n_gpus] if n_gpus else [],
        )
        return executor.submit(function, *args)

    def run(self) -> None:
        """Run the job. The worker's `hydra.run.dir` is pinned to the dispatcher
        snapshot dir so its outputs land alongside the code copy. After submit,
        a sibling symlink `<jobid> → <run_id>` lets the run be found by the id
        wandb reports.
        """
        exec_env = self.copy_project_files()
        snapshot_dir = get_hydra_output_dir().resolve()

        # Local: invoke the host venv's python by absolute path. Slurm: literal
        # "python", resolved inside the apptainer container the worker enters
        # via `slurm_python`.
        interpreter = sys.executable if self.cluster == "local" else "python"
        command = [
            interpreter,
            self.get_absolute_program_path(snapshot_dir / sys.argv[0]),
            *self.filter_args(sys.argv[1:]),
            "cfg/wandb=log",
            f"hydra.run.dir={snapshot_dir}",
        ]
        function = CommandFunction(command, env=exec_env)

        if self.cluster == "local":
            job = self._submit_local(function, snapshot_dir)
        else:
            executor = AutoExecutor(
                folder=snapshot_dir,
                cluster=self.cluster,
                slurm_python=self.python_command,
            )
            executor.update_parameters(
                **self.slurm_config.to_submitit_params(self.cluster_profile),
                **self.kwargs,
                slurm_setup=self._slurm_setup,
            )
            job = executor.submit(function)
        logger.info(f"Submitted job {job.job_id}")
        self._symlink_jobid(snapshot_dir.parent / str(job.job_id), snapshot_dir)

    @staticmethod
    def _symlink_jobid(symlink_path: Path, snapshot_dir: Path) -> None:
        if symlink_path.exists() or symlink_path.is_symlink():
            return
        try:
            symlink_path.symlink_to(snapshot_dir.name, target_is_directory=True)
            logger.info(f"Symlinked {symlink_path.name} -> {snapshot_dir.name}")
        except OSError as e:
            logger.warning(f"Could not create symlink {symlink_path}: {e}")

    def copy_project_files(self) -> dict[str, str]:
        shutil.copytree(
            Path.cwd(),
            get_hydra_output_dir(),
            ignore=shutil.ignore_patterns(
                "__pycache__", "*.pyc", ".git", ".venv", "*_cache", "*.sif", "*.def", "outputs", "wandb"
            ),
            dirs_exist_ok=True,
        )
        # symlink the live outputs dir into the snapshot so the worker writes
        # back to the shared location instead of a buried copy.
        if not (get_hydra_output_dir() / "outputs").exists():
            os.symlink(Path.cwd() / "outputs", get_hydra_output_dir() / "outputs", target_is_directory=True)

        if (wandb_config := WandBConfig.from_env()) is None:
            raise RuntimeError("No WandB config found in environment.")
        exec_env = os.environ.copy()
        exec_env["PYTHONPATH"] = f"{get_hydra_output_dir()}:{exec_env.get('PYTHONPATH', '')}"
        exec_env["WANDB_PROJECT"] = wandb_config.WANDB_PROJECT
        exec_env["WANDB_ENTITY"] = wandb_config.WANDB_ENTITY
        # deactivate tqdm on jobs to save on disk bandwidth
        exec_env["TQDM_DISABLE"] = "1"
        # Route caches (wandb, etc.) to node-local scratch instead of $HOME.
        # `/cache` exists thanks to the apptainer `$LOCAL_JOB_DIR:/cache` bind;
        # there's no such mount under cluster=local, so leave CACHE_DIR alone.
        if self.cluster != "local":
            exec_env.setdefault("CACHE_DIR", "/cache")
        return exec_env


@dataclass
class SweepJob(Job):
    """Job to run a wandb sweep on a cluster."""

    sweep_id: str = "no_sweep_id"  # for collection of results
    num_workers: int = 2
    parameters: dict[str, Any] = field(default_factory=dict)
    metric_name: str = "loss"
    metric_goal: Literal["maximize", "minimize"] = "minimize"
    method: Literal["grid", "random", "bayes"] = "grid"

    def register_sweep(self, sweep_config: dict) -> str:
        """Register a wandb sweep via the Python API and return its id."""
        import wandb

        if (wandb_config := WandBConfig.from_env()) is None:
            raise RuntimeError("No WandB config found in environment.")
        sweep_id = wandb.sweep(sweep_config, project=wandb_config.WANDB_PROJECT, entity=wandb_config.WANDB_ENTITY)
        logger.info(f"Created wandb sweep: {sweep_id}")
        return sweep_id

    def _dispatch_sweep_workers(self, function: CommandFunction, sweep_id: str, snapshot_dir: Path) -> list[Any]:
        """Dispatch `num_workers` `wandb agent` workers — slurm array vs local.

        Slurm: one `map_array` call, parallelism throttled by
        `slurm_array_parallelism`. Local: a fresh `LocalExecutor` per worker,
        each pinned to a distinct block of `gpus_per_task` GPUs and capped at
        `len(visible) // gpus_per_task` (the local executor has no array
        throttle, so uncapped workers would clash on the same GPU).
        """
        if self.cluster != "local":
            executor = AutoExecutor(
                folder=snapshot_dir,
                cluster=self.cluster,
                slurm_python=self.python_command,
            )
            executor.update_parameters(
                slurm_array_parallelism=self.num_workers,
                slurm_setup=self._slurm_setup,
                **self.slurm_config.to_submitit_params(self.cluster_profile),
                **self.kwargs,
            )
            return executor.map_array(function, [sweep_id] * self.num_workers)

        visible = _detect_local_visible_gpus()
        n_gpus = self.slurm_config.gpus_per_task
        if n_gpus is None:
            n_gpus = self.cluster_profile.gpus_per_task or 0
        if n_gpus and not visible:
            logger.warning(f"gpus_per_task={n_gpus} but no GPUs detected on host; running CPU-only.")
            n_gpus = 0
        if n_gpus > 0:
            concurrent = max(1, len(visible) // n_gpus)
            if self.num_workers > concurrent:
                logger.warning(
                    f"cluster=local: capping num_workers from {self.num_workers} to {concurrent} "
                    f"({len(visible)} GPU(s) / {n_gpus} per task)."
                )
            actual_workers = min(self.num_workers, concurrent)
        else:
            actual_workers = self.num_workers

        timeout_min = self._local_timeout_min()
        jobs: list[Any] = []
        for i in range(actual_workers):
            gpu_block = visible[i * n_gpus : (i + 1) * n_gpus] if n_gpus else []
            executor = LocalExecutor(snapshot_dir)
            executor.update_parameters(timeout_min=timeout_min, gpus_per_node=n_gpus, visible_gpus=gpu_block)
            jobs.append(executor.submit(function, sweep_id))
        return jobs

    def run(self) -> None:
        """Register the sweep and dispatch its workers."""
        exec_env = self.copy_project_files()
        snapshot_dir = get_hydra_output_dir().resolve()

        parameters = OmegaConf.to_container(self.parameters, resolve=True)
        metric = {"goal": self.metric_goal, "name": self.metric_name}
        program, args = (
            self.get_absolute_program_path(snapshot_dir / sys.argv[0]),
            self.filter_args(sys.argv[1:]),
        )
        # `${now:%H-%M-%S-%f}` fires on the worker at config-composition time,
        # giving each task a unique microsecond-stamped leaf dir.
        hydra_run_dir = f"{snapshot_dir}/${{now:%H-%M-%S-%f}}"
        command = [
            "${env}",
            "${interpreter}",
            "${program}",
            *args,
            "cfg/wandb=log",
            f"hydra.run.dir={hydra_run_dir}",
            "${args_no_hyphens}",
        ]

        sweep_config = {
            "program": program,
            "method": self.method,
            "metric": metric,
            "parameters": parameters,
            "command": command,
        }

        sweep_id = self.register_sweep(sweep_config)
        self._symlink_jobid(snapshot_dir.parent / sweep_id, snapshot_dir)

        function = CommandFunction(["wandb", "agent"], env=exec_env)
        jobs = self._dispatch_sweep_workers(function, sweep_id, snapshot_dir)
        for job in jobs:
            logger.info(f"Submitted job {job.job_id}")
