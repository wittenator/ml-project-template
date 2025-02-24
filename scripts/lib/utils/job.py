import os
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import yaml
from lib.utils.helpers import get_hydra_output_dir
from lib.utils.wandb import WandBConfig
from loguru import logger
from omegaconf import OmegaConf
from submitit import AutoExecutor
from submitit.helpers import CommandFunction

partition_name_to_time_limit_hrs = {
    "gpu": 0.5,
    "gpu-test": 0.25,
    "cpu-2h": 2,
    "cpu-5h": 5,
    "cpu-2d": 48,
    "cpu-7d": 168,
    "gpu-2h": 2,
    "gpu-5h": 5,
    "gpu-2d": 48,
    "gpu-7d": 168,
}

MINS_IN_H = 60


@dataclass
class SlurmConfig:
    """SLURM resource configuration."""

    partition: str = "gpu-2h"
    cpus_per_task: int | None = 3
    gpus_per_task: int | None = None
    memory_gb: int | None = None
    exclude: str | None = None
    constraint: str | None = None
    time_hours: int | None = None
    nodes: int | None = None
    tasks_per_node: int | None = None

    def to_submitit_params(self) -> dict:
        """Convert to submitit parameters."""
        params = {}
        if self.partition:
            params["slurm_partition"] = self.partition
            params["timeout_min"] = int(partition_name_to_time_limit_hrs[self.partition] * MINS_IN_H)

        if self.cpus_per_task:
            params["cpus_per_task"] = self.cpus_per_task

        if self.gpus_per_task:
            params["slurm_gpus_per_node"] = self.gpus_per_task

        if self.memory_gb:
            params["mem_gb"] = self.memory_gb

        if self.exclude:
            params["slurm_exclude"] = self.exclude

        if self.time_hours:
            params["time"] = f"{self.time_hours}:00:00"

        if self.nodes:
            params["nodes"] = self.nodes

        if self.tasks_per_node:
            params["tasks_per_node"] = self.tasks_per_node

        if self.constraint:
            params["slurm_constraint"] = self.constraint

        return params


@dataclass
class Job:
    """Job to run code on a cluster using apptainer."""

    image: str
    cluster: str = "slurm"
    slurm_config: SlurmConfig = field(default_factory=SlurmConfig)
    kwargs: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.run()
        sys.exit(0)

    def get_absolute_program_path(self, prog_arg: str) -> str:
        """Get the absolute path of the program."""
        program_path = Path(prog_arg)
        if not program_path.is_absolute():
            program_path = Path(prog_arg).resolve()
        return str(program_path)

    def filter_args(self, args: list[str]) -> list[str]:
        """Filter args to prevent recursive jobs on the cluster."""
        return [arg for arg in args if "cfg/job" not in arg]

    @property
    def python_command(self) -> str:
        """Python command used by the job."""
        return f"apptainer exec --nv ${{GIT_DIR}}/{self.image} /uv/uv run python"

    def run(self) -> None:
        """Run the job on the cluster."""
        # copy source code to output directory
        exec_env = self.copy_project_files()
        hydra_run_dir = "./outputs/runs/${now:%Y-%m-%d}/${now:%H-%M-%S-%f}"

        command = [
            "python",
            self.get_absolute_program_path(get_hydra_output_dir() / sys.argv[0]),
            *self.filter_args(sys.argv[1:]),
            "cfg/wandb=log",
            f"hydra.run.dir={hydra_run_dir}",
        ]
        function = CommandFunction(command, env=exec_env)

        executor = AutoExecutor(
            folder=get_hydra_output_dir(),
            cluster=self.cluster,
            slurm_python=self.python_command,
        )
        # Combine base kwargs with SLURM config
        all_params = {
            **self.slurm_config.to_submitit_params(),
            **self.kwargs,
        }

        executor.update_parameters(**all_params)
        job = executor.submit(function)
        logger.info(f"Submitted job {job.job_id}")

    def copy_project_files(self) -> dict[str, str]:
        shutil.copytree(
            Path.cwd(),
            get_hydra_output_dir(),
            ignore=shutil.ignore_patterns(
                "__pycache__", "*.pyc", ".git", ".venv", "*_cache", "*.sif", "*.def", "outputs", "wandb"
            ),
            dirs_exist_ok=True,
        )

        if (wandb_config := WandBConfig.from_env()) is None:
            raise RuntimeError("No WandB config found in environment.")
        exec_env = os.environ.copy()
        exec_env["PYTHONPATH"] = f"{get_hydra_output_dir()}:{exec_env.get('PYTHONPATH', '')}"
        exec_env["WANDB_PROJECT"] = wandb_config.WANDB_PROJECT
        exec_env["WANDB_ENTITY"] = wandb_config.WANDB_ENTITY
        return exec_env


@dataclass
class SweepJob(Job):
    """Job to run a sweep on a cluster."""

    sweep_id: str = "no_sweep_id"  # for collection of results
    num_workers: int = 2
    parameters: dict[str, list[Any] | dict[Any]] = field(default_factory=dict)
    metric_name: str = "loss"
    metric_goal: Literal["maximize", "minimize"] = "minimize"
    method: Literal["grid", "random", "bayes"] = "grid"

    def register_sweep(self, sweep_config: dict) -> str:
        """Register a wandb sweep from a config."""
        if (wandb_config := WandBConfig.from_env()) is None:
            raise RuntimeError("No WandB config found in environment.")

        with tempfile.TemporaryDirectory() as tmp_dir:
            config_path = Path(tmp_dir) / "sweep_config.yaml"

            with Path.open(config_path, "w") as config_file:
                yaml.dump(sweep_config, config_file)

            try:
                output = subprocess.run(
                    [
                        "wandb",
                        "sweep",
                        "--project",
                        wandb_config.WANDB_PROJECT,
                        str(config_path),
                    ],
                    check=True,
                    text=True,
                    capture_output=True,
                ).stderr
            except subprocess.CalledProcessError as e:
                logger.error(e.stderr)
                raise

            sweep_id = output.split(" ")[-1].strip()

            for line in output.splitlines():
                logger.info(line)

        return sweep_id

    def run(self) -> None:
        """Run the sweep on the cluster."""
        exec_env = self.copy_project_files()
        parameters = OmegaConf.to_container(self.parameters, resolve=True)
        metric = {"goal": self.metric_goal, "name": self.metric_name}
        program, args = (
            self.get_absolute_program_path(get_hydra_output_dir() / sys.argv[0]),
            self.filter_args(sys.argv[1:]),
        )
        hydra_run_dir = "./outputs/sweeps/" + self.sweep_id + "/${now:%H-%M-%S-%f}"
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
        function = CommandFunction(["wandb", "agent"], env=exec_env)
        executor = AutoExecutor(
            folder=get_hydra_output_dir(),
            cluster=self.cluster,
            slurm_python=self.python_command,
        )
        executor.update_parameters(
            slurm_array_parallelism=self.num_workers,
            **self.slurm_config.to_submitit_params(),
            **self.kwargs,
        )
        jobs = executor.map_array(function, [sweep_id] * self.num_workers)

        for job in jobs:
            logger.info(f"Submitted job {job.job_id}")
