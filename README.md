
### This is a fork of [this](https://github.com/marvinsxtr/ml-project-template) repo. Functionality is similar - a few weird behaviours in edge cases were fixed but most changes are for UX and flavor (see below for details). Check out the repo above to see a similar (possibly better) take on the same problem.


This is a template project for ML experimentation using wandb, hydra-zen, submitit on a Slurm cluster using Docker and Apptainer for containerization.

**NOTE**: This template is optimized for the specific setup of the ML Group cluster but may be easily adapted to similar settings.

## Highlights

* Apptainer containerization using [pip-poetry](https://python-poetry.org) to configure python evironment
* Logging and visualizations via [Weights and Biases](https://wandb.com)
* Reproducibility and modular type-checked configs via [hydra-zen](https://github.com/mit-ll-responsible-ai/hydra-zen)
* Submit Slurm jobs and sweeps directly from Python via [submitit](https://github.com/facebookincubator/submitit)


## Main differences to [main repo](https://github.com/marvinsxtr/ml-project-template)

* No interactive shell in the container needed - everything is run by using the script itself as an executable
* Freely change the main function - the addon features' configuration is only added on top of your own config
* Using poetry as package manager instead of conda - dependencies are stored in the pyproject.toml
* Using apptainer to be able to update the container directly on the cluster
* Simplified structure - managed to get rid of some code by using built-in Hydra functionality


## Setup

### Step 0: Set up passwordless SSH
First you need to set up passwordless SSH to the node from which you want to run the SLURM commands from. This is usually the headnode of the SLURM cluster since it has SLURM and most often passwordless SHH already activated.

### Step 1: Env Vars
To be able to run Slurm from within Apptainer, you first have to add the following lines to your `.zshrc`/`.bashrc` file:

```bash
export COMMAND_EXECUTION_NODE= <hostname where you want to run the slurm commands from, usually the headnose >
```
If you don't set this variable, $HOSTNAME will be used as a hostname.

### Step 2 (optional): Setup Aliases
The following aliases for interacting with Apptainer are recommended:

```bash
rebuild='apptainer build --nv container.sif container.def'
add-dep='apptainer run --nv container.sif uv add --locked'
remove-dep='apptainer run --nv container.sif uv remove --locked'
```

### Step 3: Configure WandB logging
Logging to WandB is optional for running local jobs but mandatory for jobs submitted to the cluster.

WandB is enabled by specifying an API key, the project and entity. Rename `example.env` to `.env` file in the root of the repository. Fill in your information.

### Step 4: Build the container
Build the container using the `rebuild` alias above - if there are any errors try deleting the poetry.lock file and repeat.

## Run

### Local (dev container or host)

Without Apptainer, launch scripts directly with `uv run`:
```bash
uv run python scripts/train.py
```
Hydra writes run artifacts under `outputs/<date>/<run_id>/.hydra` relative to the current working directory (`<run_id>` is the SLURM/local job id on the cluster, or a short uuid otherwise).

To log to WandB, add `cfg/wandb=log`:
```bash
uv run python scripts/train.py cfg/wandb=log
```

In order to use WandB in offline mode, add `cfg.wandb.mode=offline`:
```bash
uv run python scripts/train.py cfg/wandb=log cfg.wandb.mode=offline
```

> The shebang form `./scripts/train.py ...` expects Apptainer and is intended for Slurm/cluster deployments.

### Cluster profiles

Per-cluster settings (Slurm partitions, apptainer binds, node-local scratch, resource defaults, wall-clock limits) live as pluggable profiles under `scripts/conf/clusters/`. Select one with `cfg/cluster=<name>`:

* `local` — submitit's local executor (host subprocesses, no Slurm/Apptainer). The default, handy for smoke-testing the submission path. GPUs are auto-detected and sliced across sweep workers.
* `example` — a template Slurm profile. **Copy this to `scripts/conf/clusters/<your-cluster>.py`**, edit the binds/partitions/time limits, register it in `scripts/conf/clusters/__init__.py`, and use it via `cfg/cluster=<your-cluster>`.

Per-job resource overrides go on `cfg.job.slurm_config.*` (e.g. `cfg.job.slurm_config.gpus_per_task=4`); anything left unset falls back to the selected profile.

### Single Job

To run the command as a job on the cluster, run

```bash
./scripts/train.py cfg/job=run cfg/cluster=<your-cluster>
```

This will automatically add WandB logging for you. See `scripts/conf/base_conf.py` to configure the job to your needs.

### Distributed Sweep

Run a sweep using multiple workers:

```bash
./scripts/train.py cfg/job=sweep cfg/cluster=<your-cluster>
```

This will automatically add WandB logging for you. See `scripts/conf/base_conf.py` to configure the sweep to your needs. To smoke-test the sweep locally first, swap in `cfg/cluster=local`.

## Edit main function

You can make changes to the main function's signature or rename the function. Hydra-zen will automatically detect all arguments with standard types values as long as you give a default value.
You can also add config groups and all other hydra-zen functionality.

#### **However, you should not remove or rename the first argument as it is needed to configure sweeps, wandb, etc.** 

(You can if you know the code well, but be warned! Be careful making changes to this - in a very unlucky case changing BaseConfigs structure can lead to recursive runs on your infratructure!)


```python
#! /usr/bin/env -S apptainer exec --nv container.sif uv run --no-dev python

from loguru import logger

from conf.base_conf import configure_main, BaseConfig
from lib.utils.run import run


@configure_main()
def train(
    cfg: BaseConfig,  # you must keep this argument <----
    bar: int = 42,
    foo: str = "hello",
    jup: bool = False,
    test: float = 2.2,
) -> None:
    logger.info("Running main function.")
    logger.info(f"Config: bar={bar}, foo={foo}, jup={jup}")
    logger.info(f"BaseConfig: {cfg}")


if __name__ == "__main__":
    run(train)

```
Of course you can also add more scripts - to work they only need:
1. The first line (`#! /usr/bin/env -S apptainer exec --nv container.sif uv run --no-dev python`)
2. The `configure_main()` decorator over the main function
3. Running the main function with the `run` function

The store name is derived from the module name, so multiple entry points (e.g. `train.py`, `sample.py`) can register in the same process without clashing.


## Edit dependencies
Edit dependencies in the root dir of the repo with the aliases defined above, e.g.:
```bash
add-dep wandb
remove-dep numpy
```
After editing dependencies rebuild using the ```rebuild``` alias.


### Legacy version

This template is based on a [previous example project](https://github.com/mx-e/example_project_ml_cluster).