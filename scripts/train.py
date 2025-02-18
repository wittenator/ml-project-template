#! /usr/bin/env -S apptainer exec ${GIT_DIR}/container.sif uv run python

import wandb
from conf.base_conf import BaseConfig, configure_main
from lib.utils.run import run
from loguru import logger

from scripts.lib.utils.log import log_dict


@configure_main(extra_defaults=[])
def train(
    cfg: BaseConfig,  # you must keep this argument
    cfg_version: str = "1.0",  # noqa: ARG001 saving config version to track changes in signatures eg for finetuning
    bar: int = 42,
    foo: str = "hello",
    jup: bool = False,
    test: float = 2.2,
) -> None:
    try:
        logger.info("Running main function.")
        logger.info(f"Config: bar={bar}, foo={foo}, jup={jup}")
        logger.info(f"BaseConfig: {cfg}")

        # log bar to wandb
        if cfg.wandb:
            log_dict({"loss": bar}, step=0, log_wandb=True)

    finally:
        if cfg.wandb:
            wandb.finish()


if __name__ == "__main__":
    run(train)
