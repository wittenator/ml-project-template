from src.common.logging.logger import logger
from src.common.utils.config import run
from src.runs.base.run import Run


def main(cfg: Run) -> None:
    """Run a main function from a config.

    Args:
    ----
        cfg: Config to run.
    """
    logger.info("Hello World!")


if __name__ == "__main__":
    import src.configs.stores.main  # noqa: F401

    run(main)
