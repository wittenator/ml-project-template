from loguru import logger
from lib.utils.run_utils import run, Run


def main(cfg: Run) -> None:
    """Run a main function from a config.

    Args:
    ----
        cfg: Config to run.
    """
    logger.info("Hello World!")


if __name__ == "__main__":
    import src.conf.base_conf  # noqa: F401

    run(main)
