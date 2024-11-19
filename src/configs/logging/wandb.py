from hydra_zen import builds

from src.common.logging.wandb import WandBRun

BaseWandBConfig = builds(WandBRun, group=None, mode="online")
