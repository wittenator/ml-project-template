from hydra_zen import store

from src.configs.logging.wandb import BaseWandBConfig
from src.configs.runs.base import BaseJobConfig, BaseRunConfig, BaseSweepConfig

run_config_store = store(group="cfg", hydra_defaults=["_self_", {"wandb": None}, {"job": None}])
run_config_store(BaseRunConfig, name="base")

wandb_config_store = store(group="cfg/wandb")
wandb_config_store(BaseWandBConfig, name="base")

job_config_store = store(group="cfg/job")
job_config_store(BaseJobConfig, name="base")
job_config_store(BaseSweepConfig, name="sweep")
