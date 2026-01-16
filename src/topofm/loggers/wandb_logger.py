import wandb
from omegaconf import DictConfig, OmegaConf
from topofm.loggers.logger import Logger


class WandbLogger(Logger):
    def __init__(self, enabled: bool, project: str, entity: str, run_name: str, config: DictConfig):
        super().__init__(enabled)
        if self.enabled:
            wandb_config = OmegaConf.to_container(config, resolve=True)
            
            wandb.init(
                project=project,
                entity=entity,
                name=run_name,
                config=wandb_config,
                reinit=True
            )

    def log(self, metrics: dict[str, float], step: int | None = None):
        if self.enabled:
            wandb.log(metrics, step=step)

    def finish(self):
        if self.enabled:
            wandb.finish()
