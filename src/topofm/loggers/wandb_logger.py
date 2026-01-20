import wandb
from omegaconf import DictConfig, OmegaConf
from topofm.loggers.logger import Logger


class WandbLogger(Logger):
    def __init__(
        self, 
        enabled: bool, 
        project: str, 
        entity: str, 
        run_name: str, 
        run_config: DictConfig,
        tags: list[str] | None = None,
    ):
        """
        Args:
            enabled: Whether to enable the logger.
            project: The project name.
            entity: The entity name.
            run_name: The run name.
            run_config: The configuration.
        """
        super().__init__(enabled)
        if self.enabled:
            wandb_config = OmegaConf.to_container(run_config, resolve=True)
            
            wandb.init(
                project=project,
                entity=entity,
                name=run_name,
                config=wandb_config,
                tags=tags,
                reinit=True
            )

    def log(self, metrics: dict[str, float], step: int | None = None):
        """
        Args:
            metrics: The metrics to log.
            step: The step to log.
        """
        if self.enabled:
            wandb.log(metrics, step=step)

    def finish(self):
        if self.enabled:
            wandb.finish()
