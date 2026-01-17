import logging
import torch
from pathlib import Path
from omegaconf import DictConfig, OmegaConf
from hydra.utils import instantiate
from hydra import main
from topofm.data.data_loaders import TestFMDataLoader, TrainFMDataLoader
from topofm import engine
from topofm.utils import seed_everything, to_dtype
from topofm.config import CONFIGS_DIR


def multiply_resolver(*args):
    result = 1.0
    for arg in args:
        result *= float(arg)
    return result

OmegaConf.register_new_resolver("mult", multiply_resolver)


log = logging.getLogger(__name__)


def train_and_test(cfg: DictConfig) -> None:
    # Set seed
    seed = cfg.seed
    seed_everything(seed)
    log.info(f"Seed: {seed}")

    # Parse device
    device = torch.device(cfg.device)
    log.info(f"Device: {device}")
    
    # Parse dtype
    dtype = to_dtype(cfg.dtype)
    log.info(f"Data type: {dtype}")

    # Callbacks
    early_stopping = instantiate(cfg.train.early_stopping)
    model_checkpoint = instantiate(cfg.train.checkpoint)

    # Logger
    logger = instantiate(cfg.logger, run_config=cfg)

    # Dataset
    dataset = instantiate(cfg.dataset, device=device, dtype=dtype)
    frame = dataset.frame
    eigvals = dataset.eigvals
    train_dataset, val_dataset, test_dataset = dataset.split(cfg.split, seed=seed)

    # ODE
    ode = instantiate(cfg.ode, frame=frame, eigvals=eigvals)

    # Coupling
    coupling = instantiate(cfg.train.coupling, mu0=train_dataset.mu0, mu1=train_dataset.mu1, ode=ode)

    # Data loaders
    train_loader = TrainFMDataLoader(
        coupling=coupling,
        batch_size=cfg.train.batch_size,
        num_samples=cfg.train.num_samples,
    )
    val_loader = TestFMDataLoader(
        dataset=val_dataset,
        batch_size=cfg.train.val.batch_size,
        num_samples=cfg.train.val.num_samples,
    )
    test_loader = TestFMDataLoader(
        dataset=test_dataset,
        batch_size=cfg.test.batch_size,
        num_samples=cfg.test.num_samples,
    )

    # Model
    model = instantiate(cfg.model, input_dim=frame.ambient_dim)
    model.to(device=device, dtype=dtype)

    # Optimizer
    optimizer = instantiate(cfg.train.optimizer, params=model.parameters())

    # Training
    engine.train(
        model=model,
        ode=ode,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        early_stopping=early_stopping,
        model_checkpoint=model_checkpoint,
        logger=logger,
        val_every_n_samples=cfg.train.val.every_n_samples,
    )

    test_metrics = engine.evaluate(
        model=model,
        ode=ode,
        data_loader=test_loader,
    )
    logger.log({f'test/{m}': v for m, v in test_metrics.items()}, step=cfg.test.num_samples)
    logger.finish()


def train(cfg: DictConfig) -> None:
    raise NotImplementedError


def test(cfg: DictConfig) -> None:
    raise NotImplementedError


@main(config_path=str(CONFIGS_DIR), config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    log.info(f"Running experiment with config:\n{OmegaConf.to_yaml(cfg, resolve=True)}")
    if cfg.mode == "train_and_test":
        train_and_test(cfg)
    elif cfg.mode == "train":
        train(cfg)
    elif cfg.mode == "test":
        test(cfg)
    else:
        raise ValueError(f"Unknown mode: {cfg.mode}")

if __name__ == "__main__":
    main()
