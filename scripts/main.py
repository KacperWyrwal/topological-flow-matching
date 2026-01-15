import logging
import torch
from pathlib import Path
from omegaconf import DictConfig, OmegaConf
from hydra.utils import instantiate
from hydra import main
from tqdm import tqdm
from lightning.fabric import Fabric

# --- 0. Setup Logger ---
# Hydra configures this logger automatically based on your config directory
log = logging.getLogger(__name__)

# --- 1. Explicit, Enabled-Aware Callbacks ---

class EarlyStopping:
    def __init__(self, enabled: bool, monitor: str, mode: str = "min", patience: int = 5):
        self.enabled = enabled
        self.monitor = monitor
        self.mode = mode
        self.patience = patience
        self.counter = 0
        self.best_score = float('inf') if mode == "min" else float('-inf')
        self.should_stop = False

    def check(self, fabric: Fabric, metrics: dict):
        """
        Explicitly checks if training should stop.
        """
        if not self.enabled:
            return

        score = metrics.get(self.monitor)
        if score is None:
            if fabric.global_rank == 0:
                log.warning(f"[EarlyStopping] Monitor '{self.monitor}' not found in metrics.")
            return

        improved = (score < self.best_score) if self.mode == "min" else (score > self.best_score)
        if improved:
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
                if fabric.global_rank == 0:
                    log.info(f"🛑 [EarlyStopping] Triggered. Best {self.monitor}: {self.best_score:.4f}")


class ModelCheckpoint:
    def __init__(self, enabled: bool, dirpath: str, filename: str, monitor: str, mode: str = "min"):
        self.enabled = enabled
        self.dirpath = Path(dirpath)
        self.filename = filename
        self.monitor = monitor
        self.mode = mode
        self.best_score = float('inf') if mode == "min" else float('-inf')
        
        # Only create directory if actually enabled
        if self.enabled:
            self.dirpath.mkdir(parents=True, exist_ok=True)

    def save_if_best(self, fabric: Fabric, model: torch.nn.Module, metrics: dict, step: int):
        """
        Explicitly saves the model if the monitored metric has improved.
        """
        if not self.enabled:
            return

        score = metrics.get(self.monitor)
        if score is None:
            return

        improved = (score < self.best_score) if self.mode == "min" else (score > self.best_score)
        if improved:
            self.best_score = score
            save_path = self.dirpath / f"{self.filename}"
            
            # Create state dict
            state = {
                "model": model,
                "step": step,
                "metrics": metrics,
            }
            
            # Use Fabric to save (handles unwrapping DDP models automatically)
            fabric.save(save_path, state)
            
            if fabric.global_rank == 0:
                log.info(f"💾 [ModelCheckpoint] Saved new best model ({self.monitor}={score:.4f}) to {save_path}")


# --- 2. Imports and Configuration ---
from topofm.data.data_loaders.test_fm_data_loader import TestFMDataLoader
from topofm.data.data_loaders.train_fm_data_loader import TrainFMDataLoader
from topofm.loss.fm_loss import FMLoss
from topofm.ode_solvers.ode_solver import ODESolver
from topofm.odes.neural_ode import NeuralODE

# Hydra Resolvers
OmegaConf.register_new_resolver("device", lambda x: torch.device(x))
OmegaConf.register_new_resolver("dtype", lambda x: getattr(torch, x))


def train(cfg: DictConfig) -> None:
    # --- A. Instantiate Callbacks Explicitly ---
    early_stopping = EarlyStopping(
        enabled=cfg.training.early_stopping.enabled,
        monitor=cfg.training.early_stopping.monitor,
        mode=cfg.training.early_stopping.mode,
        patience=cfg.training.early_stopping.patience,
    )

    checkpoint = ModelCheckpoint(
        enabled=cfg.training.checkpoint.enabled,
        dirpath=cfg.training.checkpoint.dirpath,
        filename=cfg.training.checkpoint.filename,
        monitor=cfg.training.checkpoint.monitor,
        mode=cfg.training.checkpoint.mode,
    )

    # --- B. Setup Fabric ---
    logger = instantiate(cfg.training.logger)
    fabric = Fabric(
        accelerator=cfg.training.accelerator,
        devices=cfg.training.devices,
        loggers=[logger],
    )
    fabric.launch()

    # --- C. Data Loading (Custom Loaders) ---
    dataset = instantiate(cfg.dataset)
    frame = dataset.frame
    train_dataset, val_dataset, _ = dataset.split(cfg.split)
    
    # NOTE: Since these are custom loaders, we do NOT use fabric.setup_dataloaders()
    train_loader = TrainFMDataLoader(train_dataset, batch_size=cfg.train.batch_size, num_samples=cfg.train.num_samples)
    val_loader = TestFMDataLoader(val_dataset, batch_size=cfg.val.batch_size, num_samples=cfg.val.num_samples)

    # --- D. Model & Solver Setup ---
    ode = instantiate(cfg.ode, frame=frame)
    model = instantiate(cfg.model)
    neural_ode = NeuralODE(base_ode=ode, model=model)
    solver = instantiate(cfg.solver, ode=neural_ode)

    time_distribution = instantiate(cfg.loss.time_distribution)
    criterion = FMLoss(ode=ode, model=model, time_distribution=time_distribution)    

    optimizer = instantiate(cfg.optimizer, params=model.parameters())

    # --- E. Fabric Model Setup ---
    # Required for device movement and precision handling (float16/bf16)
    model, optimizer = fabric.setup(model, optimizer)

    # --- F. Training Loop ---
    samples_processed = 0
    val_frequency = cfg.training.val_every_n_samples
    next_val_threshold = val_frequency

    if fabric.global_rank == 0:
        log.info("Starting training loop...")

    model.train()
    # Only show progress bar on main process
    pbar = tqdm(train_loader, desc="Training", disable=fabric.global_rank != 0)
    
    for x0, x1 in pbar:
        # Manual Device Placement (Required because we skipped setup_dataloaders)
        x0 = fabric.to_device(x0)
        x1 = fabric.to_device(x1)

        # Optimization Step
        optimizer.zero_grad()
        loss = criterion(x0, x1)
        fabric.backward(loss)
        optimizer.step()
        
        samples_processed += len(x0)
        pbar.set_postfix({"loss": loss.item()})

        # --- G. Validation & Callbacks ---
        if samples_processed >= next_val_threshold:
            model.eval()
            
            # Run validation
            metrics = validate(model, solver, val_loader, fabric)
            
            model.train()

            # Log metrics to TensorBoard/WandB
            fabric.log_dict(metrics)

            # Explicit Callback Calls
            checkpoint.save_if_best(fabric, model, metrics, step=samples_processed)
            early_stopping.check(fabric, metrics)
            
            if early_stopping.should_stop:
                if fabric.global_rank == 0:
                    log.info("Stopping training due to EarlyStopping.")
                break

            next_val_threshold += val_frequency


def train(
    model: Model,
    solver: ODESolver,
    train_loader: TrainFMDataLoader,
    val_loader: TestFMDataLoader,
    optimizer: Optimizer,
    fabric: Fabric,
    early_stopping: EarlyStopping,
    checkpoint: ModelCheckpoint,
    samples_processed: int,
    next_val_threshold: int,
    val_frequency: int,
    criterion: FMLoss,
) -> None:
    model.train()
    # Only show progress bar on main process
    pbar = tqdm(train_loader, desc="Training", disable=fabric.global_rank != 0)
    
    for x0, x1 in pbar:
        # Manual Device Placement (Required because we skipped setup_dataloaders)
        x0 = fabric.to_device(x0)
        x1 = fabric.to_device(x1)

        # Optimization Step
        optimizer.zero_grad()
        loss = criterion(x0, x1)
        loss.backward()
        optimizer.step()
        
        samples_processed += len(x0)
        pbar.set_postfix({"loss": loss.item()})

        # --- G. Validation & Callbacks ---
        if samples_processed >= next_val_threshold:
            model.eval()
            
            # Run validation
            metrics = validate(model, solver, val_loader, fabric)
            
            model.train()

            # Log metrics to TensorBoard/WandB
            fabric.log_dict(metrics)

            # Explicit Callback Calls
            checkpoint.save_if_best(fabric, model, metrics, step=samples_processed)
            early_stopping.check(fabric, metrics)
            
            if early_stopping.should_stop:
                if fabric.global_rank == 0:
                    log.info("Stopping training due to EarlyStopping.")
                break

            next_val_threshold += val_frequency


@torch.inference_mode()
def validate(
    model: Model,
    solver: ODESolver,
    data_loader: TestFMDataLoader,
) -> dict[str, float]:
    model.eval()
    # Predict x1 from x0
    x1_pred, x1 = [], []
    for x0_batch, x1_batch in data_loader:
        x1_pred.append(model(x0_batch))
        x1.append(x1_batch)
    x1_pred = torch.cat(x1_pred, dim=0)
    x1 = torch.cat(x1, dim=0)

    # Compute validation metrics
    metrics = {
        "w1": wasserstein_distance(x1_pred, x1, p=1),
        "w2": wasserstein_distance(x1_pred, x1, p=2),
    }
    return metrics


@torch.inference_mode()
def test(cfg: DictConfig) -> None:
    raise NotImplementedError


@main(config_path="config", config_name="config")
def main(cfg: DictConfig) -> None:
    if cfg.mode == "train_and_test":
        train(cfg)
        test(cfg)
    elif cfg.mode == "train":
        train(cfg)
    elif cfg.mode == "test":
        test(cfg)
    else:
        raise ValueError(f"Unknown mode: {cfg.mode}")

if __name__ == "__main__":
    main()
