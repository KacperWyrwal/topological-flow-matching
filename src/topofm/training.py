from collections import defaultdict
from pandas.core.strings.accessor import NoNewAttributesMixin
import tqdm
import torch
from torch_ema import ExponentialMovingAverage
from dataclasses import dataclass
from typing import Optional

from .control import ModelControl, bridge_control
from .ot import OTSolver, wasserstein_distance
from .sde_solvers import SDESolver, EulerMaruyamaSolver
from .sde import SDE
from .time import UniformTimeSteps
from .data import TimeSampler, DiscreteTimeSampler, MatchingTrainLoader, MatchingTestLoader


@dataclass
class EarlyStopping:
    metric_name: str = 'W2'
    patience: int = 5
    min_delta: float = 0.05  # 5% relative improvement
    mode: str = "min"
    
    def __post_init__(self):
        assert self.mode in ["min", "max"], "Mode must be either 'min' or 'max'"
        assert self.min_delta >= 0, "Min delta must be non-negative"

        self.best_score = None
        self.counter = 0
        self.best_params: Optional[dict] = None
    
    def update(self, metrics: dict, model) -> bool:
        """Returns True if training should stop"""
        assert self.metric_name in metrics, f"Metric '{self.metric_name}' not found in validation metrics"
        
        current_score = metrics[self.metric_name]

        if self.best_score is None:
            improved = True
        elif self.mode == "min":
            # For minimization: stop if improvement is less than min_delta% of current best
            relative_improvement = (self.best_score - current_score) / abs(self.best_score) if self.best_score != 0 else 0
            improved = relative_improvement > self.min_delta
        else:
            # For maximization: stop if improvement is less than min_delta% of current best
            relative_improvement = (current_score - self.best_score) / abs(self.best_score) if self.best_score != 0 else 0
            improved = relative_improvement > self.min_delta
            
        if improved:
            self.best_score = current_score
            self.counter = 0
            self.best_params = {name: param.clone().detach() for name, param in model.named_parameters()}
        else:
            self.counter += 1
            
        return self.counter >= self.patience
    
    def restore_best(self, model):
        """Restore model to best parameters"""
        if self.best_params is not None:
            for name, param in model.named_parameters():
                if name in self.best_params:
                    param.data.copy_(self.best_params[name])


# Defaults
EMA_DECAY = 0.99
ADAMW_LR = 1e-4
WASSERSTEIN_DISTANCE_BLUR = 0.05
EULER_MARUYAMA_NUM_STEPS = 1000
DISCRETE_TIME_SAMPLER_NUM_STEPS = 1000
OT_SOLVER_NORMALIZE_VARIANCE = False


def make_optimizer(model: torch.nn.Module, *, lr: float = ADAMW_LR) -> torch.optim.Optimizer:
    return torch.optim.AdamW(model.parameters(), lr=lr)


def make_ema(model: torch.nn.Module, *, decay: float = EMA_DECAY) -> ExponentialMovingAverage:
    return ExponentialMovingAverage(model.parameters(), decay=decay)


def make_objective() -> torch.nn.Module:
    return torch.nn.MSELoss()


def make_sde_solver(sde: SDE, *, num_steps: int = EULER_MARUYAMA_NUM_STEPS) -> SDESolver:
    time_steps = UniformTimeSteps(n=num_steps)
    return EulerMaruyamaSolver(sde=sde, time_steps=time_steps)


def make_ot_solver(sde: SDE, *, normalize_variance: bool = OT_SOLVER_NORMALIZE_VARIANCE) -> OTSolver:
    return OTSolver(sde=sde, normalize_variance=normalize_variance)


def make_time_sampler(*, num_steps: int = DISCRETE_TIME_SAMPLER_NUM_STEPS) -> TimeSampler:
    time_steps = UniformTimeSteps(n=num_steps)
    return DiscreteTimeSampler(time_steps=time_steps)


def train(
    *,
    sde: SDE,
    model: torch.nn.Module,
    data_loader: MatchingTrainLoader,
    time_sampler: TimeSampler,
    optimizer: torch.optim.Optimizer,
    ema: ExponentialMovingAverage,
    objective: torch.nn.Module,
) -> float:
    epoch_loss = 0.0
    for x0, x1 in data_loader:
        t = time_sampler.sample((data_loader.batch_size,))
        x = sde.marginal_distribution(t=t, x0=x0, x1=x1).sample()
        ut = bridge_control(sde=sde, t=t, x=x, x1=x1)
        ut_preds = model(t, x)
        loss = objective(ut, ut_preds)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        ema.update()
        epoch_loss += loss.detach().item()
    return epoch_loss / len(data_loader)


@torch.inference_mode()
def evaluate(
    *,
    sde_solver: SDESolver,
    model: torch.nn.Module,
    data_loader: MatchingTestLoader,
    ema: ExponentialMovingAverage,
) -> dict[str, float]:
    control = ModelControl(model)
    W1 = 0.0
    W2 = 0.0
    for x0, x1 in data_loader:
        x1_pred = sde_solver.pushforward(x0=x0, control=control)
        W1 += wasserstein_distance(x0=x1_pred, x1=x1, p=1).item()
        W2 += wasserstein_distance(x0=x1_pred, x1=x1, p=2).item()
    W1 /= data_loader.num_batches
    W2 /= data_loader.num_batches
    return {"W1": W1, "W2": W2}


def fit(
    sde: SDE,
    model: torch.nn.Module,
    train_data_loader: MatchingTrainLoader,
    *,
    num_epochs: int,
    time_sampler: TimeSampler | None = None,
    sde_solver: SDESolver | None = None,
    optimizer: torch.optim.Optimizer | None = None,
    ema: ExponentialMovingAverage | None = None,
    objective: torch.nn.Module | None = None,
    eval_data_loader: MatchingTestLoader | None = None,
    early_stopping: EarlyStopping | None = None,
    use_wandb: bool = False,
    wandb_run = None,
) -> dict[str, torch.Tensor]:
    # Early stopping requires validation data
    assert early_stopping is None or eval_data_loader is not None, "Early stopping requires eval_data_loader"
    
    if optimizer is None:
        optimizer = make_optimizer(model)
    if ema is None:
        ema = make_ema(model)
    if objective is None:
        objective = make_objective()
    if sde_solver is None:
        sde_solver = make_sde_solver(sde)
    if time_sampler is None:
        time_sampler = make_time_sampler()

    history = defaultdict(list)
    pbar = tqdm.trange(num_epochs, desc="Epochs")
    break_early = False
    for epoch in pbar:
        # Training epoch
        avg_loss = train(
            sde=sde,
            model=model,
            data_loader=train_data_loader,
            optimizer=optimizer,
            ema=ema,
            objective=objective,
            time_sampler=time_sampler,
        )
        # Validation epoch
        if eval_data_loader is not None:
            metrics = evaluate(
                sde_solver=sde_solver,
                model=model,
                data_loader=eval_data_loader,
                ema=ema,
            )
        else:
            metrics = {}

        # Update history
        metrics["loss"] = avg_loss
        for k, v in metrics.items():
            history[k].append(v)        
        if early_stopping is not None:
            break_early = early_stopping.update(metrics, model)
            metrics[f"best_{early_stopping.metric_name}"] = early_stopping.best_score

        # Update progress bar
        pbar.set_postfix(metrics)

        # Log to W&B if enabled
        if use_wandb and wandb_run is not None:
            # Log training metrics with consistent naming
            wandb_run.log({
                "epoch": epoch,
                "train/loss": avg_loss,
                "val/W1": metrics.get("W1", 0.0),
                "val/W2": metrics.get("W2", 0.0),
                f"val/best_{early_stopping.metric_name}": metrics.get(f"best_{early_stopping.metric_name}", 0.0) if early_stopping else 0.0,
            })

        # Early stopping
        if break_early:
            print(f"Early stopping triggered after {epoch + 1} epochs")
            break
    
    # Restore best model
    if early_stopping is not None:
        assert early_stopping.best_params is not None, "Early stopping should always have best_params when used"
        early_stopping.restore_best(model)
    
    # Log final training state to W&B
    if use_wandb and wandb_run is not None:
        wandb_run.log({
            "final_epoch": epoch + 1,
        })
        # Store early stopping info in run summary (not plotted)
        wandb_run.summary["early_stopping_triggered"] = break_early
    
    return history


