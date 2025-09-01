from collections import defaultdict
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
from .wandb_logger import WandBLogger
from .utils import single_cell_to_times, single_cell_to_phate
from .frames import Frame


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
    model.train()

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
    frame: Frame,
) -> dict[str, float]:
    with ema.average_parameters():
        model.eval()
        control = ModelControl(model)
        W1 = 0.0
        W2 = 0.0
        for x0, x1 in data_loader:
            # Push inputs forward using model control
            x1_pred = sde_solver.pushforward(x0=x0, control=control)

            # Convert to original frame, in case there is a dimensionality reduction
            x1_pred = frame.inverse_transform(x1_pred)
            x1 = frame.inverse_transform(x1)

            # Compute Wasserstein distance 
            W1 += wasserstein_distance(x0=x1_pred, x1=x1, p=1).item()
            W2 += wasserstein_distance(x0=x1_pred, x1=x1, p=2).item()
        W1 /= data_loader.num_batches
        W2 /= data_loader.num_batches
        return {"W1": W1, "W2": W2}


@torch.inference_mode()
def evaluate__single_cell(
    *,
    sde_solver: SDESolver,
    model: torch.nn.Module,
    data_loader: MatchingTestLoader,
    true_times: torch.Tensor,
    phate: torch.Tensor,
    ema: ExponentialMovingAverage,
    frame: Frame,
) -> dict[str, float]:
    with ema.average_parameters():
        model.eval()
        times = [0, 1, 2, 3, 4] # TODO maybe make this a parameter
        control = ModelControl(model)
        W1 = defaultdict(float)
        W2 = defaultdict(float)
        for x0, x1 in data_loader:
            x1_pred = sde_solver.pushforward(x0=x0, control=control)
            x1_pred = frame.inverse_transform(x1_pred)

            # FIXME Terrible 3am code 
            x1_pred = x1_pred.squeeze(0) # [D]
            x1_pred_times = single_cell_to_times(x1_pred, true_times) # [D]
            for t in times: 
                x1_pred_phate = single_cell_to_phate(phate=phate, times=x1_pred_times, t=t)
                x1_phate = single_cell_to_phate(phate=phate, times=true_times, t=t)

                assert x1_pred_phate.ndim == 2, f"x1_pred_phate must be 2D, got {x1_pred_phate.ndim}D"
                assert x1_phate.ndim == 2, f"x1_phate must be 2D, got {x1_phate.ndim}D"
                assert x1_pred_phate.shape == x1_phate.shape, f"x1_pred_phate and x1_phate must have the same shape, got {x1_pred_phate.shape} and {x1_phate.shape}"

                W1[t] += wasserstein_distance(x0=x1_pred_phate, x1=x1_phate, p=1).item()
                W2[t] += wasserstein_distance(x0=x1_pred_phate, x1=x1_phate, p=2).item()
        for t in times:
            W1[t] /= data_loader.num_batches
            W2[t] /= data_loader.num_batches
        return {**{f"W1_{t}": W1[t] for t in times}, **{f"W2_{t}": W2[t] for t in times}}



def fit__single_cell(
    sde: SDE,
    model: torch.nn.Module,
    train_data_loader: MatchingTrainLoader,
    true_times: torch.Tensor,
    phate: torch.Tensor,
    *,
    num_epochs: int,
    time_sampler: TimeSampler | None = None,
    sde_solver: SDESolver | None = None,
    optimizer: torch.optim.Optimizer | None = None,
    ema: ExponentialMovingAverage | None = None,
    objective: torch.nn.Module | None = None,
    eval_data_loader: MatchingTestLoader | None = None,
    early_stopping: EarlyStopping | None = None,
    logger: WandBLogger | None = None,
    frame: Frame | None = None,
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
    for epoch in pbar:
        # Training epoch
        avg_loss = train(
            sde=sde,
            model=model,
            data_loader=train_data_loader,
            optimizer=optimizer,
            objective=objective,
            ema=ema,
            time_sampler=time_sampler,
        )
        # Validation epoch
        if eval_data_loader is not None:
            eval_metrics = evaluate__single_cell(
                sde_solver=sde_solver,
                model=model,
                data_loader=eval_data_loader,
                true_times=true_times,
                ema=ema,
                phate=phate,
                frame=frame,
            )
        else:
            eval_metrics = {}

        # Bookkeeping
        metrics = {
            "loss": avg_loss,
            **eval_metrics,
        }
        for k, v in metrics.items():
            history[k].append(v)    
        pbar.set_postfix(metrics)

        # Log to W&B via logger if enabled (scalars + curves)
        if logger is not None and logger.is_enabled():
            logger.log_training(epoch=epoch, loss=avg_loss, eval_metrics=eval_metrics, history=history)

        # Early stopping
        if (
            early_stopping is not None 
            and early_stopping.update(metrics, model) is True
        ):
            print(f"Early stopping triggered after {epoch + 1} epochs")
            assert early_stopping.best_params is not None, "Early stopping should always have best_params when used"
            early_stopping.restore_best(model)
            break
    return history




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
    logger: WandBLogger | None = None,
    frame: Frame | None = None,
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
            eval_metrics = evaluate(
                sde_solver=sde_solver,
                model=model,
                data_loader=eval_data_loader,
                ema=ema,
                frame=frame,
            )
        else:
            eval_metrics = {}

        # Bookkeeping
        metrics = {
            "loss": avg_loss,
            **eval_metrics,
        }
        for k, v in metrics.items():
            history[k].append(v)    
        pbar.set_postfix(metrics)

        # Log to W&B via logger if enabled (scalars + curves)
        if logger is not None and logger.is_enabled():
            logger.log_training(epoch=epoch, loss=avg_loss, eval_metrics=eval_metrics, history=history)

        # Early stopping
        if (
            early_stopping is not None 
            and early_stopping.update(metrics, model) is True
        ):
            print(f"Early stopping triggered after {epoch + 1} epochs")
            assert early_stopping.best_params is not None, "Early stopping should always have best_params when used"
            early_stopping.restore_best(model)
            break
    return history


