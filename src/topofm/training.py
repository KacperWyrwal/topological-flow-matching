from collections import defaultdict
import tqdm
import torch
from torch_ema import ExponentialMovingAverage

from .control import ModelControl, bridge_control
from .ot import OTSolver, wasserstein_distance
from .solvers import SDESolver, EulerMaruyamaSolver
from .sde import SDE
from .time import UniformTimeSteps
from .data import MatchingDataLoader, TimeSampler, DiscreteTimeSampler


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
    data_loader: MatchingDataLoader,
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
    data_loader: MatchingDataLoader,
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
    train_data_loader: MatchingDataLoader,
    *,
    num_epochs: int,
    time_sampler: TimeSampler | None = None,
    sde_solver: SDESolver | None = None,
    optimizer: torch.optim.Optimizer | None = None,
    ema: ExponentialMovingAverage | None = None,
    objective: torch.nn.Module | None = None,
    eval_data_loader: MatchingDataLoader | None = None,
) -> dict[str, torch.Tensor]:
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
        avg_loss = train(
            sde=sde,
            model=model,
            data_loader=train_data_loader,
            optimizer=optimizer,
            ema=ema,
            objective=objective,
            time_sampler=time_sampler,
        )
        if eval_data_loader is not None:
            metrics = evaluate(
                sde_solver=sde_solver,
                model=model,
                data_loader=eval_data_loader,
                ema=ema,
            )
        else:
            metrics = {}
        metrics["loss"] = avg_loss
        for k, v in metrics.items():
            history[k].append(v)
        pbar.set_postfix(metrics)
    return history


