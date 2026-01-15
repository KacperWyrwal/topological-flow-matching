from ..ode import ODE
from ..models import Model
from ..distributions.time import TimeDistribution
from ..data import FMDataLoader
from .fm_loss import FMLoss
from torch.optim import Optimizer


def train(
    *,
    ode: ODE,
    model: Model,
    time_distribution: TimeDistribution,
    fm_data_loader: FMDataLoader,
    optimizer: Optimizer,
) -> float:
    objective = FMLoss(ode=ode, model=model, time_distribution=time_distribution)
    model.train()

    total_loss = 0.0
    for x0, x1 in fm_data_loader:
        loss = objective(x0=x0, x1=x1)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.detach().item()
    return total_loss / fm_data_loader.num_samples
