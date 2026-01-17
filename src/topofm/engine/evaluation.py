import torch
from topofm.models import Model
from topofm.ode_solvers import ODESolver
from topofm.utils import preserve_mode
from topofm.metrics import wasserstein_distance
from topofm.data.data_loaders import TestFMDataLoader
from topofm.odes import ODE, NeuralODE


@torch.inference_mode()
def evaluate(
    model: Model, 
    ode: ODE,
    data_loader: TestFMDataLoader,
) -> dict[str, float]:
    """
    Evaluate the model on a test set.

    Args:
        model: The model to use.
        ode: The ODE to use.
        data_loader: The data loader to use.
    Returns:
        metrics: A dictionary of metrics.
    """
    with preserve_mode(model):
        model.eval()

        neural_ode = NeuralODE(base_ode=ode, model=model)
        solver = ODESolver(ode=neural_ode)

        # Predict x1 from x0
        x1_pred, x1 = [], []
        for x0_batch, x1_batch in data_loader:
            x1_pred.append(solver.x1(x0=x0_batch))
            x1.append(x1_batch)
        x1_pred = torch.cat(x1_pred, dim=0)
        x1 = torch.cat(x1, dim=0)

        # Compute validation metrics
        metrics = {
            "w1": wasserstein_distance(x1_pred, x1, p=1, epsilon=0.05),
            "w2": wasserstein_distance(x1_pred, x1, p=2, epsilon=0.05),
            "w1_exact": wasserstein_distance(x1_pred, x1, p=1, epsilon=0.0),
            "w2_exact": wasserstein_distance(x1_pred, x1, p=2, epsilon=0.0),
        }

        return metrics
