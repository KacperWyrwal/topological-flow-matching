import torch
from torch import nn, Tensor
from topofm.distributions.time import TimeDistribution
from topofm.odes.ode import ODE
from topofm.models.model import Model, ModelMode


class FMLoss(nn.Module):
    """
    Flow matching loss.

    Args:
        ode: The ODE.
        model: The model.
        time_distribution: The time distribution.
    """
    def __init__(self, ode: ODE, model: Model, time_distribution: TimeDistribution) -> None:
        super().__init__()
        self.ode = ode
        self.model = model
        self.time_distribution = time_distribution

    def target(self, t: Tensor, x0: Tensor, x1: Tensor) -> Tensor:
        if self.model.mode == ModelMode.V:
            return self.ode.v(x0=x0, x1=x1)
        if self.model.mode == ModelMode.SV:
            return self.ode.sv(t=t, x0=x0, x1=x1)

    def weight(self, t: Tensor, diff: Tensor) -> Tensor:
        """
        Weight the difference between the target and the prediction 
        according to the model mode. If the model is in V mode, then 
        weight it by the ODE's s function; otherwise, remain unweighted.

        Args:
            t: (..., 1)
            diff: (..., d)
        Returns:
            weight: (..., 1)
        """
        if self.model.mode == ModelMode.V:
            return self.ode.s(t=t, v=diff)
        if self.model.mode == ModelMode.SV:
            return diff

    def forward(self, x0: Tensor, x1: Tensor) -> Tensor:
        """
        Args:
            x0: (..., d)
            x1: (..., d)
        Returns:
            loss: ()
        """
        batch_size = x0.shape[:-1]

        t = self.time_distribution.sample(batch_size)
        xt = self.ode.x(t=t, x0=x0, x1=x1)
        target = self.target(t=t, x0=x0, x1=x1)
        pred = self.model(t=t, xt=xt)
        diff = target - pred
        return torch.mean(self.weight(t=t, diff=diff).square())
