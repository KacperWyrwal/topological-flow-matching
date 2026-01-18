import torch
from torch import Tensor
from topofm.odes.ode import ODE
from topofm.distributions.covariance import Covariance


class TrivialODE(ODE):
    """
    dx_t = 0 dt
    """
    def __init__(self, **kwargs) -> None:
        super().__init__()

    def b(self, t: Tensor, xt: Tensor) -> Tensor:
        return torch.zeros_like(xt)

    def s(self, t: Tensor, v: Tensor) -> Tensor:
        return v

    def v(self, x0: Tensor, x1: Tensor) -> Tensor:
        return x1 - x0

    def x(self, t: Tensor, x0: Tensor, x1: Tensor) -> Tensor:
        return (1.0 - t) * x0 + t * x1

    def c(self, x0: Tensor, x1: Tensor) -> Tensor:
        """
        Args:
            x0: (..., n, d)
            x1: (..., m, d)
        Returns:
            c: (..., n, m)
        """
        return torch.cdist(x0, x1, p=2).square()

    def Phi10(self, x: Tensor | Covariance) -> Tensor | Covariance:
        return x
