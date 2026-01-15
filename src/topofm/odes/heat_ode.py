import torch
from torch import Tensor

from topofm.odes.ode import ODE, SpectralBaseODE
from topofm.odes.trivial_ode import TrivialODE
from topofm.frames.frame import Frame


class _PositiveEigenvalueSpectralHeatODE(ODE):
    def __init__(self, kappa: float, eigvals: Tensor) -> None:
        """
        Args:
            kappa: float
            eigvals: (E)
        """
        super().__init__()
        self.D = -kappa * eigvals

    def b(self, t: Tensor, xt: Tensor) -> Tensor:
        """
        b(t, x_t) = D x_t

        Args:
            t: (..., 1)
            x: (..., d)
        Returns:
            b: (..., d)
        """
        return self.D * xt

    def s(self, t: Tensor, v: Tensor) -> Tensor:
        """
        s(t)v(t, x_t) = D exp(-t D) / sinh(D) v(t, x_t)

        Args:
            t: (..., 1)
            v: (..., d)
        Returns:
            sv: (..., d)
        """
        return (
            (self.D * torch.exp(-t * self.D) / torch.sinh(self.D)) * v
        )

    def v(self, x0: Tensor, x1: Tensor) -> Tensor:
        """
        v(t, x_t) = x_1 - exp(D) x_0

        Args:
            x0: (..., d)
            x1: (..., d)
        Returns:
            v: (..., d)
        """
        return (
            x1 - 
            torch.exp(self.D) * x0
        )

    def sv(self, t: Tensor, x0: Tensor, x1: Tensor) -> Tensor:
        """
        s(t) v(t, x_t) = D exp(-t D) / sinh(D) v(t, x_t)

        Args:
            t: (..., 1)
            x0: (..., d)
            x1: (..., d)
        Returns:
            sv: (..., d)
        """
        return (
            ((self.D * torch.exp(-t * self.D) / torch.sinh(self.D)) * x1) - 
            ((self.D * torch.exp((1.0 - t) * self.D) / torch.sinh(self.D)) * x0)
        )
    
    def x(self, t: Tensor, x0: Tensor, x1: Tensor) -> Tensor:
        """
        x_t = (sinh((1 - t) D) x_0 + sinh(t D) x_1) / sinh(D)

        Args:
            t: (..., 1)
            x0: (..., d)
            x1: (..., d)
        Returns:
            x: (..., d)
        """
        return (
            (
                (torch.sinh((1 - t) * self.D) * x0) + 
                (torch.sinh(t * self.D) * x1)
            ) /
            torch.sinh(self.D)
        )
    
    def _c_transform(self, x0: Tensor, x1: Tensor) -> tuple[Tensor, Tensor]:
        """
        Args:
            x0: (..., d)
            x1: (..., d)
        Returns:
            z0: (..., d)
            z1: (..., d)
        """
        return (
            (self.D * torch.exp(self.D) / torch.sinh(self.D)).sqrt() * x0,
            (self.D * torch.exp(-self.D) / torch.sinh(self.D)).sqrt() * x1
        )

    def c(self, x0: Tensor, x1: Tensor) -> Tensor:
        """
        c(x_0, x_1) = ||x_1 - exp(D) x_0||^2_{2 D exp(-D) / sinh(D)}

        Args:
            x0: (..., n, d)
            x1: (..., m, d)
        Returns:
            c: (..., n, m)
        """
        z0, z1 = self._c_transform(x0, x1)
        return torch.cdist(z0, z1, p=2).square()


class _SpectralHeatODE(ODE):

    def __init__(self, kappa: float, eigvals: Tensor) -> None:
        """
        Args:
            kappa: float
            eigvals: (E,)
        """
        super().__init__()
        # TODO Maybe use a small epsilon instead of 0.0
        self.zero_eigenvalue_mask = eigvals == 0.0
        safe_eigvals = torch.where(self.zero_eigenvalue_mask, torch.ones_like(eigvals), eigvals)
        self.positive_eigenvalue_ode = _PositiveEigenvalueSpectralHeatODE(kappa=kappa, eigvals=safe_eigvals)
        self.zero_eigenvalue_ode = TrivialODE()

    def b(self, t: Tensor, xt: Tensor) -> Tensor:
        return torch.where(
            self.zero_eigenvalue_mask,
            self.zero_eigenvalue_ode.b(t, xt),
            self.positive_eigenvalue_ode.b(t, xt)
        )

    def s(self, t: Tensor, v: Tensor) -> Tensor:
        return torch.where(
            self.zero_eigenvalue_mask,
            self.zero_eigenvalue_ode.s(t, v),
            self.positive_eigenvalue_ode.s(t, v)
        )

    def v(self, x0: Tensor, x1: Tensor) -> Tensor:
        return torch.where(
            self.zero_eigenvalue_mask,
            self.zero_eigenvalue_ode.v(x0, x1),
            self.positive_eigenvalue_ode.v(x0, x1)
        )

    def x(self, t: Tensor, x0: Tensor, x1: Tensor) -> Tensor:
        return torch.where(
            self.zero_eigenvalue_mask,
            self.zero_eigenvalue_ode.x(t, x0, x1),
            self.positive_eigenvalue_ode.x(t, x0, x1)
        )

    def c(self, x0: Tensor, x1: Tensor) -> Tensor:
        z0, z1 = self.positive_eigenvalue_ode._c_transform(x0, x1)
        z0 = torch.where(self.zero_eigenvalue_mask, x0, z0)
        z1 = torch.where(self.zero_eigenvalue_mask, x1, z1)
        return torch.cdist(z0, z1, p=2).square()


class HeatODE(SpectralBaseODE):
    def __init__(self, kappa: float, eigvals: Tensor, frame: Frame) -> None:
        """
        Args:
            kappa: float
            eigval: (E,)
            frame: The frame. Either Euclidean or Spectral.
        """
        base_ode = _SpectralHeatODE(kappa=kappa, eigvals=eigvals)
        super().__init__(base_ode=base_ode, frame=frame)
        