import torch
from torch import Tensor
from torchdiffeq import odeint
from ..odes import ODE, SpectralBaseODE


class _ODESolver:
    def __init__(self, ode: ODE, n_steps: int) -> None:
        self.ode = ode
        self.n_steps = n_steps

    def dx(self, t: float, xt: Tensor) -> Tensor:
        """
        Args:
            t: float
            xt: (..., d)
        Returns:
            dx: (..., d)
        """
        return self.ode.b(t=t, xt=xt)

    def x1(self, x0: Tensor) -> Tensor:
        """
        Args: 
            x0: (..., d)
        Returns:
            x: (..., d)
        """
        t = torch.tensor([0.0, 1.0])
        return odeint(self.dx, x0, t)[-1]

    def x(self, x0: Tensor) -> Tensor:
        """
        Args: 
            x0: (..., d)
        Returns:
            x: (n_steps + 1, ..., d)
        """
        t = torch.linspace(0.0, 1.0, self.n_steps + 1)
        return odeint(self.dx, x0, t)


class _SpectralBaseODESolver:
    def __init__(self, ode: SpectralBaseODE, n_steps: int) -> None:
        self._ode_solver = _ODESolver(ode=ode.base_ode, n_steps=n_steps)
        self.frame = ode.frame

    def x1(self, x0: Tensor) -> Tensor:
        """
        Args: 
            x0: (..., d)
        Returns:
            x: (..., d)
        """
        y0 = self.frame.ambient_to_spectral(x0)
        y1 = self._ode_solver.x1(y0)
        return self.frame.spectral_to_ambient(y1)

    def x(self, x0: Tensor) -> Tensor:
        """
        Args: 
            x0: (..., d)
        Returns:
            x: (n_steps + 1, ..., d)
        """
        y0 = self.frame.ambient_to_spectral(x0)
        y = self._ode_solver.x(y0)
        return self.frame.spectral_to_ambient(y)


class ODESolver:
    def __init__(self, ode: ODE, n_steps: int) -> None:
        if isinstance(ode, SpectralBaseODE):
            self._ode_solver = _SpectralBaseODESolver(ode=ode, n_steps=n_steps)
        else:
            self._ode_solver = _ODESolver(ode=ode, n_steps=n_steps)

    def x1(self, x0: Tensor) -> Tensor:
        return self._ode_solver.x1(x0)
    
    def x(self, x0: Tensor) -> Tensor:
        return self._ode_solver.x(x0)
