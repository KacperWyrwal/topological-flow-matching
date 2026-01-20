from torch import Tensor
from torchdiffeq import odeint
from topofm.odes import ODE, SpectralBaseODE


class _ODESolver:
    def __init__(self, ode: ODE) -> None:
        self.ode = ode

    def _odeint_func(self, t: Tensor, xt: Tensor) -> Tensor:
        """
        Args:
            t: ()
            xt: (..., d)
        Returns:
            dx: (..., d)
        """
        t = t.unsqueeze(0).unsqueeze(0)
        return self.dx(t, xt)

    def dx(self, t: Tensor, xt: Tensor) -> Tensor:
        """
        Args:
            t: (...,)
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
        t = x0.new_tensor([0.0, 1.0])
        return self.x(t=t, x0=x0)[-1]

    def x(self, t: Tensor, x0: Tensor) -> Tensor:
        """
        Args: 
            t: (...,)
            x0: (..., d)
        Returns:
            x: (..., d)
        """
        return odeint(self._odeint_func, x0, t, options={'dtype': x0.dtype})


class _SpectralBaseODESolver:
    def __init__(self, ode: SpectralBaseODE) -> None:
        self._ode_solver = _ODESolver(ode=ode.base_ode)
        self.space = ode.space

    def x1(self, x0: Tensor) -> Tensor:
        """
        Args: 
            x0: (..., d)
        Returns:
            x: (..., d)
        """
        y0 = self.space.frame.to_spectral(x0)
        y1 = self._ode_solver.x1(x0=y0)
        return self.space.frame.from_spectral(y1)

    def x(self, t: Tensor, x0: Tensor) -> Tensor:
        """
        Args: 
            t: (...,)
            x0: (..., d)
        Returns:
            x: (..., d)
        """
        y0 = self.space.frame.to_spectral(x0)
        y = self._ode_solver.x(t=t, x0=y0)
        return self.space.frame.from_spectral(y)


class ODESolver:
    def __init__(self, ode: ODE) -> None:
        self.ode = ode
        if isinstance(ode, SpectralBaseODE):
            self._ode_solver = _SpectralBaseODESolver(ode=ode)
        else:
            self._ode_solver = _ODESolver(ode=ode)

    def x1(self, x0: Tensor) -> Tensor:
        return self._ode_solver.x1(x0=x0)
    
    def x(self, t: Tensor, x0: Tensor) -> Tensor:
        return self._ode_solver.x(t=t, x0=x0)
