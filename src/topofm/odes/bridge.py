from torch import Tensor
from .ode import ODE, SpectralBaseODE

class _Bridge(ODE):
    def __init__(self, ode: ODE, x0: Tensor, x1: Tensor) -> None:
        super().__init__()
        self.ode = ode
        self.x0 = x0
        self.x1 = x1
    
    def b(self, t: Tensor, xt: Tensor) -> Tensor:
        return self.ode.b(t=t, xt=xt) + self.ode.sv(t=t, x0=self.x0, x1=self.x1)


class _SpectralBaseBridge(SpectralBaseODE):
    def __init__(self, ode: SpectralBaseODE, x0: Tensor, x1: Tensor) -> None:
        y0 = ode.space.frame.to_spectral(x0)
        y1 = ode.space.frame.to_spectral(x1)
        base_bridge = _Bridge(ode=ode.base_ode, x0=y0, x1=y1)
        super().__init__(base_ode=base_bridge, space=ode.space)


class Bridge(ODE):
    def __init__(self, ode: ODE, x0: Tensor, x1: Tensor) -> None:
        super().__init__()
        if isinstance(ode, SpectralBaseODE):
            self._bridge = _SpectralBaseBridge(ode=ode, x0=x0, x1=x1)
        else:
            self._bridge = _Bridge(ode=ode, x0=x0, x1=x1)
    
    def b(self, t: Tensor, xt: Tensor) -> Tensor:
        return self._bridge.b(t=t, xt=xt)
