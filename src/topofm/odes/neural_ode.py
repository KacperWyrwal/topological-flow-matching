from .ode import ODE
from ..models.model import Model, ModelMode


class NeuralODE(ODE):
    def __init__(self, base_ode: ODE, model: Model) -> None:
        self.base_ode = base_ode
        self.model = model

    def b(self, t: Tensor, xt: Tensor) -> Tensor:
        if self.model.mode == ModelMode.V:
            return self.base_ode.b(t=t, xt=xt) + self.base_ode.s(t=t, v=self.model(t=t, xt=xt))
        if self.model.mode == ModelMode.SV:
            return self.base_ode.b(t=t, xt=xt) + self.model(t=t, xt=xt)
        