from abc import ABC, abstractmethod
import torch

from .sde import SDE, OUDiagonalSDE


def _bridge_control_generic(
    sde: SDE, t: torch.Tensor, x: torch.Tensor, *, t1: torch.Tensor | None = None, x1: torch.Tensor
) -> torch.Tensor:
    if t1 is None:
        t1 = x1.new_ones(x1.shape[:-1])
    score = sde.transition(t0=t, t1=t1) * (x1 - sde.mean(t0=t, x0=x, t=t1)) / sde.variance(t0=t, t=t1)
    return sde.diffusion(t=t).square() * score


def _bridge_control_ou(
    sde: OUDiagonalSDE, t: torch.Tensor, x: torch.Tensor, *, t1: torch.Tensor | None = None, x1: torch.Tensor
) -> torch.Tensor:
    return sde.transition(t0=t, t1=t1) * (x1 - sde.mean(t0=t, x0=x, t=t1)) / sde._variance_with_initial_condition_over_gamma_squared(
        t0=t, t=t1
    )


def bridge_control(
    sde: SDE, t: torch.Tensor, x: torch.Tensor, *, t1: torch.Tensor | None = None, x1: torch.Tensor
) -> torch.Tensor:
    if t1 is None:
        t1 = x1.new_ones(x1.shape[:-1])
    if isinstance(sde, OUDiagonalSDE):
        return _bridge_control_ou(sde=sde, t=t, x=x, t1=t1, x1=x1)
    else:
        return _bridge_control_generic(sde=sde, t=t, x=x, t1=t1, x1=x1)


class Control(ABC):
    @abstractmethod
    def control(self, t: torch.Tensor, x: torch.Tensor, **kwargs) -> torch.Tensor: ...

    def __call__(self, t: torch.Tensor, x: torch.Tensor, **kwargs) -> torch.Tensor:
        return self.control(t=t, x=x, **kwargs)


class ZeroControl(Control):
    def control(self, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        return x.new_zeros(x.shape)


class BridgeControl(Control):
    def __init__(self, sde: SDE, x1: torch.Tensor, t1: torch.Tensor | None = None):
        super().__init__()
        self.sde = sde
        self._x1 = None
        self._t1 = None
        self.condition(x1=x1, t1=t1)

    def condition(self, x1: torch.Tensor, t1: torch.Tensor | None = None) -> "BridgeControl":
        if t1 is None:
            t1 = x1.new_ones(x1.shape[:-1])
        self._x1 = x1
        self._t1 = t1
        return self

    def control(self, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        return bridge_control(sde=self.sde, t=t, x=x, t1=self._t1, x1=self._x1)


class ModelControl(Control):
    def __init__(self, model: torch.nn.Module):
        super().__init__()
        self.model = model

    def control(self, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        return self.model(t, x)


