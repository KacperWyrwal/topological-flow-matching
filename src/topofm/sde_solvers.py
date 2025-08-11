from abc import ABC, abstractmethod
import torch

from .control import Control, ZeroControl
from .sde import SDE
from .time import TimeSteps


class SDESolver(ABC):
    def __init__(self, sde: SDE) -> None:
        self.sde = sde

    @abstractmethod
    def pushforward(self, x0: torch.Tensor, *, control: Control | None = None) -> torch.Tensor: ...

    @abstractmethod
    def sample_path(self, x0: torch.Tensor, *, control: Control | None = None) -> tuple[torch.Tensor, torch.Tensor]: 
        """
        Sample a path from the SDE.

        Args:
            x0: Initial state.
            control: Control function.

        Returns:
            tuple[torch.Tensor, torch.Tensor]: xt, t - path and time steps.
        """


class EulerMaruyamaSolver(SDESolver):
    def __init__(self, sde: SDE, *, time_steps: TimeSteps):
        super().__init__(sde=sde)
        self.time_steps = time_steps

    def dx(self, dt: torch.Tensor, t: torch.Tensor, x: torch.Tensor, *, control: Control) -> torch.Tensor:
        t = t.expand(x.shape[:-1])
        drift = self.sde.drift(t=t, x=x) + control(t=t, x=x)
        diffusion = self.sde.diffusion(t=t, x=x)
        return drift * dt + diffusion * torch.sqrt(dt) * torch.randn_like(x)

    def pushforward(self, x0: torch.Tensor, *, control: Control | None = None) -> torch.Tensor:
        if control is None:
            control = ZeroControl()
        ts = self.time_steps.t
        dts = self.time_steps.dt
        x = x0
        for t, dt in zip(ts[:-1], dts):
            x = x + self.dx(dt=dt, t=t, x=x, control=control)
        return x

    def sample_path(self, x0: torch.Tensor, *, control: Control | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        if control is None:
            control = ZeroControl()
        ts = self.time_steps.t
        dts = self.time_steps.dt
        x = x0
        xs = [x]
        for t, dt in zip(ts[:-1], dts):
            x = x + self.dx(dt=dt, t=t, x=x, control=control)
            xs.append(x)
        xs = torch.stack(xs, dim=-2)
        return xs, ts


