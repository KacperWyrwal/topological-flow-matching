import torch


class TimeSteps:
    def __init__(self, t: torch.Tensor, dt: torch.Tensor) -> None:
        assert t.dtype == dt.dtype
        assert t.device == dt.device
        self._t = t
        self._dt = dt

    @property
    def t(self) -> torch.Tensor:
        return self._t

    @property
    def dt(self) -> torch.Tensor:
        return self._dt

    @property
    def device(self) -> torch.device:
        return self.t.device

    @property
    def dtype(self) -> torch.dtype:
        return self.t.dtype


class UniformTimeSteps(TimeSteps):
    def __init__(self, n: int, *, t0: float = 1e-4, device: torch.device | None = None, dtype: torch.dtype | None = None):
        assert n > 0
        t = torch.linspace(0, 1, n + 1, dtype=dtype, device=device)
        t[0] = t0
        dt = torch.full((n,), 1.0 / n, dtype=dtype, device=device)
        super().__init__(t=t, dt=dt)


