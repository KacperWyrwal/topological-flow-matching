import torch
from torch import Tensor, Size
from .boundary_distribution import BoundaryDistribution


class EmpiricalDistribution(BoundaryDistribution):
    def __init__(self, x: Tensor, frame: Frame, device: torch.device, dtype: torch.dtype) -> None:
        """
        Args:
            x: (n, d) samples in standard coordinates.
            frame: The frame to use.
            device: The device to use.
            dtype: The dtype to use.
        """
        super().__init__(device=device, dtype=dtype)
        self._x = frame.from_standard_to_ambient(x)

    def sample(self, shape: Size) -> Tensor:
        idx = torch.randint(0, self._x.shape[0], shape, device=self.device)
        return self._x[idx]

    @property
    def x(self) -> Tensor:
        return self._x
