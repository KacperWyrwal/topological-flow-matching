import torch
from torch import Tensor, Size
from topofm.distributions.boundary.boundary_distribution import BoundaryDistribution


class Normal(BoundaryDistribution):
    def __init__(self, mean: Tensor, stddev: Tensor, device: torch.device, dtype: torch.dtype) -> None:
        super().__init__(device=device, dtype=dtype)
        self._dist = torch.distributions.Normal(mean, stddev)

    def sample(self, shape: Size) -> Tensor:
        return self._dist.sample(shape)
