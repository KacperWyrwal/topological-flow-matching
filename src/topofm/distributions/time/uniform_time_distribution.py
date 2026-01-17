import torch
from torch import Size, Tensor
from topofm.distributions.time.time_distribution import TimeDistribution


class UniformTimeDistribution(TimeDistribution):
    """Samples time from a uniform distribution over [0, 1)."""
    def sample(self, shape: Size) -> Tensor:
        return torch.rand(shape).unsqueeze(-1)
