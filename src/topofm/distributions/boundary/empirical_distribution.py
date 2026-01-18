import torch
from torch import Tensor, Size
from topofm.distributions.boundary.boundary_distribution import BoundaryDistribution
from topofm.spaces import Space


class EmpiricalDistribution(BoundaryDistribution):
    """
    Empirical distribution in ambient coordinates.
    """
    def __init__(self, samples: Tensor, space: Space) -> None:
        """
        Args:
            samples: (n, d) samples in ambient coordinates.
            space: The space to use.
        """
        super().__init__()
        self.samples = samples
        self.space = space
        self.device = samples.device
        self.dtype = samples.dtype

    def sample(self, shape: Size) -> Tensor:
        idx = torch.randint(0, self.samples.shape[0], shape, device=self.device)
        return self.samples[idx]

    @classmethod
    def from_standard(cls, samples: Tensor, space: Space) -> "EmpiricalDistribution":
        samples = space.frame.from_standard(samples)
        return cls(samples=samples, space=space)

    @property
    def p(self) -> Tensor:
        return torch.ones(self.samples.shape[0], device=self.device, dtype=self.dtype) / self.samples.shape[0]
