import torch
from torch import Tensor, Size
from topofm.distributions.boundary.boundary_distribution import BoundaryDistribution
from topofm.frames.frame import Frame


class EmpiricalDistribution(BoundaryDistribution):
    """
    Empirical distribution in ambient coordinates.
    """
    def __init__(self, samples: Tensor, frame: Frame, device: torch.device, dtype: torch.dtype) -> None:
        """
        Args:
            samples: (n, d) samples in ambient coordinates.
            frame: The frame to use.
            device: The device to use.
            dtype: The dtype to use.
        """
        super().__init__(device=device, dtype=dtype)
        self.samples = samples
        self.frame = frame

    def sample(self, shape: Size) -> Tensor:
        idx = torch.randint(0, self.samples.shape[0], shape, device=self.device)
        return self.samples[idx]

    @classmethod
    def from_standard(cls, samples: Tensor, frame: Frame, device: torch.device, dtype: torch.dtype) -> "EmpiricalDistribution":
        samples = frame.standard_to_ambient(samples)
        return cls(samples=samples, frame=frame, device=device, dtype=dtype)

    @property
    def p(self) -> Tensor:
        return torch.ones(self.samples.shape[0], device=self.device, dtype=self.dtype) / self.samples.shape[0]
