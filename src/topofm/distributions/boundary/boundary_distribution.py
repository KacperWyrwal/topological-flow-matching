import torch
from abc import ABC, abstractmethod
from torch import Size, Tensor


class BoundaryDistribution(ABC):
    """
    A distribution serving as either initial or final distribution in flow matching.
    """
    @abstractmethod
    def sample(self, shape: Size) -> Tensor:
        """
        Samples from the distribution.

        Args:
            shape: The shape of the sample.
        Returns:
            sample: (..., d)
        """
        pass
