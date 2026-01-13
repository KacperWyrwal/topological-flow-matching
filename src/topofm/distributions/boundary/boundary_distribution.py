from abc import ABC, abstractmethod
from torch import Size, Tensor
from ..frames import Frame


class BoundaryDistribution(ABC):
    """
    A distribution serving as either initial or final distribution in flow matching.
    """
    def __init__(self, device: torch.device, dtype: torch.dtype) -> None:
        super().__init__()
        self.device = device
        self.dtype = dtype

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
