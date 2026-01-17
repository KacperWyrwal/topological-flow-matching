from abc import ABC, abstractmethod
from torch import Size, Tensor


class TimeDistribution(ABC):
    @abstractmethod
    def sample(self, shape: Size) -> Tensor:
        """
        Sample from the time distribution.
        
        Args:
            shape: The shape of the sample.
        Returns:
            sample: (shape, 1)
        """
        pass
