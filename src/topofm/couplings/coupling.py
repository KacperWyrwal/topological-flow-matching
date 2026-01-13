from abc import ABC
from torch import Size, Tensor

from ..distributions.boundary import BoundaryDistribution


class Coupling(ABC):
    def __init__(self, mu0: BoundaryDistribution, mu1: BoundaryDistribution) -> None:
        super().__init__()
        self.mu0 = mu0
        self.mu1 = mu1

    @abstractmethod
    def sample(self, shape: Size) -> tuple[Tensor, Tensor]:
        """
        Samples from the coupling.

        Args:
            shape: The shape of the sample.
        Returns:
            sample: tuple of (..., d), (..., d)
        """
        pass
