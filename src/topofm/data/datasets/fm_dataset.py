import torch
from abc import ABC, abstractmethod
from torch import Tensor
from topofm.distributions.boundary.boundary_distribution import BoundaryDistribution
from topofm.frames.frame import Frame


class FMDataset(ABC):

    def __init__(
        self,
        mu0: BoundaryDistribution,
        mu1: BoundaryDistribution,
        eigvals: Tensor,
        frame: Frame,
        device: torch.device,
        dtype: torch.dtype,
    ) -> None:
        super().__init__()
        self.device = device
        self.dtype = dtype
        self.mu0 = mu0
        self.mu1 = mu1
        self.eigvals = eigvals
        self.frame = frame

    @property
    @abstractmethod
    def dim(self) -> int:
        """
        Returns:
            int: The dimension of the dataset.
        """
        pass

    @property
    @abstractmethod
    def num_samples(self) -> int:
        """
        Returns:
            int: The number of samples in the dataset.
        """
        pass

    @classmethod
    @abstractmethod
    def from_disk(cls, device: torch.device, dtype: torch.dtype):
        """
        Load the dataset from disk.
        """

    @abstractmethod
    def split(self, split: tuple[float, float, float] = (0.7, 0.1, 0.2)) -> tuple["FMDataset", "FMDataset", "FMDataset"]:
        """
        Split the dataset into training, validation, and test sets.

        Args:
            split (tuple[float, float, float]): The ratio of the dataset to be used for training, validation, and test.

        Returns:
            tuple[FMDataset, FMDataset, FMDataset]: The training, validation, and test datasets.
        """
