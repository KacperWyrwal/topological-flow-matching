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
    ) -> None:
        super().__init__()
        self.mu0 = mu0
        self.mu1 = mu1
        self.eigvals = eigvals
        self.frame = frame

    @classmethod
    @abstractmethod
    def from_disk(cls):
        """
        Load the dataset from disk.
        """

    @abstractmethod
    def _split(self, split: tuple[float, float, float] = (0.7, 0.1, 0.2)) -> tuple["FMDataset", "FMDataset", "FMDataset"]:
        """
        Split the dataset into training, validation, and test sets.

        Args:
            split (tuple[float, float, float]): The ratio of the dataset to be used for training, validation, and test.

        Returns:
            tuple[FMDataset, FMDataset, FMDataset]: The training, validation, and test datasets.
        """

    def split(
        self,
        split: tuple[float, float, float] = (0.7, 0.1, 0.2),
        seed: int | None = None,
    ) -> tuple["FMDataset", "FMDataset", "FMDataset"]:
        """
        Split the dataset into training, validation, and test sets. Temporarily sets the provided random seed.

        Args:
            split (tuple[float, float, float]): The ratio of the dataset to be used for training, validation, and test.
            seed (int | None): The seed for the random number generator.

        Returns:
            tuple[FMDataset, FMDataset, FMDataset]: The training, validation, and test datasets.
        """
        # Save the current random seed
        current_seed = torch.get_rng_state()
        torch.manual_seed(seed)

        # Split the dataset
        train, val, test = self._split(split)

        torch.set_rng_state(current_seed)
        return train, val, test
