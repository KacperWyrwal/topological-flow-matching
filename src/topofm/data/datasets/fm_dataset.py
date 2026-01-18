import torch
from abc import ABC, abstractmethod
from topofm.distributions.boundary import BoundaryDistribution
from topofm.spaces import Space
from topofm.odes import ODE
from topofm.covariance import Covariance
from topofm.distributions.boundary.normal import MultivariateNormal

class FMDataset(ABC):

    def __init__(
        self,
        mu0: BoundaryDistribution,
        mu1: BoundaryDistribution,
    ) -> None:
        super().__init__()
        self.mu0 = mu0
        self.mu1 = mu1

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


class GenerationDataset(FMDataset):
    @staticmethod
    def _backward_transport_mu0(
        mu1: EmpiricalDistribution,
        ode: ODE,
        mode: Literal['full', 'identity'] = 'full',
    ) -> EmpiricalDistribution:
        """
        Transport the final distribution backwards in time.
        TODO Could make this a method of the ODE class, and 
        have the dataset simply pass it an appropriate normal distribution.
        Args:
            mu1 (EmpiricalDistribution): The final distribution.
            ode (ODE): The ODE to use for backward transport.
            mode (Literal['full', 'identity']): The mode to use for backward transport.
            If 'full', the full covariance matrix of mu1 is transported backwards, otherwise
            the identity matrix is transported. TODO could add a 'diagonal' mode.

        Returns:
            MultivariateNormal: The initial distribution.
        """
        mean = torch.mean(mu1.samples, dim=0)
        cov = Covariance.from_samples(mu1.samples, mode=mode)
        cov = ode.Phi10(cov)
        mu0 = MultivariateNormal(mean, cov)
        return mu0
        