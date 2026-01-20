import torch
from abc import ABC, abstractmethod
from typing import Literal
from topofm.distributions.boundary import BoundaryDistribution
from topofm.odes import ODE
from topofm.distributions.covariance import Covariance
from topofm.distributions.boundary import (
    EmpiricalDistribution,
    MultivariateNormal
)


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
    def _split(self, ratio: tuple[float, float, float] = (0.7, 0.1, 0.2)) -> tuple["FMDataset", "FMDataset", "FMDataset"]:
        """
        Split the dataset into training, validation, and test sets.

        Args:
            ratio (tuple[float, float, float]): The ratio of the dataset to be used for training, validation, and test.

        Returns:
            tuple[FMDataset, FMDataset, FMDataset]: The training, validation, and test datasets.
        """

    def split(
        self,
        ratio: tuple[float, float, float] = (0.7, 0.1, 0.2),
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
        train, val, test = self._split(ratio)

        torch.set_rng_state(current_seed)
        return train, val, test


class GenerationDataset(FMDataset):
    @staticmethod
    def _mu0_from_mu1(
        mu1: EmpiricalDistribution,
        covariance_mode: Literal['full', 'diagonal','identity'] = 'full',
        transport: bool = False,
        ode: ODE | None = None,
    ) -> MultivariateNormal:
        """
        Transport the final distribution backwards in time.
        TODO Move to ODE class, and have the dataset pass an appropriate normal distribution.
        Args:
            mu1 (EmpiricalDistribution): The final distribution.
            covariance_mode (Literal['full', 'identity']): The mode to use for backward transport.
            If 'full', the full covariance matrix of mu1 is transported backwards, otherwise
            the identity matrix is transported.
            transport (bool): Whether to transport the final distribution backwards in time.
            ode (ODE | None): The ODE to use for backward transport.

        Returns:
            MultivariateNormal: The initial distribution.
        """
        mean = torch.mean(mu1.samples, dim=0)
        cov = Covariance.from_samples(mu1.samples, mode=covariance_mode)
        if transport:
            if ode is None:
                raise ValueError("ode must be provided if transport is True")
            cov = ode.Phi10(cov)
        mu0 = MultivariateNormal(mean, cov, space=mu1.space)
        return mu0

    def split(
        self,
        split: tuple[float, float, float] = (0.7, 0.1, 0.2),
        seed: int | None = None,
        mu0_covariance_mode: Literal['full', 'diagonal','identity'] = 'full',
        mu0_backward_transport: bool = False,
        ode: ODE | None = None,
    ) -> tuple["FMDataset", "FMDataset", "FMDataset"]:
        """
        Split the dataset into training, validation, and test sets. Temporarily sets the provided random seed.

        Args:
            split (tuple[float, float, float]): The ratio of the dataset to be used for training, validation, and test.
            seed (int | None): The seed for the random number generator.
            mu0_covariance_mode (Literal['full', 'diagonal','identity']): The mode to use for backward transport.
            mu0_backward_transport (bool): Whether to transport the final distribution backwards in time.
            ode (ODE | None): The ODE to use for backward transport.
        Returns:
            tuple[FMDataset, FMDataset, FMDataset]: The training, validation, and test datasets.
        """
        # Save the current random seed
        current_seed = torch.get_rng_state()
        torch.manual_seed(seed)

        # Split the dataset
        train, val, test = self._split(split, mu0_covariance_mode=mu0_covariance_mode, mu0_backward_transport=mu0_backward_transport, ode=ode)

        torch.set_rng_state(current_seed)
        return train, val, test
        
        