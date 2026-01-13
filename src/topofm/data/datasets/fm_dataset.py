import torch
from abc import ABC, abstractmethod
from functools import cached_property
from ...distributions.boundary import BoundaryDistribution
from ...frames import Frame


class FMDataset(ABC):

    def __init__(self, device: torch.device, dtype: torch.dtype, ambient_is_spectral: bool) -> None:
        super().__init__()
        self.device = device
        self.dtype = dtype
        self.ambient_is_spectral = ambient_is_spectral

    @property
    @abstractmethod
    def eigvec(self) -> Tensor:
        pass

    @property
    @abstractmethod
    def eigval(self) -> Tensor:
        pass

    @cached_property
    def spectral_frame(self) -> Frame:
        return Frame(eigvec=self.eigvec)


class MatchingDataset(FMDataset):
    @property
    @abstractmethod
    def mu0(self) -> BoundaryDistribution:
        pass

    @property
    @abstractmethod
    def mu1(self) -> BoundaryDistribution:
        pass


class GenerationDataset(FMDataset):
    @property
    @abstractmethod
    def mu1(self) -> BoundaryDistribution:
        pass



