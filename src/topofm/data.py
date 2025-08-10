from abc import ABC, abstractmethod
from ast import Index
import os

import pandas as pd
import scipy
from torch._C import K
from topofm.distributions import Empirical, DistributionInFrame
from torch.distributions import Distribution
from torch.utils.data import DataLoader
import torch

from .coupling import Coupling
from .time import TimeSteps


class TimeSampler(ABC):
    def __init__(self, device: torch.device | None = None, dtype: torch.dtype | None = None):
        self.device = device
        self.dtype = dtype

    @abstractmethod
    def sample(self, shape: torch.Size) -> torch.Tensor: ...


class UniformTimeSampler(TimeSampler):
    def sample(self, shape: torch.Size) -> torch.Tensor:
        return torch.rand(shape, device=self.device, dtype=self.dtype)


class DiscreteTimeSampler(TimeSampler):
    def __init__(self, time_steps: TimeSteps):
        device = time_steps.device
        dtype = time_steps.dtype
        super().__init__(device, dtype)
        self.time_steps = time_steps

    def sample(self, shape: torch.Size) -> torch.Tensor:
        indices = torch.randint(0, self.time_steps.t.shape[-1] - 1, shape, device=self.device)
        return self.time_steps.t[indices]


def _train_test_split_index(n: int, test_size: float) -> tuple[torch.Tensor, torch.Tensor]:
    num_test = int(n * test_size)
    idx = torch.randperm(n)
    return idx[:-num_test], idx[-num_test:]


def _chunk_index(n: int, k: int) -> list[torch.Tensor]:
    idx = torch.randperm(n)
    return idx.chunk(k)


class MatchingDataset(torch.utils.data.Dataset):
    def __init__(self, mu0: Distribution, mu1: Distribution) -> None:
        super().__init__()
        self.mu0 = mu0 
        self.mu1 = mu1 

    def sample(self, shape: torch.Size) -> tuple[torch.Tensor, torch.Tensor]:
        assert len(shape) <= 1, "Sample shape should be (S,) or ()."
        return self.mu0.sample(shape), self.mu1.sample(shape)

    @property 
    def target_is_empirical(self) -> bool:
        return isinstance(self.mu1.base, Empirical)
    
    @property 
    def source_is_empirical(self) -> bool:
        return isinstance(self.mu0.base, Empirical)

    @abstractmethod
    def train_test_split(self, test_size: float = 0.2): ... 

    @abstractmethod
    def chunk(self, k: int): ...


class AnalyticToAnalyticDataset(MatchingDataset):
    def train_test_split(self, test_size: float = 0.2):
        return self, self

    def chunk(self, k: int):
        return [self] * k


class AnalyticToEmpiricalDataset(MatchingDataset):
    def train_test_split(self, test_size: float = 0.2):
        n = self.mu1.num_samples
        train_idx, test_idx = _train_test_split_index(n, test_size)
        mu1_train = self.mu1[train_idx]
        mu1_test = self.mu1[test_idx]
        return AnalyticToEmpiricalDataset(self.mu0, mu1_train), AnalyticToEmpiricalDataset(self.mu0, mu1_test)

    def chunk(self, k: int):
        n = self.mu1.num_samples
        return [AnalyticToEmpiricalDataset(self.mu0, self.mu1[idx]) for idx in _chunk_index(n=n, k=k)]


class EmpiricalToAnalyticDataset(MatchingDataset):
    def train_test_split(self, test_size: float = 0.2):
        n = self.mu0.num_samples
        train_idx, test_idx = _train_test_split_index(n, test_size)
        mu0_train = self.mu0[train_idx]
        mu0_test = self.mu0[test_idx]
        return EmpiricalToAnalyticDataset(mu0_train, self.mu1), EmpiricalToAnalyticDataset(mu0_test, self.mu1)

    def chunk(self, k: int):
        n = self.mu0.num_samples
        return [EmpiricalToAnalyticDataset(self.mu0[idx], self.mu1) for idx in _chunk_index(n=n, k=k)]


class EmpiricalToEmpiricalDataset(MatchingDataset):
    def train_test_split(self, test_size: float = 0.2):
        n0 = self.mu0.num_samples
        n1 = self.mu1.num_samples
        assert n0 == n1, "Source and target must have same number of samples"
        n = n0
        train_idx, test_idx = _train_test_split_index(n, test_size)
        mu0_train = self.mu0[train_idx]
        mu1_train = self.mu1[train_idx]
        mu0_test = self.mu0[test_idx]
        mu1_test = self.mu1[test_idx]
        return EmpiricalToEmpiricalDataset(mu0_train, mu1_train), EmpiricalToEmpiricalDataset(mu0_test, mu1_test)

    def chunk(self, k: int):
        n0 = self.mu0.num_samples
        n1 = self.mu1.num_samples
        assert n0 == n1, "Source and target must have same number of samples"
        n = n0
        return [EmpiricalToEmpiricalDataset(self.mu0[idx], self.mu1[idx]) for idx in _chunk_index(n=n, k=k)]


class MatchingTrainDataLoader:
    def __init__(self, coupling: Coupling, batch_size: int, num_batches: int) -> None:
        self.coupling = coupling
        self.batch_size = batch_size
        self.num_batches = num_batches

    def __iter__(self):
        for _ in range(self.num_batches):
            yield self.coupling.sample((self.batch_size,))

    def __len__(self) -> int:
        return self.num_batches


class MatchingTestDataLoader:
    def __init__(self, coupling: IndependentCoupling, batch_size: int, num_batches: int | None = None):
        pass 

    def __iter__(self):
        # TODO 
        # 1. Non-empirical to non-empirical


        # 2. Non-empirical to empirical 

        # 3. Empirical to empirical 
        pass 


# Brain signals utils
BRAIN_DIR = '../datasets/brain/'


def download_brain_regions_centroids(data_dir: str = BRAIN_DIR):
    csv_url = "https://bitbucket.org/dpat/tools/raw/master/REF/ATLASES/HCP-MMP1_UniqueRegionList.csv"
    brain_regions_centroids_df = pd.read_csv(
        csv_url,
        usecols=["x-cog", "y-cog", "z-cog"],
        dtype={"x-cog": float, "y-cog": float, "z-cog": float},
    ).rename(columns={"x-cog": "x", "y-cog": "y", "z-cog": "z"})
    os.makedirs(data_dir, exist_ok=True)
    brain_regions_centroids_df.to_csv(os.path.join(data_dir, "brain_regions_centroids.csv"), index=False)


def load_brain_regions_centroids(data_dir: str = BRAIN_DIR):
    return pd.read_csv(os.path.join(data_dir, "brain_regions_centroids.csv"))


def load_brain_laplacian(data_dir: str = BRAIN_DIR) -> torch.Tensor:
    laplacian = scipy.io.loadmat(os.path.join(data_dir, "lap.mat"))['L']
    return torch.as_tensor(laplacian)


def load_brain_data(data_dir: str = BRAIN_DIR) -> tuple[torch.Tensor, torch.Tensor]:
    x1 = scipy.io.loadmat(os.path.join(data_dir, "aligned.mat"))['Xa'].T
    x0 = scipy.io.loadmat(os.path.join(data_dir, "liberal.mat"))['Xl'].T
    return torch.as_tensor(x0), torch.as_tensor(x1)
