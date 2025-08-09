from abc import ABC, abstractmethod
import os
from typing import Tuple

import numpy as np
import pandas as pd
import scipy
import torch

from .frames import Frame, StandardFrame
from .ot import OTSolver
from .utils import joint_multinomial
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


class MatchingDataset(torch.utils.data.Dataset):
    def __init__(self, *, frame: Frame | None = None) -> None:
        super().__init__()
        self.frame = frame if frame is not None else StandardFrame()

    def sample(self, shape: torch.Size) -> Tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError


class MatchingTensorDataset(MatchingDataset):
    def __init__(self, x0: torch.Tensor, x1: torch.Tensor, *, frame: Frame | None = None) -> None:
        super().__init__(frame=frame)
        self.x0, self.x1 = frame.transform(x0, x1)

    def sample(self, shape: torch.Size) -> Tuple[torch.Tensor, torch.Tensor]:
        assert len(shape) <= 1
        idx_x0 = torch.randint(0, self.x0.shape[-2] - 1, shape)
        idx_x1 = torch.randint(0, self.x1.shape[-2] - 1, shape)
        return self.x0[idx_x0], self.x1[idx_x1]


class MatchingDistributionDataset(MatchingDataset):
    def __init__(
        self,
        x0_distribution: torch.distributions.Distribution,
        x1_distribution: torch.distributions.Distribution,
        *,
        frame: Frame | None = None,
    ) -> None:
        super().__init__(frame=frame)
        self.x0_distribution = x0_distribution
        self.x1_distribution = x1_distribution

    def sample(self, shape: torch.Size) -> Tuple[torch.Tensor, torch.Tensor]:
        x0 = self.x0_distribution.sample(shape)
        x1 = self.x1_distribution.sample(shape)
        x0, x1 = self.frame.transform(x0, x1)
        return x0, x1


class OTBatchSampler(torch.utils.data.Sampler):
    def __init__(self, dataset: MatchingDataset, ot_solver: OTSolver, batch_size: int, num_batches: int, *, batchwise_ot: bool = False):
        super().__init__()
        self.dataset = dataset
        self.batchwise_ot = batchwise_ot
        self.ot_solver = ot_solver
        self.batch_size = batch_size
        self.num_batches = num_batches
        self.ot_plan = None
        if isinstance(dataset, MatchingDistributionDataset):
            self.batchwise_ot = True

    def _get_ot_plan(self, x0: torch.Tensor, x1: torch.Tensor) -> torch.Tensor:
        if self.batchwise_ot is True or self.ot_plan is None:
            ot_plan = self.ot_solver.solve(x0, x1)
        else:
            ot_plan = self.ot_plan
        return ot_plan

    def _resample_from_ot_plan(self, x0: torch.Tensor, x1: torch.Tensor):
        x0_idx, x1_idx = joint_multinomial(self._get_ot_plan(x0, x1), num_samples=self.batch_size)
        return x0[x0_idx], x1[x1_idx]

    def _sample_batch(self):
        if self.batchwise_ot is False:
            x0, x1 = self.dataset.x0, self.dataset.x1
        else:
            x0, x1 = self.dataset.sample((self.batch_size,))
        return self._resample_from_ot_plan(x0, x1)

    def __iter__(self):
        for _ in range(len(self)):
            yield self._sample_batch()

    def __len__(self) -> int:
        return self.num_batches


class MatchingDataLoader:
    def __init__(self, dataset: MatchingDataset, batch_sampler: OTBatchSampler):
        self.dataset = dataset
        self.batch_sampler = batch_sampler

    def __iter__(self):
        yield from self.batch_sampler

    def __len__(self) -> int:
        return self.batch_sampler.batch_size * self.batch_sampler.num_batches

    @property
    def batch_size(self) -> int:
        return self.batch_sampler.batch_size

    @property
    def num_batches(self) -> int:
        return len(self) // self.batch_size


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
