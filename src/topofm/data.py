from abc import ABC, abstractmethod
import os
from pathlib import Path

import pickle
import requests
import pandas as pd
import scipy
import torch

from .distributions import Distribution, EmpiricalInFrame, Empirical, AnalyticInFrame
from .coupling import Coupling
from .time import TimeSteps

# Try to use Hydra to resolve absolute paths if available
try:
    from hydra.utils import to_absolute_path as _to_absolute_path
except Exception:  # pragma: no cover - hydra not always present at import time
    _to_absolute_path = None


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
        assert mu0.event_shape == mu1.event_shape, f"Need {mu0.event_shape} == {mu1.event_shape}"
        self.mu0 = mu0 
        self.mu1 = mu1

    # TODO maybe remove this method 
    def sample(self, shape: torch.Size) -> tuple[torch.Tensor, torch.Tensor]:
        assert len(shape) <= 1, "Sample shape should be (S,) or ()."
        return self.mu0.sample(shape), self.mu1.sample(shape)

    @abstractmethod
    def train_test_split(self, test_size: float = 0.2): ... 

    @abstractmethod
    def chunk(self, k: int): ...

    @property
    def dim(self) -> int: # TODO This implementation could be pushed up
        return self.mu0.event_shape[0]


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
    def train_test_split(
        self, 
        test_size: float = 0.2,
    ) -> tuple["EmpiricalToEmpiricalDataset", "EmpiricalToEmpiricalDataset"]:
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


class MatchingTrainLoader:
    def __init__(self, coupling: Coupling, batch_size: int, epoch_size: int) -> None:
        self.coupling = coupling
        self.batch_size = batch_size
        self.epoch_size = epoch_size
        self.num_batches, res = divmod(epoch_size, self.batch_size)
        # TODO 

    def __iter__(self):
        for _ in range(self.num_batches):
            yield self.coupling.sample((self.batch_size,))

    def __len__(self) -> int:
        return self.num_batches


class MatchingTestLoader(ABC):
    @abstractmethod
    def __iter__(self): ...


class EmpiricalToEmpiricalTestLoader(MatchingTestLoader):
    def __init__(self, dataset: EmpiricalToEmpiricalDataset, batch_size: int | None = None):
        super().__init__()
        self.x0 = dataset.mu0.samples 
        self.x1 = dataset.mu1.samples 
        assert self.x0.ndim == 2
        assert self.x0.shape == self.x1.shape
        self.batch_size = batch_size

        if self.batch_size is None:
            self.num_batches = 1
        else:
            num_batches, last = divmod(self.x0.shape[0], batch_size)
            self.num_batches = num_batches + int(last > 0)

    def __iter__(self):
        if self.batch_size is None:
            yield self.x0, self.x1
        else:
            yield from zip(
                torch.chunk(self.x0, self.batch_size), 
                torch.chunk(self.x1, self.batch_size),
            )

    def __len__(self) -> int:
        return self.num_batches


class AnalyticToEmpiricalTestLoader(MatchingTestLoader):
    def __init__(self, dataset: AnalyticToEmpiricalDataset, batch_size: int | None = None):
        super().__init__()
        self.mu0 = dataset.mu0
        self.mu1: EmpiricalInFrame | Empirical = dataset.mu1
        self.batch_size = batch_size

        if self.batch_size is None:
            self.num_batches = 1
        else:
            num_batches, last = divmod(self.x0.shape[0], batch_size)
            self.num_batches = num_batches + int(last > 0)

    def __iter__(self):
        mu0_samples = self.mu0.sample((self.mu1.num_samples,))
        if self.batch_size is None:
            yield mu0_samples, self.mu1.samples
        else:
            yield from zip(
                torch.chunk(mu0_samples, self.batch_size),
                torch.chunk(self.mu1.samples, self.batch_size),
            )

    def __len__(self) -> int:
        return self.num_batches


class AnalyticToAnalyticTestLoader(MatchingTestLoader):
    def __init__(self, dataset: AnalyticToAnalyticDataset, batch_size: int, epoch_size: int):
        super().__init__()
        self.mu0 = dataset.mu0
        self.mu1 = dataset.mu1 
        self.batch_size = batch_size
        self.epoch_size = epoch_size 

        self.num_batches, res = divmod(self.epoch_size, self.batch_size)
    
    def __iter__(self):
        for _ in range(len(self)):
            yield self.mu0.sample((self.batch_size, )), self.mu1.sample((self.batch_size, ))

    def __len__(self) -> int:
        return self.num_batches


# Brain signals utils
def _resolve_brain_dir(data_dir: str | None) -> str:
    """Resolve brain dataset directory.

    Priority:
    1) explicit provided `data_dir`
    2) Hydra-aware path: to_absolute_path("datasets/brain")
    3) Fallback relative to repo root (two levels above this file): ../../datasets/brain
    """
    if data_dir is not None:
        return data_dir
    if _to_absolute_path is not None:
        return _to_absolute_path("datasets/brain")
    return str(Path(__file__).resolve().parents[2] / "datasets" / "brain")


def download_brain_regions_centroids(data_dir: str | None = None):
    data_dir = _resolve_brain_dir(data_dir)
    csv_url = "https://bitbucket.org/dpat/tools/raw/master/REF/ATLASES/HCP-MMP1_UniqueRegionList.csv"
    brain_regions_centroids_df = pd.read_csv(
        csv_url,
        usecols=["x-cog", "y-cog", "z-cog"],
        dtype={"x-cog": float, "y-cog": float, "z-cog": float},
    ).rename(columns={"x-cog": "x", "y-cog": "y", "z-cog": "z"})
    os.makedirs(data_dir, exist_ok=True)
    brain_regions_centroids_df.to_csv(os.path.join(data_dir, "brain_regions_centroids.csv"), index=False)


def load_brain_regions_centroids(data_dir: str | None = None):
    data_dir = _resolve_brain_dir(data_dir)
    return pd.read_csv(os.path.join(data_dir, "brain_regions_centroids.csv"))


def load_brain_laplacian(data_dir: str | None = None) -> torch.Tensor:
    data_dir = _resolve_brain_dir(data_dir)
    laplacian = scipy.io.loadmat(os.path.join(data_dir, "lap.mat"))['L']
    return torch.as_tensor(laplacian)


def load_brain_data(data_dir: str | None = None) -> tuple[torch.Tensor, torch.Tensor]:
    data_dir = _resolve_brain_dir(data_dir)
    x1 = scipy.io.loadmat(os.path.join(data_dir, "aligned.mat"))['Xa'].T
    x0 = scipy.io.loadmat(os.path.join(data_dir, "liberal.mat"))['Xl'].T
    return torch.as_tensor(x0), torch.as_tensor(x1)


"""
Dataset Summary
---------------

This pickle file contains five sequentially pickled objects representing 
a processed earthquake event dataset mapped onto a spherical mesh:

1. `G` (networkx.Graph)
   - 576 nodes: unique mesh vertices that had at least one M5.5+ event between 1990–2018.
   - 1,524 undirected edges constructed via a 10-nearest neighbour search (mutual k-NN), 
     based on geodesic distances between vertices.
   - Average degree ≈ 5.29, degree range 0–11, with 14 isolated vertices.
   - All edges are unweighted (`weight = 1.0`).

2. `L` (numpy.ndarray, shape=(576, 576))
   - Symmetric normalized graph Laplacian computed from `G`.
   - Verified to match `networkx.normalized_laplacian_matrix(G)`.

3. `evs` (numpy.ndarray, shape=(576,))
   - Eigenvalues of the Laplacian `L`.
   - All values lie within [0, 2], as expected for the normalized Laplacian.

4. `V` (numpy.ndarray, shape=(576, 576))
   - Eigenvectors of `L`.
   - Columns are orthonormal and satisfy `L @ V ≈ V @ diag(evs)`.

5. `GS` (numpy.ndarray, shape=(29, 576))
   - Yearly graph signals: earthquake magnitudes aggregated per vertex per year 
     (1990–2018 → 29 rows; vertices → columns).
   - Column means are approximately zero, consistent with preprocessing by 
     subtracting the mean over years.

Data Processing Pipeline (original experiment)
----------------------------------------------
- Start with IRIS catalogue: M5.5+ events (1990–2018), total 12,940 events.
- Map each event to the nearest vertex in an icosahedral triangulated mesh of the Earth 
  (level-3 refinement, 1,922 vertices).
- Keep only the 576 vertices with at least one event.
- Construct a mutual 10-nearest neighbour graph using geodesic distances.
- Compute the symmetric normalized Laplacian, its eigenvalues, and eigenvectors.
- Aggregate events yearly per vertex, take magnitudes as signals, 
  and remove the mean magnitude over years.
"""

def _resolve_earthquake_dir(data_dir: str | None) -> str:
    if data_dir is not None:
        return data_dir
    if _to_absolute_path is not None:
        return _to_absolute_path("datasets/earthquakes")
    return str(Path(__file__).resolve().parents[2] / "datasets" / "earthquakes")


# Earthquake data utils
def download_earthquake_data(data_dir: str | None = None):
    """
    Download earthquake data from https://github.com/cookbook-ms/topological_SB_matching/blob/7558367c1847b0b274ca006796ef28e3e809da01/TSBLearning/code/datasets/earthquakes/eqs.pkl. 
    """
    data_dir = _resolve_earthquake_dir(data_dir)
    os.makedirs(data_dir, exist_ok=True)
    url = (
        "https://raw.githubusercontent.com/cookbook-ms/topological_SB_matching/"
        "7558367c1847b0b274ca006796ef28e3e809da01/TSBLearning/code/datasets/earthquakes/eqs.pkl"
    )
    response = requests.get(url, timeout=60)
    response.raise_for_status()
    with open(os.path.join(data_dir, 'eqs.pkl'), 'wb') as f:
        f.write(response.content)
    print(f"Downloaded earthquake data to {data_dir}")


def load_earthquake_laplacian(data_dir: str | None = None) -> torch.Tensor:
    """Load the symmetric normalized graph Laplacian L (576x576).

    The pickle contains: G, L, evs, V, GS. We read in order and return L.
    """
    data_dir = _resolve_earthquake_dir(data_dir)
    with open(os.path.join(data_dir, 'eqs.pkl'), 'rb') as f:
        _G = pickle.load(f)   # networkx.Graph (unused here)
        L = pickle.load(f)    # (576, 576) numpy array
        # Consume remaining objects to leave file pointer at end for safety
        _evs = pickle.load(f) # (576,) eigenvalues (unused)
        _V = pickle.load(f)   # (576, 576) eigenvectors (unused)
        _GS = pickle.load(f)  # (29, 576) yearly signals (unused)
    return torch.as_tensor(L)


def load_earthquake_data(data_dir: str | None = None) -> torch.Tensor:
    """Load yearly graph signals GS with shape (29, 576).

    Each row is a year's magnitudes per vertex; columns correspond to the 576
    vertices used to build the graph. No rows are dropped per the dataset
    description (1990–2018 inclusive → 29 rows).
    """
    data_dir = _resolve_earthquake_dir(data_dir)
    with open(os.path.join(data_dir, 'eqs.pkl'), 'rb') as f:
        _G = pickle.load(f)   # networkx.Graph (unused)
        _L = pickle.load(f)   # Laplacian (unused here)
        _evs = pickle.load(f) # eigenvalues (unused)
        _V = pickle.load(f)   # eigenvectors (unused)
        GS = pickle.load(f)[:-1]   # (29, 576)
    return torch.as_tensor(GS)
