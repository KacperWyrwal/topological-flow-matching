import torch
import pandas as pd
from torch import Tensor
from topofm.config import DATA_DIR
from topofm.data.datasets.fm_dataset import FMDataset
from topofm.distributions.boundary.empirical_distribution import EmpiricalDistribution
from topofm.spaces import Space


BRAIN_DATA_DIR = DATA_DIR / "brain"


def load_brain_regions_centroids() -> pd.DataFrame:
    return pd.read_csv(BRAIN_DATA_DIR / "brain_regions_centroids.csv")


class BrainGraph(Space):
    # TODO Inherit from spaces.Graph, which should store a graph structure
    # optionally with nodes, edges, and node coordinates.
    def __init__(
        self, 
        coords: Coordinates | str, 
        dtype: torch.dtype, 
        device: torch.device,
    ) -> None:
        eigvals = self.load_eigvals().to(device=device, dtype=dtype)
        eigvecs = self.load_eigvecs().to(device=device, dtype=dtype)
        super().__init__(eigvals=eigvals, eigvecs=eigvecs, coords=coords)

    def load_eigvals(self) -> Tensor:
        return torch.load(BRAIN_DATA_DIR / 'eigenvalues.pt', map_location="cpu")

    def load_eigvecs(self) -> Tensor:
        return torch.load(BRAIN_DATA_DIR / 'eigenvectors.pt', map_location="cpu")


class BrainDataset(FMDataset):

    @staticmethod
    def load_data() -> tuple[Tensor, Tensor]:
        x0 = torch.load(BRAIN_DATA_DIR / 'x0_liberal.pt', map_location="cpu")
        x1 = torch.load(BRAIN_DATA_DIR / 'x1_aligned.pt', map_location="cpu")
        return x0, x1

    @classmethod
    def from_disk(
        cls,
        space: BrainGraph,
        ode: ODE,
        device: torch.device, 
        dtype: torch.dtype,
    ) -> "BrainDataset":
        x0, x1 = BrainDataset.load_data()
        x0 = x0.to(device=device, dtype=dtype)
        x1 = x1.to(device=device, dtype=dtype)
        mu0 = EmpiricalDistribution.from_standard(x0, space=space)
        mu1 = EmpiricalDistribution.from_standard(x1, space=space)
        return cls(
            mu0=mu0,
            mu1=mu1,
        )
    
    def _split(self, split: tuple[float, float, float] = (0.7, 0.1, 0.2)) -> tuple["BrainDataset", "BrainDataset", "BrainDataset"]:
        """
        Split the dataset into training, validation, and test sets.

        Args:
            split (tuple[float, float, float]): The ratio of the dataset to be used for training, validation, and test.

        Returns:
            tuple[BrainDataset, BrainDataset, BrainDataset]: The training, validation, and test datasets.
        """
        # Samples in ambient coordinates
        x0 = self.mu0.samples
        x1 = self.mu1.samples

        # Shuffle samples
        idx0 = torch.randperm(x0.shape[0])
        x0 = x0[idx0]

        idx1 = torch.randperm(x1.shape[0])
        x1 = x1[idx1]

        # Split into train, validation, and test
        total_size = x0.shape[0]
        train_size, val_size, test_size = split
        x0_train_size = int(train_size * total_size)
        x0_val_size = int(val_size * total_size)
        x0_train = x0[:x0_train_size]
        x0_val = x0[x0_train_size:x0_train_size + x0_val_size]
        x0_test = x0[x0_train_size + x0_val_size:]

        x1_train_size = int(train_size * total_size)
        x1_val_size = int(val_size * total_size)
        x1_train = x1[:x1_train_size]
        x1_val = x1[x1_train_size:x1_train_size + x1_val_size]
        x1_test = x1[x1_train_size + x1_val_size:]

        # Create split distributions
        mu0_train = EmpiricalDistribution(x0_train, space=self.mu0.space)
        mu0_val = EmpiricalDistribution(x0_val, space=self.mu0.space)
        mu0_test = EmpiricalDistribution(x0_test, space=self.mu0.space)

        mu1_train = EmpiricalDistribution(x1_train, space=self.mu1.space)
        mu1_val = EmpiricalDistribution(x1_val, space=self.mu1.space)
        mu1_test = EmpiricalDistribution(x1_test, space=self.mu1.space)
        
        # Create split datasets
        return BrainDataset(
            mu0=mu0_train,
            mu1=mu1_train,
        ), BrainDataset(
            mu0=mu0_val,
            mu1=mu1_val,
        ), BrainDataset(
            mu0=mu0_test,
            mu1=mu1_test,
        )
        