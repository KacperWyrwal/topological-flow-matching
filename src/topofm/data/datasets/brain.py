import torch
import pandas as pd
from torch import Tensor
from topofm.config import DATA_DIR
from topofm.data.datasets.fm_dataset import FMDataset
from topofm.distributions.boundary.empirical_distribution import EmpiricalDistribution
from topofm.frames.frame import Frame, AmbientCoordinates


BRAIN_DATA_DIR = DATA_DIR / "brain"


def load_brain_regions_centroids(data_dir: str | None = None):
    return pd.read_csv(os.path.join(data_dir, "brain_regions_centroids.csv"))


class BrainDataset(FMDataset):

    @property
    def dim(self) -> int:
        return self.mu0.samples.shape[1]

    @property
    def num_samples(self) -> int:
        return self.mu0.samples.shape[0]

    @staticmethod
    def load_data(device: torch.device, dtype: torch.dtype) -> tuple[Tensor, Tensor]:
        x0 = torch.load(BRAIN_DATA_DIR / 'x0_liberal.pt').to(device, dtype)
        x1 = torch.load(BRAIN_DATA_DIR / 'x1_aligned.pt').to(device, dtype)
        return x0, x1

    @staticmethod
    def load_eigvals(device: torch.device, dtype: torch.dtype) -> Tensor:
        eigvals = torch.load(BRAIN_DATA_DIR / 'eigenvalues.pt').to(device, dtype)
        return eigvals

    @staticmethod
    def load_eigvecs(device: torch.device, dtype: torch.dtype) -> Tensor:
        eigvecs = torch.load(BRAIN_DATA_DIR / 'eigenvectors.pt').to(device, dtype)
        return eigvecs

    @classmethod
    def from_disk(cls, ambient: AmbientCoordinates | str, device: torch.device, dtype: torch.dtype):
        eigvals = BrainDataset.load_eigvals(device=device, dtype=dtype)
        eigvecs = BrainDataset.load_eigvecs(device=device, dtype=dtype)
        frame = Frame(eigvecs=eigvecs, ambient=ambient)
        x0, x1 = BrainDataset.load_data(device=device, dtype=dtype)
        mu0 = EmpiricalDistribution.from_standard(x0, frame=frame, device=device, dtype=dtype)
        mu1 = EmpiricalDistribution.from_standard(x1, frame=frame, device=device, dtype=dtype)
        return cls(
            mu0=mu0,
            mu1=mu1,
            eigvals=eigvals,
            frame=frame,
            device=device,
            dtype=dtype,
        )
    
    def split(self, split: tuple[float, float, float] = (0.7, 0.1, 0.2)) -> tuple["BrainDataset", "BrainDataset", "BrainDataset"]:
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
        mu0_train = EmpiricalDistribution(x0_train, frame=self.frame, device=self.device, dtype=self.dtype)
        mu0_test = EmpiricalDistribution(x0_test, frame=self.frame, device=self.device, dtype=self.dtype)

        mu0_val = EmpiricalDistribution(x0_val, frame=self.frame, device=self.device, dtype=self.dtype)
        mu1_val = EmpiricalDistribution(x1_val, frame=self.frame, device=self.device, dtype=self.dtype)
        
        mu1_train = EmpiricalDistribution(x1_train, frame=self.frame, device=self.device, dtype=self.dtype)
        mu1_test = EmpiricalDistribution(x1_test, frame=self.frame, device=self.device, dtype=self.dtype)
        
        # Create split datasets
        return BrainDataset(
            mu0=mu0_train,
            mu1=mu1_train,
            eigvals=self.eigvals,
            frame=self.frame,
            device=self.device,
            dtype=self.dtype,
        ), BrainDataset(
            mu0=mu0_val,
            mu1=mu1_val,
            eigvals=self.eigvals,
            frame=self.frame,
            device=self.device,
            dtype=self.dtype,
        ), BrainDataset(
            mu0=mu0_test,
            mu1=mu1_test,
            eigvals=self.eigvals,
            frame=self.frame,
            device=self.device,
            dtype=self.dtype,
        )
        