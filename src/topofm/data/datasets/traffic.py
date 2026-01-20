"""
Traffic dataset
"""
import torch 
from typing import Literal
from torch import Tensor
from topofm.config import DATA_DIR
from topofm.data.datasets.fm_dataset import GenerationDataset
from topofm.distributions.boundary import EmpiricalDistribution, MultivariateNormal
from topofm.spaces import Space
from topofm.frames.frame import Coordinates
from topofm.odes.ode import ODE
from topofm.utils import to_device, to_dtype


TRAFFIC_DATA_DIR = DATA_DIR / "traffic"


class TrafficComplex(Space):

    @classmethod
    def from_disk(
        cls,
        coords: Coordinates | str,
        dtype: torch.dtype,
        device: torch.device,
    ) -> "TrafficComplex":
        dtype = to_dtype(dtype)
        device = to_device(device)
        eigvals = TrafficComplex.load_eigvals().to(device=device, dtype=dtype)
        eigvecs = TrafficComplex.load_eigvecs().to(device=device, dtype=dtype)
        return cls(eigvals=eigvals, eigvecs=eigvecs, coords=coords)

    @staticmethod
    def load_eigvals() -> Tensor:
        return torch.load(TRAFFIC_DATA_DIR / 'eigenvalues.pt', map_location="cpu")

    @staticmethod
    def load_eigvecs() -> Tensor:
        return torch.load(TRAFFIC_DATA_DIR / 'eigenvectors.pt', map_location="cpu")


def load_traffic_b1(data_dir: str | None = None) -> torch.Tensor:
    b1 = np.load(os.path.join(data_dir, 'PEMSD4_B1.npz'))['arr_0']
    return torch.as_tensor(b1)


class TrafficDataset(GenerationDataset):

    @staticmethod
    def load_data() -> Tensor:
        return torch.load(TRAFFIC_DATA_DIR / "x1.pt")

    @classmethod
    def from_disk(
        cls,
        space: TrafficComplex,
        device: torch.device,
        dtype: torch.dtype,
    ) -> "TrafficDataset":
        dtype = to_dtype(dtype)
        device = to_device(device)
        x1 = TrafficDataset.load_data().to(device=device, dtype=dtype)
        mu1 = EmpiricalDistribution(x1, space=space)
        mu0 = MultivariateNormal(
            mean=torch.zeros(space.dim, dtype=dtype, device=device),
            cov=torch.eye(space.dim, dtype=dtype, device=device),
            space=space,
        )
        return cls(mu0=mu0, mu1=mu1)

    def _split(
        self, 
        ratio: tuple[float, float, float] = (0.7, 0.1, 0.2),
        mu0_covariance_mode: Literal['full', 'diagonal', 'identity'] = 'full',
        mu0_backward_transport: bool = False,
        ode: ODE | None = None,
    ) -> tuple["TrafficDataset", "TrafficDataset", "TrafficDataset"]:
        """
        Split the dataset into training, validation, and test sets.

        Args:
            ratio (tuple[float, float, float]): The ratio of the dataset to be used for training, validation, and test.

        Returns:
            tuple[TrafficDataset, TrafficDataset, TrafficDataset]: The training, validation, and test datasets.
        """
        x1 = self.mu1.samples

        # Shuffle samples
        idx1 = torch.randperm(x1.shape[0])
        x1 = x1[idx1]

        # Split into train, validation, and test
        total_size = x1.shape[0]
        train_size, val_size, test_size = ratio
        x1_train_size = int(train_size * total_size)
        x1_val_size = int(val_size * total_size)
        x1_train = x1[:x1_train_size]
        x1_val = x1[x1_train_size:x1_train_size + x1_val_size]
        x1_test = x1[x1_train_size + x1_val_size:]

        # Create split distributions
        mu1_train = EmpiricalDistribution(x1_train, space=self.mu1.space)
        mu1_val = EmpiricalDistribution(x1_val, space=self.mu1.space)
        mu1_test = EmpiricalDistribution(x1_test, space=self.mu1.space)

        # Create the initial distribution from training set statistics
        mu0_train = self._mu0_from_mu1(
            mu1=mu1_train,
            covariance_mode=mu0_covariance_mode,
            transport=mu0_backward_transport,
            ode=ode,
        )
        mu0_val = mu0_train
        mu0_test = mu0_train
        
        # Create split datasets
        return (
            TrafficDataset(mu0=mu0_train, mu1=mu1_train),
            TrafficDataset(mu0=mu0_val, mu1=mu1_val),
            TrafficDataset(mu0=mu0_test, mu1=mu1_test),
        )
    
