import torch
from torch import Tensor
from topofm.config import DATA_DIR
from topofm.data.datasets.fm_dataset import GenerationDataset
from topofm.distributions.boundary import MultivariateNormal
from topofm.frames import Coordinates
from topofm.distributions.boundary import EmpiricalDistribution
from topofm.spaces import Space
from topofm.odes import ODE
from topofm.utils import to_device, to_dtype
from typing import Literal


EARTHQUAKES_DATA_DIR = DATA_DIR / "earthquakes"


class EarthquakesGraph(Space):

    @classmethod
    def from_disk(
        cls,
        coords: Coordinates | str,
        dtype: torch.dtype,
        device: torch.device,
    ) -> "EarthquakesGraph":
        dtype = to_dtype(dtype)
        device = to_device(device)
        eigvals = EarthquakesGraph.load_eigvals().to(device=device, dtype=dtype)
        eigvecs = EarthquakesGraph.load_eigvecs().to(device=device, dtype=dtype)
        return cls(eigvals=eigvals, eigvecs=eigvecs, coords=coords)

    @staticmethod
    def load_eigvals() -> Tensor:
        return torch.load(EARTHQUAKES_DATA_DIR / 'eigenvalues.pt', map_location="cpu")

    @staticmethod
    def load_eigvecs() -> Tensor:
        return torch.load(EARTHQUAKES_DATA_DIR / 'eigenvectors.pt', map_location="cpu")


class EarthquakesDataset(GenerationDataset):

    @staticmethod
    def load_data() -> Tensor:
        return torch.load(EARTHQUAKES_DATA_DIR / 'x1.pt')

    @classmethod
    def from_disk(
        cls,
        space: EarthquakesGraph,
        device: torch.device,
        dtype: torch.dtype,
    ) -> "EarthquakesDataset":
        device = to_device(device)
        dtype = to_dtype(dtype)
        x1 = EarthquakesDataset.load_data().to(device=device, dtype=dtype)
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
    ) -> tuple["EarthquakesDataset", "EarthquakesDataset", "EarthquakesDataset"]:
        """
        Split the dataset into training, validation, and test sets.

        Args:
            ratio (tuple[float, float, float]): The ratio of the dataset to be 
            used for training, validation, and test.

        Returns:
            tuple[EarthquakesDataset, EarthquakesDataset, EarthquakesDataset]: 
            The training, validation, and test datasets.
        """
        # Since the dataset is very small, we don't split. This is in accordance with 
        # Topological Schrodinger Bridge Matching.
        mu1_train = self.mu1
        mu1_val = self.mu1
        mu1_test = self.mu1

        # Use the training set statistics to create the validation and test sets.
        mu0_train = self._mu0_from_mu1(
            mu1=mu1_train,
            covariance_mode=mu0_covariance_mode,
            transport=mu0_backward_transport,
            ode=ode,
        )
        mu0_val = mu0_train
        mu0_test = mu0_train
        return (
            EarthquakesDataset(mu0=mu0_train, mu1=mu1_train),
            EarthquakesDataset(mu0=mu0_val, mu1=mu1_val),
            EarthquakesDataset(mu0=mu0_test, mu1=mu1_test)
        )