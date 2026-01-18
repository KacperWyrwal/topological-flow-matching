from torch import Tensor
from topofm.config import DATA_DIR
from topofm.data.datasets.fm_dataset import GenerationDataset
from topofm.frames import Coordinates
from topofm.distributions.boundary import EmpiricalDistribution
from topofm.spaces import Space


EARTHQUAKES_DATA_DIR = DATA_DIR / "earthquakes"


class EarthquakesGraph(Space):

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
        return torch.load(EARTHQUAKES_DATA_DIR / 'eigenvalues.pt', map_location="cpu")

    def load_eigvecs(self) -> Tensor:
        return torch.load(EARTHQUAKES_DATA_DIR / 'eigenvectors.pt', map_location="cpu")


class EarthquakesDataset(GenerationDataset):

    @staticmethod
    def load_data() -> Tensor:
        return torch.load(EARTHQUAKES_DATA_DIR / 'x1.pt')

    @classmethod
    def from_disk(
        cls,
        space: EarthquakesGraph,
        ode: ODE,
        device: torch.device,
        dtype: torch.dtype,
        mode: Literal['full', 'identity'] = 'full',
    ) -> "EarthquakesDataset":
        x1 = EarthquakesDataset.load_data().to(device=device, dtype=dtype)
        mu1 = EmpiricalDistribution(x1, space=space)
        mu0 = cls._backward_transport_mu0(mu1, ode, mode=mode) 
        return cls(mu0=mu0, mu1=mu1)

    def _split(
        self, 
        split: tuple[float, float, float] = (0.7, 0.1, 0.2),
        seed: int | None = None,
    ) -> tuple["EarthquakesDataset", "EarthquakesDataset", "EarthquakesDataset"]:
        """
        Split the dataset into training, validation, and test sets.

        Args:
            split (tuple[float, float, float]): The ratio of the dataset to be used for training, validation, and test.

        Returns:
            tuple[EarthquakesDataset, EarthquakesDataset, EarthquakesDataset]: The training, validation, and test datasets.
        """
        return self, self, self