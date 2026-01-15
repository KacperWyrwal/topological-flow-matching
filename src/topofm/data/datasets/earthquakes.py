from torch import Tensor
from topofm.config import DATA_DIR
from topofm.distributions.boundary.GP import NodeGP
from topofm.data.datasets.fm_dataset import FMDataset
from topofm.frames.frame import Frame, AmbientCoordinates
from topofm.distributions.empirical import EmpiricalDistribution


EARTHQUAKES_DATA_DIR = DATA_DIR / "earthquakes"


class EarthquakesDataset(FMDataset):

    @staticmethod
    def load_data(device: torch.device, dtype: torch.dtype) -> Tensor:
        return torch.load(EARTHQUAKES_DATA_DIR / 'x1.pt').to(device=device, dtype=dtype)

    @staticmethod
    def load_eigvals(device: torch.device, dtype: torch.dtype) -> Tensor:
        return torch.load(EARTHQUAKES_DATA_DIR / 'eigenvalues.pt').to(device=device, dtype=dtype)

    @staticmethod
    def load_eigvecs(device: torch.device, dtype: torch.dtype) -> Tensor:
        return torch.load(EARTHQUAKES_DATA_DIR / 'eigenvectors.pt').to(device=device, dtype=dtype)


    @classmethod
    def from_disk(
        cls,
        mu0_kappa: float,
        ambient: AmbientCoordinates | str,
        device: torch.device,
        dtype: torch.dtype,
    ) -> FMDataset:
        eigvals = EarthquakesDataset.load_eigvals(device=device, dtype=dtype)
        eigvecs = EarthquakesDataset.load_eigvecs(device=device, dtype=dtype)
        frame = Frame(eigvec=eigvecs, ambient=ambient)
        x1 = EarthquakesDataset.load_data(device=device, dtype=dtype)
        mu1 = EmpiricalDistribution(x1, frame=frame)
        mu0 = NodeGP(
            eigvals=eigvals,
            sigma=1.0, # Could be a good place to estimate the mean and variance from data
            kappa=mu0_kappa,
            frame=frame,
        )
        return cls(mu0=mu0, mu1=mu1, eigvals=eigvals, frame=frame, device=device, dtype=dtype)

    @classmethod
    def train_test_split(cls, test_size: float = 0.2) -> tuple[FMDataset, FMDataset]:
        pass
