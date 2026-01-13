import torch
import pandas as pd
from .fm_dataset import MatchingDataset
from ...config import DATA_DIR
from ...data.boundary import EmpiricalDistribution
from ...frames.frame import Frame, AmbientCoordinates


BRAIN_DATA_DIR = DATA_DIR / "brain"


def load_brain_regions_centroids(data_dir: str | None = None):
    return pd.read_csv(os.path.join(data_dir, "brain_regions_centroids.csv"))


class BrainDataset(MatchingDataset):

    def __init__(self, ambient: AmbientCoordinates | str, device: torch.device, dtype: torch.dtype):
        super().__init__(device=device, dtype=dtype)
        self._eigval = BrainDataset.load_eigval(device=device, dtype=dtype)
        self._eigvec = BrainDataset.load_eigvec(device=device, dtype=dtype)
        
        x0, x1 = BrainDataset.load_data(device=device, dtype=dtype)
        frame = Frame(eigvec=self._eigvec, ambient=ambient)
        self._mu0 = EmpiricalDistribution(x0, frame=frame)
        self._mu1 = EmpiricalDistribution(x1, frame=frame)

    @property
    def eigvec(self) -> Tensor:
        return self._eigvec

    @property
    def eigval(self) -> Tensor:
        return self._eigval

    @property
    def mu0(self) -> EmpiricalDistribution:
        return self._mu0

    @property
    def mu1(self) -> EmpiricalDistribution:
        return self._mu1

    @staticmethod
    def load_data(device: torch.device, dtype: torch.dtype) -> tuple[Tensor, Tensor]:
        x0 = torch.load(BRAIN_DATA_DIR / 'x0_liberal.pt').to(device, dtype)
        x1 = torch.load(BRAIN_DATA_DIR / 'x1_aligned.pt').to(device, dtype)
        return x0, x1

    @staticmethod
    def load_eigval(device: torch.device, dtype: torch.dtype) -> Tensor:
        eigval = torch.load(BRAIN_DATA_DIR / 'eigenvalues.pt').to(device, dtype)
        return eigval

    @staticmethod
    def load_eigvec(device: torch.device, dtype: torch.dtype) -> Tensor:
        eigvec = torch.load(BRAIN_DATA_DIR / 'eigenvectors.pt').to(device, dtype)
        return eigvec
