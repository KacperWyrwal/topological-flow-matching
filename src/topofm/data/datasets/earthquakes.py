from .fm_dataset import GenerationDataset
from ...frames.frame import Frame, AmbientCoordinates


EARTHQUAKES_DATA_DIR = DATA_DIR / "earthquakes"


class EarthquakesDataset(GenerationDataset):

    def __init__(self, ambient: AmbientCoordinates | str, device: torch.device, dtype: torch.dtype):
        super().__init__(device=device, dtype=dtype)
        self._eigval = EarthquakesDataset.load_eigval(device=device, dtype=dtype)
        self._eigvec = EarthquakesDataset.load_eigvec(device=device, dtype=dtype)
        x1 = EarthquakesDataset.load_data(device=device, dtype=dtype)
        frame = Frame(eigvec=self._eigvec, ambient=ambient)
        self._mu1 = EmpiricalDistribution(x1, frame=frame)

    @staticmethod
    def load_data(device: torch.device, dtype: torch.dtype) -> Tensor:
        return torch.load(EARTHQUAKES_DATA_DIR / 'x1.pt').to(device=device, dtype=dtype)

    @staticmethod
    def load_eigval(device: torch.device, dtype: torch.dtype) -> Tensor:
        return torch.load(EARTHQUAKES_DATA_DIR / 'eigenvalues.pt').to(device=device, dtype=dtype)

    @staticmethod
    def load_eigvec(device: torch.device, dtype: torch.dtype) -> Tensor:
        return torch.load(EARTHQUAKES_DATA_DIR / 'eigenvectors.pt').to(device=device, dtype=dtype)
