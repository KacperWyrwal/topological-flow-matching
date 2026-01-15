from .fm_dataset import MatchingDataset
from ...config import DATA_DIR
from ...distributions.boundary import GP
from ...frames import Frame
from ...ambient_coordinates import AmbientCoordinates


OCEAN_DATA_DIR = DATA_DIR / "ocean"


class OceanDataset(MatchingDataset):

    def __init__(self, device: torch.device, dtype: torch.dtype, ambient: AmbientCoordinates | str = AmbientCoordinates.STANDARD):
        super().__init__(device=device, dtype=dtype)
        harm_eigvals, grad_eigvals, curl_eigvals = OceanDataset.load_eigvals(device, dtype)
        harm_eigvecs, grad_eigvecs, curl_eigvecs = OceanDataset.load_eigvecs(device, dtype)
        # Important to be in the order harmonic, gradient, curl
        eigvecs = torch.concat([harm_eigvecs, grad_eigvecs, curl_eigvecs], dim=-1)
        eigvals = torch.concat([harm_eigvals, grad_eigvals, curl_eigvals], dim=-1)
        frame = Frame(eigvecs=eigvecs, ambient=ambient)
        self._mu0 = GP(
            harm_sigma=1e-5,
            grad_sigma=11.808,
            curl_sigma=12.6,
            harm_kappa=0.0,
            grad_kappa=10.36,
            curl_kappa=9.53,
            harm_eigvals=harm_eigvals,
            grad_eigvals=grad_eigvals,
            curl_eigvals=curl_eigvals,
            frame=frame,
        )
        self._mu1 = GP(
            harm_sigma=1e-5,
            grad_sigma=1.0,
            curl_sigma=1e-5,
            harm_kappa=0.0,
            grad_kappa=10.0,
            curl_kappa=0.0,
            harm_eigvals=harm_eigvals,
            grad_eigvals=grad_eigvals,
            curl_eigvals=curl_eigvals,
            frame=frame,
        )
        self._frame = frame
        self._eigvals = eigvals

    @property
    def eigvals(self) -> Tensor:
        return self._eigvals
    
    @property
    def frame(self) -> Frame:
        return self._frame

    @property
    def mu0(self) -> GP:
        return self._mu0

    @property
    def mu1(self) -> GP:
        return self._mu1

    @staticmethod
    def load_eigvals(device: torch.device, dtype: torch.dtype) -> tuple[Tensor, Tensor, Tensor]:
        harm_eigvals = torch.load(OCEAN_DATA_DIR / "harmonic_eigenvaluess.pt").to(device, dtype)
        grad_eigvals = torch.load(OCEAN_DATA_DIR / "gradient_eigenvaluess.pt").to(device, dtype)
        curl_eigvals = torch.load(OCEAN_DATA_DIR / "curl_eigenvaluess.pt").to(device, dtype)
        return harm_eigvals, grad_eigvals, curl_eigvals

    @staticmethod
    def load_eigvecs(device: torch.device, dtype: torch.dtype) -> tuple[Tensor, Tensor, Tensor]:
        harm_eigvecs = torch.load(OCEAN_DATA_DIR / "harmonic_eigenvectors.pt").to(device, dtype)
        grad_eigvecs = torch.load(OCEAN_DATA_DIR / "gradient_eigenvectors.pt").to(device, dtype)
        curl_eigvecs = torch.load(OCEAN_DATA_DIR / "curl_eigenvectors.pt").to(device, dtype)
        return harm_eigvecs, grad_eigvecs, curl_eigvecs
