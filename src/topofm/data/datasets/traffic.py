"""
Traffic dataset
"""
import torch 
from torch import Tensor
from topofm.config import DATA_DIR
from topofm.data.datasets.fm_dataset import GenerationDataset


TRAFFIC_DATA_DIR = DATA_DIR / "traffic"


def load_traffic_data(data_dir: str | None = None) -> torch.Tensor:
    y = np.load(os.path.join(data_dir, 'PEMSD4_edge_features_matrix.npz'))['arr_0'].squeeze()
    return torch.as_tensor(y)


def load_traffic_laplacian(data_dir: str | None = None) -> torch.Tensor:
    L = np.load(os.path.join(data_dir, 'PEMSD4_hodge_Laplacian.npz'))['arr_0']
    return torch.as_tensor(L, device='cpu', dtype=torch.float64)


def load_traffic_b1(data_dir: str | None = None) -> torch.Tensor:
    b1 = np.load(os.path.join(data_dir, 'PEMSD4_B1.npz'))['arr_0']
    return torch.as_tensor(b1)


class TrafficDataset(GenerationDataset):
    def __init__(self, mu0_kappa: float, ambient: AmbientCoordinates | str, device: torch.device, dtype: torch.dtype) -> None:
        super().__init__(device=device, dtype=dtype)
        harm_eigvals, grad_eigvals, curl_eigvals = TrafficDataset.load_eigvals(device, dtype)
        harm_eigvecs, grad_eigvecs, curl_eigvecs = TrafficDataset.load_eigvecs(device, dtype)
        # Important to be in the order harmonic, gradient, curl
        eigvecs = torch.concat([harm_eigvecs, grad_eigvecs, curl_eigvecs], dim=-1)
        eigvals = torch.concat([harm_eigvals, grad_eigvals, curl_eigvals], dim=-1)
        frame = Frame(eigvecs=eigvecs, ambient=ambient)
        mu0 = GP(
            harm_sigma=1.0,
            grad_sigma=1.0,
            curl_sigma=1.0,
            harm_kappa=mu0_kappa,
            grad_kappa=mu0_kappa,
            curl_kappa=mu0_kappa,
            harm_eigvals=harm_eigvals,
            grad_eigvals=grad_eigvals,
            curl_eigvals=curl_eigvals,
            frame=frame,
        )
        x1 = TrafficDataset.load_data(device, dtype)
        mu1 = EmpiricalDistribution(x1, frame=frame)
        self._mu0 = mu0
        self._mu1 = mu1
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
    def mu1(self) -> EmpiricalDistribution:
        return self._mu1

    @staticmethod
    def load_data(device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        return torch.load(TRAFFIC_DATA_DIR / "x1.pt").to(device=device, dtype=dtype)
        
    @staticmethod
    def load_eigvals(device: torch.device, dtype: torch.dtype) -> tuple[Tensor, Tensor, Tensor]:
        harm_eigvals = torch.load(TRAFFIC_DATA_DIR / "harmonic_eigenvaluess.pt").to(device, dtype)
        grad_eigvals = torch.load(TRAFFIC_DATA_DIR / "gradient_eigenvaluess.pt").to(device, dtype)
        curl_eigvals = torch.load(TRAFFIC_DATA_DIR / "curl_eigenvaluess.pt").to(device, dtype)
        return harm_eigvals, grad_eigvals, curl_eigvals

    @staticmethod
    def load_eigvecs(device: torch.device, dtype: torch.dtype) -> tuple[Tensor, Tensor, Tensor]:
        harm_eigvecs = torch.load(TRAFFIC_DATA_DIR / "harmonic_eigenvectors.pt").to(device, dtype)
        grad_eigvecs = torch.load(TRAFFIC_DATA_DIR / "gradient_eigenvectors.pt").to(device, dtype)
        curl_eigvecs = torch.load(TRAFFIC_DATA_DIR / "curl_eigenvectors.pt").to(device, dtype)
        return harm_eigvecs, grad_eigvecs, curl_eigvecs
    