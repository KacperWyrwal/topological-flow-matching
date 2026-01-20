import torch
from torch import Tensor
from topofm.data.datasets import FMDataset
from topofm.config import DATA_DIR
from topofm.distributions.boundary import GP
from topofm.spaces import Space
from topofm.frames import Coordinates
from topofm.utils import to_device, to_dtype


OCEAN_DATA_DIR = DATA_DIR / "ocean"


class OceanComplex(Space):
    def __init__(
        self, 
        harm_eigvals: Tensor,
        grad_eigvals: Tensor,
        curl_eigvals: Tensor,
        harm_eigvecs: Tensor,
        grad_eigvecs: Tensor,
        curl_eigvecs: Tensor,
        coords: Coordinates | str,
        ) -> None:
        eigvals = torch.concat([harm_eigvals, grad_eigvals, curl_eigvals], dim=0)
        eigvecs = torch.concat([harm_eigvecs, grad_eigvecs, curl_eigvecs], dim=1)
        super().__init__(eigvals=eigvals, eigvecs=eigvecs, coords=coords)
        self.harm_eigvals = harm_eigvals
        self.grad_eigvals = grad_eigvals
        self.curl_eigvals = curl_eigvals
        self.harm_eigvecs = harm_eigvecs
        self.grad_eigvecs = grad_eigvecs
        self.curl_eigvecs = curl_eigvecs

    @classmethod
    def from_disk(cls, coords: Coordinates | str, dtype: torch.dtype, device: torch.device) -> "OceanComplex":
        dtype = to_dtype(dtype)
        device = to_device(device)
        harm_eigvals, grad_eigvals, curl_eigvals = cls.load_eigvals()
        harm_eigvals = harm_eigvals.to(device=device, dtype=dtype)
        grad_eigvals = grad_eigvals.to(device=device, dtype=dtype)
        curl_eigvals = curl_eigvals.to(device=device, dtype=dtype)        
        harm_eigvecs, grad_eigvecs, curl_eigvecs = cls.load_eigvecs()
        harm_eigvecs = harm_eigvecs.to(device=device, dtype=dtype)
        grad_eigvecs = grad_eigvecs.to(device=device, dtype=dtype)
        curl_eigvecs = curl_eigvecs.to(device=device, dtype=dtype)
        return cls(
            harm_eigvals=harm_eigvals,
            grad_eigvals=grad_eigvals,
            curl_eigvals=curl_eigvals,
            harm_eigvecs=harm_eigvecs,
            grad_eigvecs=grad_eigvecs,
            curl_eigvecs=curl_eigvecs,
            coords=coords,
        )

    @staticmethod
    def load_eigvals() -> tuple[Tensor, Tensor, Tensor]:
        harm_eigvals = torch.load(OCEAN_DATA_DIR / "harmonic_eigenvalues.pt")
        grad_eigvals = torch.load(OCEAN_DATA_DIR / "gradient_eigenvalues.pt")
        curl_eigvals = torch.load(OCEAN_DATA_DIR / "curl_eigenvalues.pt")
        return harm_eigvals, grad_eigvals, curl_eigvals

    @staticmethod
    def load_eigvecs() -> tuple[Tensor, Tensor, Tensor]:
        harm_eigvecs = torch.load(OCEAN_DATA_DIR / "harmonic_eigenvectors.pt")
        grad_eigvecs = torch.load(OCEAN_DATA_DIR / "gradient_eigenvectors.pt")
        curl_eigvecs = torch.load(OCEAN_DATA_DIR / "curl_eigenvectors.pt")
        return harm_eigvecs, grad_eigvecs, curl_eigvecs


class OceanDataset(FMDataset):

    @classmethod
    def from_disk(
        cls, 
        space: OceanComplex,
        device: torch.device,
        dtype: torch.dtype,
    ) -> "OceanDataset":
        dtype = to_dtype(dtype)
        device = to_device(device)

        mu0 = GP(
            harm_sigma=1e-5,
            grad_sigma=11.808,
            curl_sigma=12.6,
            harm_kappa=0.0,
            grad_kappa=10.36,
            curl_kappa=9.53,
            harm_eigvals=space.harm_eigvals,
            grad_eigvals=space.grad_eigvals,
            curl_eigvals=space.curl_eigvals,
            space=space,
        )
        mu1 = GP(
            harm_sigma=1e-5,
            grad_sigma=1.0,
            curl_sigma=1e-5,
            harm_kappa=0.0,
            grad_kappa=10.0,
            curl_kappa=0.0,
            harm_eigvals=space.harm_eigvals,
            grad_eigvals=space.grad_eigvals,
            curl_eigvals=space.curl_eigvals,
            space=space,
        )
        return cls(
            mu0=mu0,
            mu1=mu1,
        )

    def _split(
        self,
        ratio: tuple[float, float, float] = (0.7, 0.1, 0.2),
    ) -> tuple["OceanDataset", "OceanDataset", "OceanDataset"]:
        return self, self, self
        