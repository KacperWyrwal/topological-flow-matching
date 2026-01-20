import torch
from torch import Tensor, Size, distributions
from topofm.distributions.boundary.boundary_distribution import BoundaryDistribution
from topofm.distributions.covariance import Covariance
from topofm.utils import numerical_error_threshold, ensure_psd
from topofm.spaces import Space


class MultivariateNormal(BoundaryDistribution):
    def __init__(
        self, 
        mean: Tensor, 
        cov: Tensor | Covariance,
        space: Space,
    ) -> None:
        super().__init__()
        if not isinstance(cov, Covariance):
            cov = Covariance(cov)
        self.cov = cov
        self.mean = mean
        self.space = space
        self.dim = mean.shape[-1]

        # Eigendecomposition and fix small eigenvalues to 0
        self.sqrt_eigvals = torch.sqrt(cov.eigvals)
        self.eigvecs = cov.eigvecs

    def sample(self, shape: Size = Size([])) -> Tensor:
        eps = torch.randn((*shape, self.dim), dtype=self.mean.dtype, device=self.mean.device)

        return (
            self.mean +
            torch.einsum('ij,...j->...i', self.eigvecs, self.sqrt_eigvals * eps)
        )

