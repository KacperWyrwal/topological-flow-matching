import torch
from torch import Tensor, Size, distributions
from topofm.distributions.boundary.boundary_distribution import BoundaryDistribution
from topofm.covariance import Covariance


class MultivariateNormal(BoundaryDistribution):
    def __init__(
        self, 
        mean: Tensor, 
        cov: Tensor | Covariance, 
    ) -> None:
        super().__init__()
        if isinstance(cov, Covariance):
            cov = cov.matrix
        self._base_dist = distributions.MultivariateNormal(mean, cov)

    def sample(self, shape: Size = Size([])) -> Tensor:
        return self._base_dist.sample(shape)
