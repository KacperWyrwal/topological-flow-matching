import torch
from torch import Tensor, Size
from topofm.distributions.boundary.boundary_distribution import BoundaryDistribution
from topofm.spaces import Space


class _SpectralGP(BoundaryDistribution):
    """
    Generic Gaussian process.
    """
    def __init__(
        self, 
        grad_eigvals: Tensor,
        curl_eigvals: Tensor,
        harm_eigvals: Tensor,
        harm_sigma: float, 
        grad_sigma: float, 
        curl_sigma: float, 
        harm_kappa: float, 
        grad_kappa: float, 
        curl_kappa: float,
    ):
        """
        Args: 
            harm_evals: [A]
            grad_evals: [B]
            curl_evals: [C]
            harm_evecs: [A, D]
            grad_evecs: [B, D]
            curl_evecs: [C, D]
        """
        super().__init__()

        # Harmonic forms
        harm_spectral_density = _SpectralGP.spectral_density(harm_sigma, harm_kappa, harm_eigvals)
        grad_spectral_density = _SpectralGP.spectral_density(grad_sigma, grad_kappa, grad_eigvals)
        curl_spectral_density = _SpectralGP.spectral_density(curl_sigma, curl_kappa, curl_eigvals)
        self._stddev = torch.concat([harm_spectral_density, grad_spectral_density, curl_spectral_density], dim=0)
        self._mean = torch.zeros_like(self._stddev)
        self.dtype = self._stddev.dtype
        self.device = self._stddev.device

    @staticmethod
    def spectral_density(sigma: float, kappa: float, eigvals: Tensor) -> Tensor:
        """
        Args:
            sigma: float
            kappa: float
            eigvals: [...]
        
        Returns: [...]
        """
        return sigma * torch.exp(-(kappa ** 2.0 / 4.0 * eigvals))

    def sample(self, shape: Size = Size([])) -> Tensor:
        epsilon = torch.randn(*shape, *self._stddev.shape, dtype=self.dtype, device=self.device)
        return self._mean + self._stddev * epsilon

    @property
    def mean(self) -> Tensor:
        return self._mean

    @property 
    def stddev(self) -> Tensor:
        return self._stddev


class GP(BoundaryDistribution):

    def __init__(
        self,
        grad_eigvals: Tensor,
        curl_eigvals: Tensor,
        harm_eigvals: Tensor,
        harm_sigma: float, 
        grad_sigma: float, 
        curl_sigma: float, 
        harm_kappa: float, 
        grad_kappa: float, 
        curl_kappa: float,
        space: Space,
    ):
        super().__init__()
        # TODO Find a way to indicate a requirement that the frame has the 
        # order "harmonic, gradient, curl" of eigenvectors.
        self.space = space
        self._spectral_gp = _SpectralGP(
            grad_eigvals=grad_eigvals,
            curl_eigvals=curl_eigvals,
            harm_eigvals=harm_eigvals,
            harm_sigma=harm_sigma,
            grad_sigma=grad_sigma,
            curl_sigma=curl_sigma,
            harm_kappa=harm_kappa,
            grad_kappa=grad_kappa,
            curl_kappa=curl_kappa,
        )

    def sample(self, shape: Size = Size([])) -> Tensor:
        return self.space.frame.from_spectral(self._spectral_gp.sample(shape))


class NodeGP(GP):
    def __init__(
        self,
        eigvals: Tensor,
        sigma: float,
        kappa: float,
        space: Space,
    ) -> None:
        # TODO Maybe small eigenvalues should be rounded to zero?
        harm_eigvals = eigvals[eigvals == 0.0]
        curl_eigvals = eigvals[eigvals != 0.0]
        grad_eigvals = harm_eigvals.new_empty(0)
        harm_sigma = sigma
        curl_sigma = sigma
        grad_sigma = sigma
        harm_kappa = kappa
        curl_kappa = kappa
        grad_kappa = kappa
        super().__init__(
            grad_eigvals=grad_eigvals,
            curl_eigvals=curl_eigvals,
            harm_eigvals=harm_eigvals,
            harm_sigma=harm_sigma,
            grad_sigma=grad_sigma,
            curl_sigma=curl_sigma,
            harm_kappa=harm_kappa,
            grad_kappa=grad_kappa,
            curl_kappa=curl_kappa,
            space=space,
        )
        