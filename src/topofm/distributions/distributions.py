import torch
from torch.distributions import Distribution
from .frames import Frame, StandardFrame


class EdgeGP(Distribution):
    """
    Zero-mean GP on edges of a 2-simplicial complex. 
    """
    def __init__(
        self, 
        grad_vecs: torch.Tensor,
        curl_vecs: torch.Tensor,
        harm_vecs: torch.Tensor,
        grad_vals: torch.Tensor,
        curl_vals: torch.Tensor,
        harm_vals: torch.Tensor,
        gp_type: str = 'diffusion',
        harm_sigma: float = 1.0, 
        grad_sigma: float = 1.0, 
        curl_sigma: float = 1.0, 
        harm_kappa: float = 1.0, 
        grad_kappa: float = 1.0, 
        curl_kappa: float = 1.0, 
    ):
        """
        Args: 
            harm_evals: [A]
            grad_evals: [B]
            curl_evals: [C]
            harm_evecs: [A, D]
            grad_evecs: [B, D]
            curl_evecs: [C, D]
            gp_type: 'diffusion'
        """
        super().__init__(validate_args=False)
        self.gp_type = gp_type

        # Harmonic forms 
        harm_variance = harm_sigma * torch.exp(- harm_kappa ** 2.0 / 2.0 * harm_vals) # [A]
        grad_variance = grad_sigma * torch.exp(- grad_kappa ** 2.0 / 2.0 * grad_vals) # [B]
        curl_variance = curl_sigma * torch.exp(- curl_kappa ** 2.0 / 2.0 * curl_vals) # [C]

        # Reshape eigenvalues and spectral variance
        self.spectral_variance = torch.concat([harm_variance, grad_variance, curl_variance], dim=0) # [A + B + C]
        # self.spectral_variance = self.spectral_variance / self.spectral_variance.sum().sqrt()
        self.spectral_stddev = self.spectral_variance.sqrt()
        self.eigenvectors = torch.concat([harm_vecs, grad_vecs, curl_vecs], dim=0) # [A + B + C, D]
        
        # Shapes 
        self._batch_shape = torch.Size()
        self._event_shape = self.spectral_variance.shape

        self.spectral_mean = torch.zeros_like(self.spectral_variance)
        self.spectral_noise = torch.ones_like(self.spectral_variance)

        self.dtype = self.spectral_stddev.dtype
        self.device = self.spectral_stddev.device
        print(f"EdgeGP dtype: {self.dtype}, device: {self.device}")

    def sample(self, shape: torch.Size):
        return self.sample_spectral(shape)

    def sample_spectral(self, shape: torch.Size):
        """
        Sample spectral weights. 

        Args: 
            shape: [...]

        returns: [..., 3M]
        """
        epsilon = torch.randn(*shape, *self._event_shape, dtype=self.dtype, device=self.device) # [..., 3M]
        return self.spectral_mean + self.spectral_noise * self.spectral_stddev * epsilon # [..., 3M]
        
    def sample_euclidean(self, shape: torch.Size):
        spectral_samples = self.sample_spectral(shape) # [..., 3M]
        return torch.einsum('md, ...m -> ...d', self.eigenvectors, spectral_samples)

    @property
    def loc(self) -> torch.Tensor:
        return self.spectral_mean
    
    @loc.setter
    def loc(self, value_spectral: torch.Tensor):
        self.spectral_mean = value_spectral
    
    @property
    def scale(self) -> torch.Tensor:
        return self.spectral_noise
    
    @scale.setter
    def scale(self, value_spectral: torch.Tensor):
        self.spectral_noise = value_spectral


class Moons(Distribution):
    """A PyTorch Distribution representing the two moons dataset."""

    def __init__(self, noise_std: float = 0.05) -> None:
        super().__init__(validate_args=False)
        self.noise_std = noise_std
        self._batch_shape = torch.Size()
        self._event_shape = torch.Size([2])

    @property
    def batch_shape(self) -> torch.Size:
        return self._batch_shape

    @property
    def event_shape(self) -> torch.Size:
        return self._event_shape

    def sample(self, shape) -> torch.Tensor:
        from .utils import sample_moons

        return sample_moons(shape, noise_std=self.noise_std)


class EightGaussians(Distribution):
    """A PyTorch Distribution representing eight Gaussians on a circle."""

    def __init__(self, radius: float = 2.0, noise_std: float = 0.2) -> None:
        super().__init__(validate_args=False)
        self.radius = radius
        self.noise_std = noise_std
        self._batch_shape = torch.Size()
        self._event_shape = torch.Size([2])

    @property
    def batch_shape(self) -> torch.Size:
        return self._batch_shape

    @property
    def event_shape(self) -> torch.Size:
        return self._event_shape

    def sample(self, shape) -> torch.Tensor:
        from .utils import sample_eight_gaussians

        return sample_eight_gaussians(shape, radius=self.radius, noise_std=self.noise_std)


class PossiblyDegenerateNormal(torch.distributions.Normal):
    """Normal distribution that supports zero scale in a Dirac-like sense."""

    arg_constraints = {
        "loc": torch.distributions.constraints.real,
        "scale": torch.distributions.constraints.greater_than_eq(0.0),
    }

    def __init__(
        self,
        loc: torch.Tensor | float,
        scale: torch.Tensor | float,
        validate_args: bool | None = None,
    ) -> None:
        super().__init__(loc=loc, scale=scale, validate_args=validate_args)
        self.is_degenerate = scale == 0.0

    def log_prob(self, value: torch.Tensor) -> torch.Tensor:
        log_prob_if_nondegenerate = super().log_prob(value)
        log_prob_if_degenerate = torch.where(self.loc == value, 0.0, -torch.inf)
        return torch.where(self.is_degenerate, log_prob_if_degenerate, log_prob_if_nondegenerate)

    def cdf(self, value: torch.Tensor) -> torch.Tensor:
        cdf_if_nondegenerate = super().cdf(value)
        cdf_if_degenerate = torch.where(self.loc <= value, 1.0, 0.0)
        return torch.where(self.is_degenerate, cdf_if_degenerate, cdf_if_nondegenerate)

    def icdf(self, value: torch.Tensor) -> torch.Tensor:
        icdf_if_nondegenerate = super().icdf(value)
        icdf_if_degenerate = self.loc
        return torch.where(self.is_degenerate, icdf_if_degenerate, icdf_if_nondegenerate)
