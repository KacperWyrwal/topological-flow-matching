from pandas._libs.groupby import group_cumsum
import torch
from abc import abstractmethod
from torch.distributions import Distribution
from .frames import Frame, StandardFrame


class EdgeGP(Distribution):
    def __init__(
        self, 
        eigenvalues: tuple[torch.Tensor, torch.Tensor, torch.Tensor], 
        eigenvectors: tuple[torch.Tensor, torch.Tensor, torch.Tensor], 
        gp_type: str = 'diffusion',
        harm_sigma: float = 1.0, 
        grad_sigma: float = 1.0, 
        curl_sigma: float = 1.0, 
        harm_kappa: float = 1.0, 
        grad_kappa: float = 1.0, 
        curl_kappa: float = 1.0, 
        alpha: float = 1.0,
    ):
        self.harm_evals, self.grad_evals, self.curl_evals = eigenvalues # [M]
        self.harm_evecs, self.grad_evecs, self.curl_evecs = eigenvectors # [M, D]

        self.gp_type = gp_type
        self.harm_sigma2 = harm_sigma
        self.grad_sigma2 = grad_sigma
        self.curl_sigma2 = curl_sigma
        self.harm_kappa = harm_kappa
        self.grad_kappa = grad_kappa
        self.curl_kappa = curl_kappa
        self.alpha = alpha
        self._mean = torch.zeros(self.dim) # Probably set by 

        self.cov_spectral = self.get_cov_spectral()
        self.hodge_evecs = torch.cat([self.harm_evecs, self.grad_evecs, self.curl_evecs], dim=1) # [M, 3D]

    @property
    def dim(self) -> int:
        return self.harm_evals.shape[0] # M
    
    def mean(self) -> torch.Tensor:
        return self._mean
                       
    def get_cov_spectral(self):
        harm_cov = self.harm_sigma2 * torch.exp(- self.harm_kappa**2/2 * self.harm_evals)
        grad_cov = self.grad_sigma2 * torch.exp(- self.grad_kappa**2/2 * self.grad_evals)
        curl_cov = self.curl_sigma2 * torch.exp(- self.curl_kappa**2/2 * self.curl_evals)
        return torch.cat([harm_cov, grad_cov, curl_cov], dim=1)

    def ocean_sampler_gp(self, mean, cov_spectral, hodge_evecs, sample_size):
        '''use this to sample from the GP for initial distribution '''
        v = torch.randn(sample_size, self.dim)

        
        n_spectrals = cov_spectral.shape[0]
        v = np.random.multivariate_normal(np.zeros(n_spectrals), np.eye(n_spectrals), sample_size)
        assert v.shape == (sample_size,) + (n_spectrals,)
        ocean_flows = mean[:,None] + hodge_evecs @ (np.sqrt(cov_spectral)[:, None] * v.T)
        return ocean_flows.T
        
    def sample(self, shape: torch.Size):
        cov_spectral = self.cov_spectral # [M, 3M]
        hodge_evecs = self.hodge_evecs # [M, 3D]
        epsilon = torch.randn(*shape, self.dim) # []
        return (
            self.mean + 
            self.eigenvectors @ (
                torch.sqrt(cov_spectral) * epsilon.mT # [M, 3M] @ [3M, ]
            )
        )
        return self.mean + self.eigenvectors @ (torch.sqrt(cov_spectral)[:, None] * epsilon.mT)
        


        ocean_samples = ocean_sampler_gp(self.mean, cov_spectral, hodge_evecs, n, if_seed)
        return torch.Tensor(ocean_samples)


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


class Empirical(Distribution):
    """Empirical distribution defined by a finite set of samples (with replacement).

    Samples are drawn by indexing the stored tensor along the first dimension.

    Args:
        samples: Tensor of shape (N, *event_shape)
    """

    def __init__(self, samples: torch.Tensor) -> None:
        super().__init__(validate_args=False)
        assert samples.ndim == 2, "`samples` should be of shape [N, D]"
        self._samples = samples
        self._batch_shape = torch.Size()
        self._event_shape = torch.Size([samples.shape[-1]])

    @property
    def samples(self) -> torch.Tensor:
        return self._samples

    @property 
    def num_samples(self) -> int:
        return self.samples.shape[0]

    @property
    def batch_shape(self) -> torch.Size:
        return self._batch_shape

    @property
    def event_shape(self) -> torch.Size:
        return self._event_shape

    def sample(self, shape: torch.Size) -> torch.Tensor:
        assert len(shape) <= 1, "Sample shape should be (S,) or ()."
        indices = torch.randint(0, self.num_samples, shape, device=self.samples.device)
        return self.samples[indices]

    def __getitem__(self, idx) -> "Empirical":
        return Empirical(samples=self.samples[idx])


class InFrame(Distribution):
    def __init__(self, base: Distribution, frame: Frame | None = None) -> None:
        assert not isinstance(base, InFrame), "Wrapping InFrame in InFrame is likely not the indended usage."
        super().__init__(validate_args=False)
        self.frame = StandardFrame() if frame is None else frame
        self.base = base

    @abstractmethod
    def sample(self, shape: torch.Size) -> torch.Tensor: ... 

    @property
    def event_shape(self) -> torch.Size:
        return self.base.event_shape


class EmpiricalInFrame(InFrame):
    def __init__(self, base: Empirical, frame: Frame | None = None) -> None:
        assert isinstance(base, Empirical), "EmpiricalInFrame requires an Empirical base distribution."
        super().__init__(base, frame)
        # Precompute transformed samples for efficiency
        self._base_transformed = Empirical(samples=self.frame.transform(base.samples))
        self.base = base

    @property 
    def samples(self) -> torch.Tensor:
        return self._base_transformed.samples

    def sample(self, shape: torch.Size) -> torch.Tensor:
        # Already transformed, just sample
        return self._base_transformed.sample(shape)

    def __getitem__(self, idx) -> "EmpiricalInFrame":
        return type(self)(base=self.base[idx], frame=self.frame)

    @property
    def num_samples(self) -> int:
        return self._base_transformed.num_samples


class AnalyticInFrame(InFrame):
    def __init__(self, base: Distribution, frame: Frame | None = None) -> None:
        assert not isinstance(base, Empirical), "AnalyticInFrame should not wrap an Empirical distribution."
        super().__init__(base, frame)

    def sample(self, shape: torch.Size) -> torch.Tensor:
        x = self.base.sample(shape)
        return self.frame.transform(x)
