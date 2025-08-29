import torch
import math
from abc import abstractmethod
from torch.distributions import Distribution
from .frames import Frame, StandardFrame
# HodgeBasis is imported here to avoid circular import
# from .data import HodgeBasis


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
        self.spectral_stddev = self.spectral_variance.sqrt()
        self.eigenvectors = torch.concat([harm_vecs, grad_vecs, curl_vecs], dim=0) # [A + B + C, D]
        
        # Shapes 
        self._batch_shape = torch.Size()
        self._event_shape = self.spectral_variance.shape

    def sample(self, shape: torch.Size):
        return self.sample_spectral(shape)

    def sample_spectral(self, shape: torch.Size):
        """
        Sample spectral weights. 

        Args: 
            shape: [...]

        returns: [..., 3M]
        """
        epsilon = torch.randn(*shape, *self._event_shape) # [..., 3M]
        return self.spectral_stddev * epsilon # [..., 3M]
        
    def sample_euclidean(self, shape: torch.Size):
        spectral_samples = self.sample_spectral(shape) # [..., 3M]
        return torch.einsum('md, ...m -> ...d', self.eigenvectors, spectral_samples)


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
        assert samples.shape[0] > 0, f"Empirical distribution has no samples. {samples.shape=}"
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
        assert self.num_samples > 0, f"Empirical distribution has no samples. {self.samples.shape=}"
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
        self._event_shape = None # Will be computed on first access

    @abstractmethod
    def sample(self, shape: torch.Size) -> torch.Tensor: ... 

    @property
    def event_shape(self) -> torch.Size:
        if self._event_shape is None:
            self._event_shape = self.sample(torch.Size()).shape
        return self._event_shape


class EmpiricalInFrame(InFrame):
    def __init__(self, base: Empirical, frame: Frame | None = None) -> None:
        assert isinstance(base, Empirical), "EmpiricalInFrame requires an Empirical base distribution."
        super().__init__(base, frame)
        # Precompute transformed samples for efficiency
        self._base_transformed = Empirical(samples=self.frame.transform(base.samples))

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
