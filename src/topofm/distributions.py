import torch
from abc import abstractmethod
from torch.distributions import Distribution
from .frames import Frame, StandardFrame


class Moons(Distribution):
    """A PyTorch Distribution representing the two moons dataset."""

    def __init__(self, noise_std: float = 0.05) -> None:
        super().__init__(validate_args=False)
        self.noise_std = noise_std

    def sample(self, shape) -> torch.Tensor:
        from .utils import sample_moons

        return sample_moons(shape, noise_std=self.noise_std)


class EightGaussians(Distribution):
    """A PyTorch Distribution representing eight Gaussians on a circle."""

    def __init__(self, radius: float = 2.0, noise_std: float = 0.2) -> None:
        super().__init__(validate_args=False)
        self.radius = radius
        self.noise_std = noise_std

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

    def __init__(self, samples: torch.Tensor, validate_args: bool | None = None) -> None:
        super().__init__(validate_args=validate_args)
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


class EmpiricalInFrame(InFrame):
    def __init__(self, base: Empirical, frame: Frame | None = None) -> None:
        assert isinstance(base, Empirical), "EmpiricalInFrame requires an Empirical base distribution."
        super().__init__(base, frame)
        # Precompute transformed samples for efficiency
        self.base = Empirical(samples=self.frame.transform(base.samples))

    def sample(self, shape: torch.Size) -> torch.Tensor:
        # Already transformed, just sample
        return self.base.sample(shape)

    def __getitem__(self, idx) -> "EmpiricalInFrame":
        return type(self)(base=self.base[idx], frame=self.frame)

    @property
    def num_samples(self) -> int:
        return self.base.num_samples


class AnalyticInFrame(InFrame):
    def __init__(self, base: Distribution, frame: Frame | None = None) -> None:
        assert not isinstance(base, Empirical), "AnalyticInFrame should not wrap an Empirical distribution."
        super().__init__(base, frame)

    def sample(self, shape: torch.Size) -> torch.Tensor:
        x = self.base.sample(shape)
        return self.frame.transform(x)
