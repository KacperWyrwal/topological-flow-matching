import torch


class Moons(torch.distributions.Distribution):
    """A PyTorch Distribution representing the two moons dataset."""

    def __init__(self, noise_std: float = 0.05) -> None:
        super().__init__(validate_args=False)
        self.noise_std = noise_std

    def sample(self, shape) -> torch.Tensor:
        from .utils import sample_moons

        return sample_moons(shape, noise_std=self.noise_std)


class EightGaussians(torch.distributions.Distribution):
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


