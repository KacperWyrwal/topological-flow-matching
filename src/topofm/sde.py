from abc import ABC, abstractmethod
import torch

from .distributions import PossiblyDegenerateNormal


class SDE(ABC):
    def __init__(self, dim: int):
        self._dim = dim

    @property
    def dim(self) -> int:
        return self._dim

    @abstractmethod
    def drift(self, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor: ...

    @abstractmethod
    def diffusion(self, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor: ...

    @abstractmethod
    def transition(self, t0: torch.Tensor, t1: torch.Tensor) -> torch.Tensor: ...

    def _mean(self, t: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    @abstractmethod
    def _mean_with_initial_condition(
        self, t: torch.Tensor, *, t0: torch.Tensor | None = None, x0: torch.Tensor | None = None
    ) -> torch.Tensor: ...

    @abstractmethod
    def _mean_with_initial_and_final_condition(
        self,
        t: torch.Tensor,
        *,
        t0: torch.Tensor | None = None,
        x0: torch.Tensor | None = None,
        t1: torch.Tensor | None = None,
        x1: torch.Tensor | None = None,
    ) -> torch.Tensor: ...

    def mean(
        self,
        t: torch.Tensor,
        *,
        t0: torch.Tensor | None = None,
        x0: torch.Tensor | None = None,
        t1: torch.Tensor | None = None,
        x1: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if x0 is not None and x1 is not None:
            if t0 is None:
                t0 = t.new_zeros(t.shape)
            if t1 is None:
                t1 = t.new_ones(t.shape)
            return self._mean_with_initial_and_final_condition(t=t, t0=t0, x0=x0, t1=t1, x1=x1)
        elif x0 is not None:
            if t0 is None:
                t0 = t.new_zeros(t.shape)
            return self._mean_with_initial_condition(t=t, t0=t0, x0=x0)
        else:
            return self._mean(t)

    def _variance(self, t: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    @abstractmethod
    def _variance_with_initial_condition(self, t: torch.Tensor, *, t0: torch.Tensor | None = None) -> torch.Tensor: ...

    def _variance_with_initial_and_final_condition(
        self, t: torch.Tensor, *, t0: torch.Tensor | None = None, t1: torch.Tensor | None = None
    ) -> torch.Tensor:
        raise NotImplementedError

    def variance(
        self,
        t: torch.Tensor,
        *,
        t0: torch.Tensor | bool | None = None,
        t1: torch.Tensor | bool | None = None,
    ) -> torch.Tensor:
        if t0 is True:
            t0 = t.new_zeros(t.shape)
        if t1 is True:
            t1 = t.new_ones(t.shape)
        if t0 is not None and t1 is not None:
            return self._variance_with_initial_and_final_condition(t=t, t0=t0, t1=t1)
        elif t0 is not None:
            return self._variance_with_initial_condition(t=t, t0=t0)
        else:
            return self._variance(t)

    def _covariance(self, s: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def _covariance_with_initial_condition(
        self, s: torch.Tensor, t: torch.Tensor, *, t0: torch.Tensor | None = None
    ) -> torch.Tensor:
        r = torch.minimum(s, t)
        return self.transition(t0=r, t1=t) * self.variance(t=r, t0=t0) * self.transition(t0=r, t1=s)

    def _covariance_with_initial_and_final_condition(
        self, s: torch.Tensor, t: torch.Tensor, *, t0: torch.Tensor | None = None, t1: torch.Tensor | None = None
    ) -> torch.Tensor:
        raise NotImplementedError

    def covariance(
        self, s: torch.Tensor, t: torch.Tensor, *, t0: torch.Tensor | bool | None = None, t1: torch.Tensor | bool | None = None
    ) -> torch.Tensor:
        if (t0 is True or t0 is not None) and (t1 is True or t1 is not None):
            return self._covariance_with_initial_and_final_condition(s=s, t=t, t0=t0, t1=t1)
        elif t0 is True or t0 is not None:
            return self._covariance_with_initial_condition(s=s, t=t, t0=t0)
        else:
            return self._covariance(s=s, t=t)

    @abstractmethod
    def marginal_distribution(
        self,
        t: torch.Tensor,
        t0: torch.Tensor | None = None,
        x0: torch.Tensor | None = None,
        t1: torch.Tensor | None = None,
        x1: torch.Tensor | None = None,
    ) -> torch.Tensor: ...


class DiagonalSDE(SDE):
    def _variance_with_initial_and_final_condition(
        self, t: torch.Tensor, *, t0: torch.Tensor | None = None, t1: torch.Tensor | None = None
    ) -> torch.Tensor:
        sigmatt = self.variance(t=t, t0=t0)
        sigmat1t = self.covariance(s=t, t=t1, t0=t0)
        sigmat1t1 = self.variance(t=t1, t0=t0)
        return sigmatt - sigmat1t.square() / sigmat1t1

    def marginal_distribution(
        self,
        t: torch.Tensor,
        t0: torch.Tensor | None = None,
        x0: torch.Tensor | None = None,
        t1: torch.Tensor | None = None,
        x1: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if x0 is not None and t0 is None:
            t0 = t.new_zeros(t.shape)
        if x1 is not None and t1 is None:
            t1 = t.new_ones(t.shape)
        mean = self.mean(t=t, t0=t0, x0=x0, t1=t1, x1=x1)
        variance = self.variance(t=t, t0=t0, t1=t1)
        normal = PossiblyDegenerateNormal(loc=mean, scale=variance.sqrt())
        return torch.distributions.Independent(normal, reinterpreted_batch_ndims=1)


class OUDiagonalSDE(DiagonalSDE):
    def __init__(self, alpha_diagonal: torch.Tensor, gamma_diagonal: torch.Tensor):
        super().__init__(dim=alpha_diagonal.shape[0])
        self.alpha_diagonal = alpha_diagonal
        self.gamma_diagonal = gamma_diagonal

    def drift(self, t: torch.Tensor | None = None, x: torch.Tensor | None = None) -> torch.Tensor:
        return self.alpha_diagonal * x

    def diffusion(self, t: torch.Tensor | None = None, x: torch.Tensor | None = None) -> torch.Tensor:
        return self.gamma_diagonal

    def transition(self, t0: torch.Tensor, t1: torch.Tensor) -> torch.Tensor:
        delta = t1 - t0
        return torch.exp(torch.einsum("d, ... -> ...d", self.alpha_diagonal, delta))

    def _variance_with_initial_condition_over_gamma_squared(
        self, t: torch.Tensor, *, t0: torch.Tensor | None = None, min_alpha_nonzero: float = 1e-8
    ) -> torch.Tensor:
        variance_if_alpha_zero = (t - t0).unsqueeze(-1)
        variance_if_alpha_nonzero = (self.transition(t0=t0, t1=t).square() - 1) / (2 * self.alpha_diagonal)
        return torch.where(
            self.alpha_diagonal.abs() < min_alpha_nonzero, variance_if_alpha_zero, variance_if_alpha_nonzero
        )

    def _covariance_with_initial_condition_over_gamma_squared(
        self, s: torch.Tensor, t: torch.Tensor, *, t0: torch.Tensor | None = None, min_alpha_nonzero: float = 1e-8
    ) -> torch.Tensor:
        r = torch.minimum(s, t)
        return (
            self.transition(t0=r, t1=t)
            * self._variance_with_initial_condition_over_gamma_squared(t=r, t0=t0, min_alpha_nonzero=min_alpha_nonzero)
            * self.transition(t0=r, t1=s)
        )

    def _variance_with_initial_condition(
        self, t: torch.Tensor, *, t0: torch.Tensor | None = None, min_alpha_nonzero: float = 1e-8
    ) -> torch.Tensor:
        return self.gamma_diagonal.square() * self._variance_with_initial_condition_over_gamma_squared(
            t=t, t0=t0, min_alpha_nonzero=min_alpha_nonzero
        )

    def _mean_with_initial_condition(
        self, t: torch.Tensor, t0: torch.Tensor | None = None, x0: torch.Tensor | None = None
    ) -> torch.Tensor:
        return self.transition(t0=t0, t1=t) * x0

    def _mean_with_initial_and_final_condition(
        self,
        t: torch.Tensor,
        t0: torch.Tensor | None = None,
        x0: torch.Tensor | None = None,
        t1: torch.Tensor | None = None,
        x1: torch.Tensor | None = None,
    ) -> torch.Tensor:
        m_t = self._mean_with_initial_condition(t=t, t0=t0, x0=x0)
        m_t1 = self._mean_with_initial_condition(t=t1, t0=t0, x0=x0)
        sigma_t1t_over_g2 = self._covariance_with_initial_condition_over_gamma_squared(s=t, t=t1, t0=t0)
        sigma_t1t1_over_g2 = self._variance_with_initial_condition_over_gamma_squared(t=t1, t0=t0)
        return m_t + sigma_t1t_over_g2 / sigma_t1t1_over_g2 * (x1 - m_t1)


class PossiblyDegenerateOUDiagonalSDE(OUDiagonalSDE):
    def __init__(self, alpha_diagonal: torch.Tensor, gamma_diagonal: torch.Tensor):
        super().__init__(alpha_diagonal, gamma_diagonal)
        self.is_degenerate = self.gamma_diagonal == 0.0

    def variance(self, t: torch.Tensor, *, t0: torch.Tensor | bool | None = None, t1: torch.Tensor | bool | None = None) -> torch.Tensor:
        return torch.where(self.is_degenerate, 0.0, super().variance(t=t, t0=t0, t1=t1))

    def covariance(
        self, s: torch.Tensor, t: torch.Tensor, *, t0: torch.Tensor | bool | None = None, t1: torch.Tensor | bool | None = None
    ) -> torch.Tensor:
        return torch.where(self.is_degenerate, 0.0, super().covariance(s=s, t=t, t0=t0, t1=t1))


class HeatBMTSDE(PossiblyDegenerateOUDiagonalSDE):
    def __init__(self, eigenvalues: torch.Tensor, c: float, sigma: torch.Tensor) -> None:
        c = eigenvalues.new_tensor(c)
        sigma = eigenvalues.new_tensor(sigma)
        alpha_diagonal = -c * eigenvalues
        gamma_diagonal = sigma * torch.ones_like(eigenvalues)
        super().__init__(alpha_diagonal=alpha_diagonal, gamma_diagonal=gamma_diagonal)


