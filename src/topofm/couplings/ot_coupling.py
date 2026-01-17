import torch
import math
from torch import Tensor, Size
from geomloss import SamplesLoss
from ot import emd
from topofm.couplings.coupling import Coupling
from topofm.distributions.boundary.empirical_distribution import EmpiricalDistribution
from topofm.odes.ode import ODE


def potentials_to_coupling(f: Tensor, g: Tensor, c: Tensor, epsilon: float) -> Tensor:
    """
    Args:
        f: (b, n) potentials.
        g: (b, m) potentials.
        c: (b, n, m) cost matrix.
        epsilon: The entropic regularization parameter.
    Returns:
        pi: (b, n, m) optimal transport plan.
    """
    return torch.softmax((f[..., :, None] + g[..., None, :] - c) / epsilon, dim=-1)


def entropic_ot_coupling(x0: Tensor, x1: Tensor, ode: ODE, epsilon: float) -> Tensor:
    """
    Args:
        x0: (n, d) samples in standard coordinates.
        x1: (m, d) samples in standard coordinates.
        ode: The ODE to use for the cost matrix.
        epsilon: The entropic regularization parameter.

    Returns:
        pi: (n, m) optimal transport plan.
    """
    assert x0.ndim == 2, "x0 must be a 2D tensor."
    assert x1.ndim == 2, "x1 must be a 2D tensor."
    # p=1 to ensure that blur corresponds to epsilon
    f, g = SamplesLoss(blur=epsilon, cost=ode.c, p=1, potentials=True)(x0, x1)
    f, g = f.squeeze(0), g.squeeze(0)
    c = ode.c(x0=x0, x1=x1)
    return potentials_to_coupling(f=f, g=g, c=c, epsilon=epsilon)


def sample_from_coupling(pi: Tensor, mu0: EmpiricalDistribution, mu1: EmpiricalDistribution, shape: Size) -> tuple[Tensor, Tensor]:
    """
    Args:
        pi: (n, m) optimal transport plan.
        mu0: Empirical distribution with (n, d) samples in standard coordinates.
        mu1: Empirical distribution with (m, d) samples in standard coordinates.
        shape: The shape of the sample.
    Returns:
        sample: tuple of (..., d), (..., d)
    """
    assert pi.ndim == 2, "pi must be a 2D tensor."
    assert mu0.samples.ndim == 2, "mu0.samples must be a 2D tensor."
    assert mu1.samples.ndim == 2, "mu1.samples must be a 2D tensor."
    
    num_samples = math.prod(shape)
    
    # sample i from pi0
    i = torch.multinomial(mu0.p, num_samples, replacement=True)

    # sample j from pi conditional on i
    j = torch.multinomial(pi[i], 1, replacement=True).squeeze(-1)

    # Get data
    x0 = mu0.samples[i]
    x1 = mu1.samples[j]

    # Reshape
    x0 = x0.reshape((*shape, mu0.samples.shape[-1]))
    x1 = x1.reshape((*shape, mu1.samples.shape[-1]))

    return x0, x1


class _EntropicOTCoupling(Coupling):
    def __init__(self, mu0: EmpiricalDistribution, mu1: EmpiricalDistribution, ode: ODE, epsilon: float = 0.001):
        super().__init__(mu0=mu0, mu1=mu1)
        with torch.inference_mode():
            self.pi = entropic_ot_coupling(x0=mu0.samples, x1=mu1.samples, ode=ode, epsilon=epsilon)

    def sample(self, shape: Size = Size([])) -> tuple[Tensor, Tensor]:
        return sample_from_coupling(pi=self.pi, mu0=self.mu0, mu1=self.mu1, shape=shape)


class _ExactOTCoupling(Coupling):
    def __init__(self, mu0: EmpiricalDistribution, mu1: EmpiricalDistribution, ode: ODE):
        super().__init__(mu0=mu0, mu1=mu1)
        with torch.inference_mode():
            c = ode.c(x0=mu0.samples, x1=mu1.samples)
        self.pi = emd(a=[], b=[], M=c)

    def sample(self, shape: Size = Size([])) -> tuple[Tensor, Tensor]:
        return sample_from_coupling(pi=self.pi, mu0=self.mu0, mu1=self.mu1, shape=shape)    


class OTCoupling(Coupling):
    def __init__(self, mu0: EmpiricalDistribution, mu1: EmpiricalDistribution, ode: ODE, epsilon: float = 0.0):
        super().__init__(mu0=mu0, mu1=mu1)
        if epsilon == 0.0:
            self._coupling = _ExactOTCoupling(mu0=mu0, mu1=mu1, ode=ode)
        else:
            self._coupling = _EntropicOTCoupling(mu0=mu0, mu1=mu1, ode=ode, epsilon=epsilon)

    def sample(self, shape: Size = Size([])) -> tuple[Tensor, Tensor]:
        return self._coupling.sample(shape)
