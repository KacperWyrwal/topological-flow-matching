from torch import Size, Tensor
from functools import partial
from topofm.couplings.coupling import Coupling
from topofm.couplings.ot_coupling import OTCoupling
from topofm.distributions.boundary import BoundaryDistribution, EmpiricalDistribution
from topofm.odes.ode import ODE


class BatchwiseOTCoupling(Coupling):
    def __init__(self, mu0: BoundaryDistribution, mu1: BoundaryDistribution, ode: ODE, epsilon: float = 0.001):
        super().__init__(mu0=mu0, mu1=mu1)
        self.ot_coupling = partial(OTCoupling, ode=ode, epsilon=epsilon)

    def sample(self, shape: Size = Size([])) -> tuple[Tensor, Tensor]:
        x0 = self.mu0.sample(shape=shape)
        x1 = self.mu1.sample(shape=shape)
        mu0 = EmpiricalDistribution(samples=x0, frame=self.mu0.frame, device=self.mu0.device, dtype=self.mu0.dtype)
        mu1 = EmpiricalDistribution(samples=x1, frame=self.mu1.frame, device=self.mu1.device, dtype=self.mu1.dtype)
        return self.ot_coupling(mu0=mu0, mu1=mu1).sample(shape=shape)
