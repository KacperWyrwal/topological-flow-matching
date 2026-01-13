from .coupling import Coupling
from torch import Size, Tensor


class IndependentCoupling(Coupling):
    def sample(self, shape: Size) -> tuple[Tensor, Tensor]:
        return self.mu0.sample(shape=shape), self.mu1.sample(shape=shape)