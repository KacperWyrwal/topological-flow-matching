from torch import Size, Tensor
from topofm.couplings.coupling import Coupling


class IndependentCoupling(Coupling):
    def sample(self, shape: Size = Size([])) -> tuple[Tensor, Tensor]:
        return self.mu0.sample(shape=shape), self.mu1.sample(shape=shape)