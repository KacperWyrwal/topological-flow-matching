from abc import ABC, abstractmethod
import torch
from torch.distributions import Distribution

from .distributions import Empirical, EmpiricalInFrame
from .ot import OTSolver
from .utils import joint_multinomial


EmpiricalLike = Empirical | EmpiricalInFrame


class Coupling(ABC):
    def __init__(self, mu0: Distribution, mu1: Distribution) -> None:
        self.mu0 = mu0
        self.mu1 = mu1

    @abstractmethod
    def sample(self, shape: torch.Size) -> tuple[torch.Tensor, torch.Tensor]:
        pass


class IndependentCoupling(Coupling):
    def sample(self, shape: torch.Size) -> tuple[torch.Tensor, torch.Tensor]:
        return self.mu0.sample(shape), self.mu1.sample(shape)


class OTCoupling(Coupling):
    def __init__(self, mu0: EmpiricalLike, mu1: EmpiricalLike, ot_solver: OTSolver) -> None:
        super().__init__(mu0, mu1)
        self.x0 = mu0.samples
        self.x1 = mu1.samples
        self.ot_plan = ot_solver.solve(self.x0, self.x1)

    def sample(self, shape: torch.Size) -> tuple[torch.Tensor, torch.Tensor]:
        assert len(shape) == 1, "`OTCoupling` only supports sample shapes (S,)."
        num_samples = shape[0]
        x0_idx, x1_idx = joint_multinomial(self.ot_plan, num_samples=num_samples)
        return self.x0[x0_idx], self.x1[x1_idx]


class OnlineOTCoupling(Coupling):
    def __init__(self, mu0: Distribution, mu1: Distribution, ot_solver: OTSolver) -> None:
        super().__init__(mu0, mu1)
        self.ot_solver = ot_solver

    def sample(self, shape: torch.Size) -> tuple[torch.Tensor, torch.Tensor]:
        assert len(shape) == 1, "`OnlineOTCoupling` only supports sample shapes (S,)."
        x0, x1 = self.mu0.sample(shape), self.mu1.sample(shape)
        num_samples = shape[0]
        ot_plan = self.ot_solver.solve(x0, x1)
        x0_idx, x1_idx = joint_multinomial(ot_plan, num_samples=num_samples)
        return x0[x0_idx], x1[x1_idx]


