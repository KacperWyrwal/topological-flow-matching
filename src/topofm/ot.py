import ot
import torch
from geomloss import SamplesLoss

from .sde import SDE


class OTSolver:
    def __init__(self, sde: SDE, *, normalize_variance: bool = False):
        self.sde = sde

        variance_1_given_0 = sde.variance(t=torch.tensor(1.0), t0=torch.tensor(0.0))
        variance_1_given_0_is_degenerate = variance_1_given_0 == 0.0
        if normalize_variance is False:
            epsilon = 0.0
            possibly_scaled_variance_nondegenerate = variance_1_given_0
        else:
            epsilon = (
                torch.prod(variance_1_given_0[~variance_1_given_0_is_degenerate])
                .pow(1 / sde.dim)
                .detach()
                .cpu()
                .item()
            )
            possibly_scaled_variance_nondegenerate = variance_1_given_0 / epsilon

        self.possibly_scaled_std: torch.Tensor = torch.where(
            variance_1_given_0_is_degenerate, 1.0, possibly_scaled_variance_nondegenerate
        ).sqrt()
        self.epsilon: float = epsilon

    def cost(self, x0: torch.Tensor, x1: torch.Tensor) -> torch.Tensor:
        assert x0.ndim >= 2 and x1.ndim >= 2
        x0_expanded = x0.unsqueeze(-2)
        x1_expanded = x1.unsqueeze(-3)
        t0 = x0.new_zeros(x0.shape[:-2] + (x0.shape[-2], 1))
        t1 = x1.new_ones(x0.shape[:-2] + (1, x1.shape[-2]))
        mean = self.sde.mean(t0=t0, x0=x0_expanded, t=t1)
        z = (x1_expanded - mean) / self.possibly_scaled_std
        return torch.linalg.vector_norm(z, ord=2, dim=-1)

    def solve(
        self,
        x0: torch.Tensor,
        x1: torch.Tensor,
        method: str = "sinkhorn",
        num_iter_max: int = 1000,
        stop_threshold: float = 1e-9,
        verbose: bool = False,
        log: bool = False,
        warn: bool = True,
    ) -> torch.Tensor:
        assert x0.ndim >= 2 and x1.ndim >= 2
        cost_matrix = self.cost(x0, x1)
        px0 = x0.new_ones(x0.shape[0]) / x0.shape[0]
        px1 = x1.new_ones(x1.shape[0]) / x1.shape[0]
        cost_matrix = cost_matrix.detach().cpu().numpy()
        px0 = px0.detach().cpu().numpy()
        px1 = px1.detach().cpu().numpy()
        epsilon = self.epsilon
        if epsilon == 0.0:
            optimal_plan = ot.emd(a=px0, b=px1, M=cost_matrix, log=log)
        else:
            optimal_plan = ot.sinkhorn(
                a=px0,
                b=px1,
                M=cost_matrix,
                reg=epsilon,
                method=method,
                num_iter_max=num_iter_max,
                stop_threshold=stop_threshold,
                verbose=verbose,
                log=log,
                warn=warn,
            )
        return torch.as_tensor(optimal_plan, device=x0.device, dtype=x0.dtype)

    @property
    def is_exact(self) -> bool:
        return self.epsilon == 0.0


def wasserstein_distance(
    x0: torch.Tensor, x1: torch.Tensor, *, p: int = 1, blur: float = 0.05
) -> float:
    if p == 1:
        return SamplesLoss(loss="sinkhorn", p=1, blur=blur)(x0, x1)
    if p == 2:
        return SamplesLoss(loss="sinkhorn", p=2, blur=blur)(x0, x1).sqrt()
    raise ValueError("Only p=1 or p=2 supported.")


