import torch
from torch import Tensor
from geomloss import SamplesLoss
from geomloss.utils import distances, squared_distances
from ot import emd2


def p_norm(x0: Tensor, x1: Tensor, p: int = 2) -> Tensor:
    if p == 1:
        return distances(x0, x1)
    elif p == 2:
        return squared_distances(x0, x1)
    else:
        raise ValueError(f"Unsupported p-norm: {p}")


def _entropic_ot_distance(x0: Tensor, x1: Tensor, p: int = 2, epsilon: float = 0.0) -> float:
    p_ot_dist = SamplesLoss("sinkhorn", p=p, blur=epsilon)(x0, x1).detach().cpu().item()
    return p_ot_dist ** (1.0 / p)


def _exact_ot_distance(x0: Tensor, x1: Tensor, p: int = 2) -> float:
    M = p_norm(x0, x1, p=p).detach().cpu().numpy()
    p_ot_dist = emd2(a=[], b=[], M=M)
    return p_ot_dist ** (1.0 / p)


def ot_distance(x0: Tensor, x1: Tensor, p: int = 2, epsilon: float = 0.0) -> float:
    """
    Compute the (possibly entropic) OT distance between two sets of points.

    Args:
        x0: (n, d) samples in standard coordinates.
        x1: (m, d) samples in standard coordinates.
        p: The p-norm to use.
        epsilon: The entropic regularization parameter.
    
    Returns:
        The OT distance.
    """
    if epsilon == 0.0:
        return _exact_ot_distance(x0=x0, x1=x1, p=p)
    else:
        return _entropic_ot_distance(x0=x0, x1=x1, p=p, epsilon=epsilon)
    

def wasserstein_distance(x0: Tensor, x1: Tensor, p: int = 2, epsilon: float = 0.0) -> float:
    """
    Compute the (possibly entropic) Wasserstein distance between two sets of points.

    Args:
        x0: (n, d) samples in standard coordinates.
        x1: (m, d) samples in standard coordinates.
        p: The p-norm to use.
        epsilon: The entropic regularization parameter.
    
    Returns:
        The Wasserstein distance.
    """
    return ot_distance(x0=x0, x1=x1, p=p, epsilon=epsilon)