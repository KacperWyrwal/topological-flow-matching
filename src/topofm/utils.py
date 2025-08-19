import math
from typing import Tuple

import numpy as np
import sklearn
import torch


def sample_moons(shape: torch.Size, *, noise_std: float = 0.05) -> torch.Tensor:
    """Generate samples from the two moons dataset with Gaussian noise.

    Args:
        shape: Desired output shape (excluding the last dimension, which will be 2).
        noise_std: Noise standard deviation.

    Returns:
        Tensor of shape (*shape, 2).
    """
    n = math.prod(shape)
    x0 = sklearn.datasets.make_moons(n_samples=n, noise=noise_std)[0]
    x0 = torch.as_tensor(x0)
    x0 = x0 - torch.tensor([0.5, 0.25])
    return x0.reshape(*shape, -1)


def sample_eight_gaussians(
    shape: torch.Size, *, radius: float = 2.0, noise_std: float = 0.2
) -> torch.Tensor:
    """Generate samples from a mixture of eight 2D Gaussians arranged on a circle.

    Args:
        shape: Desired output shape (excluding the last dimension, which will be 2).
        radius: Circle radius of centers.
        noise_std: Cluster std.

    Returns:
        Tensor of shape (*shape, 2).
    """
    n = math.prod(shape)
    angles = np.linspace(0, 2 * np.pi, 8, endpoint=False)
    centers = np.column_stack([np.cos(angles), np.sin(angles)]) * radius
    x1 = sklearn.datasets.make_blobs(
        n_samples=n,
        centers=centers,
        cluster_std=noise_std,
    )[0]
    return torch.as_tensor(x1).reshape(*shape, -1)


def matmul_many(matrix: torch.Tensor, *args: torch.Tensor) -> torch.Tensor | tuple[torch.Tensor, ...]:
    """Apply matrix multiplication along the last dimension for many tensors."""
    results = tuple(torch.einsum("ij, ...j -> ...i", matrix, x) for x in args)
    return results[0] if len(results) == 1 else results


def torch_divmod(n: torch.Tensor, d: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """Equivalent to Python divmod but for tensors."""
    return n // d, n % d


def joint_multinomial(
    distribution: torch.Tensor, num_samples: int, replacement: bool = True
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample joint categorical indices from a 2D distribution.

    Args:
        distribution: (N, M) tensor of probabilities.
        num_samples: Number of samples to draw.
        replacement: Sample with replacement.

    Returns:
        Tuple of index tensors (i, j) each of shape (num_samples,).
    """
    n, _ = distribution.shape
    res = torch.multinomial(distribution.flatten(), num_samples, replacement=replacement)
    res_i, res_j = torch_divmod(res, n)
    return res_i, res_j


def as_tensors(
    *args, dtype: torch.dtype | None = None, device: torch.device | None = None
) -> tuple[torch.Tensor, ...]:
    """Convert all arguments to torch tensors."""
    return tuple(torch.as_tensor(arg, dtype=dtype, device=device) for arg in args)


"""
Single-cell utils
"""

def single_cell_to_times(x1: torch.Tensor, true_times: torch.Tensor) -> torch.Tensor:
    return true_times[torch.argsort(x1)]


def single_cell_to_phate(phate: torch.Tensor, times: torch.Tensor, *, t: int = 4) -> torch.Tensor:
    return phate[times == t]