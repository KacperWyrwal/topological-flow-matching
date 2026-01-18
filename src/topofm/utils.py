import math
import numpy as np
import random
import torch
from torch import nn, Tensor
from contextlib import contextmanager


@contextmanager
def preserve_mode(model: nn.Module) -> None:
    """
    Resets the model to its original training/eval mode upon exit.

    Args:
        model: The model to preserve the mode of.
    
    Returns:
        None
    """
    was_training = model.training
    try:
        yield model
    finally:
        model.train(was_training)


def seed_everything(seed: int) -> None:
    """
    Set random seed for torch, numpy, and python.

    Args:
        seed: The seed to set.
    
    Returns:
        None
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def to_dtype(dtype: torch.dtype | str) -> torch.dtype:
    """
    Convert a dtype to a torch.dtype.

    Args:
        dtype: The dtype to convert.
    
    Returns:
        The torch.dtype.
    """
    if isinstance(dtype, str):
        return getattr(torch, dtype)
    return dtype


def is_psd(x: Tensor) -> bool:
    """
    Check if a matrix is positive semi-definite.

    Args:
        x: The matrix to check.
    
    Returns:
        True if the matrix is positive semi-definite, False otherwise.
    """
    _, info = torch.linalg.cholesky_ex(x)
    return info == 0


def sample_moons(shape: torch.Size, *, noise_std: float = 0.05) -> torch.Tensor:
    """Generate samples from the two moons dataset with Gaussian noise.

    Args:
        shape: Desired output shape (excluding the last dimension, which will be 2).
        noise_std: Noise standard deviation.

    Returns:
        Tensor of shape (*shape, 2).
    """
    import sklearn

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


import scipy 


def scipy_csr_to_torch_sparse(csr_matrix: scipy.sparse.csr_matrix, dtype: torch.dtype | None = None, device: torch.device | None = None) -> torch.sparse_coo_tensor:
    """
    Convert a SciPy CSR sparse matrix to a PyTorch sparse COO tensor.
    
    Args:
        csr_matrix (scipy.sparse.csr_matrix): Input CSR matrix.
        dtype (torch.dtype): Desired dtype of the values (default: torch.float32).
        device (str or torch.device): Target device (default: "cpu").
    
    Returns:
        torch.sparse_coo_tensor: Sparse tensor in COO format.
    """
    coo = csr_matrix.tocoo()
    indices = torch.tensor(
        np.vstack((coo.row, coo.col)), dtype=torch.long, device=device
    )
    values = torch.tensor(coo.data, dtype=dtype, device=device)
    shape = coo.shape
    
    return torch.sparse_coo_tensor(indices, values, torch.Size(shape), device=device)


"""
Single-cell utils
"""

def single_cell_to_times(x1: torch.Tensor, true_times: torch.Tensor) -> torch.Tensor:
    return true_times[torch.argsort(x1)]


def single_cell_to_phate(phate: torch.Tensor, times: torch.Tensor, *, t: int = 4) -> torch.Tensor:
    assert phate.ndim == 2, f"phate must be 2D, got {phate.ndim}D"
    assert times.ndim == 1, f"times must be 1D, got {times.ndim}D"
    
    mask = (times == t) # [D]
    return phate[mask] # [D, 2]