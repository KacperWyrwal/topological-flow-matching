import torch
from torch import Tensor
from topofm.utils import is_psd


class Covariance:
    def __init__(self, matrix: Tensor) -> None:
        """
        Args:
            matrix: (D, D) covariance matrix.
        """
        if not is_psd(matrix):
            raise ValueError("Covariance matrix must be a positive semi-definite matrix")
        self.matrix = matrix

    @classmethod
    def from_samples(cls, x: Tensor, mode: Literal['full', 'identity']) -> "Covariance":
        """
        Args:
            x: (N, D) a tensor of samples.
        """
        assert x.ndim == 2, f"x must be 2D, got x.shape={x.shape}"

        if mode == 'full':
            return cls(torch.cov(x.mT))
        elif mode == 'identity':
            return cls(torch.eye(x.shape[1]))
        else:
            raise ValueError(f"Unknown mode: {mode}")
    