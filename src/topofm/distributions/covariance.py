import torch
from torch import Tensor
from typing import Literal
from topofm.utils import check_psd, ensure_psd


class Covariance:
    def __init__(self, matrix: Tensor) -> None:
        """
        Args:
            matrix: (D, D) covariance matrix.
        """
        self.matrix, self.eigvals, self.eigvecs = ensure_psd(matrix)

    @classmethod
    def from_samples(
        cls, 
        x: Tensor, 
        mode: Literal['full', 'diagonal','identity'],
    ) -> "Covariance":
        """
        Args:
            x: (N, D) a tensor of samples.
        """
        assert x.ndim == 2, f"x must be 2D, got x.shape={x.shape}"

        if mode == 'full':
            return cls(torch.cov(x.mT))
        elif mode == 'diagonal':
            return cls(torch.diag(torch.var(x, dim=0)))
        elif mode == 'identity':
            return cls(torch.eye(x.shape[1], dtype=x.dtype, device=x.device))
        else:
            raise ValueError(f"Unknown mode: {mode}")
    