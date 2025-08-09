import torch

from .utils import matmul_many


class Frame:
    def transform(self, *args: torch.Tensor) -> torch.Tensor | tuple[torch.Tensor, ...]:
        """Transform tensors into the frame coordinates."""

    def inverse_transform(self, *args: torch.Tensor) -> torch.Tensor | tuple[torch.Tensor, ...]:
        """Inverse transform tensors back to the original coordinates."""


class SpectralFrame(Frame):
    def __init__(self, L: torch.Tensor) -> None:
        super().__init__()
        self.D, self.U = torch.linalg.eigh(L)

    def transform(self, *args: torch.Tensor) -> torch.Tensor | tuple[torch.Tensor, ...]:
        return matmul_many(self.U.mT, *args)

    def inverse_transform(self, *args: torch.Tensor) -> torch.Tensor | tuple[torch.Tensor, ...]:
        return matmul_many(self.U, *args)

    @property
    def eigenvalues(self) -> torch.Tensor:
        return self.D


class StandardFrame(Frame):
    def transform(self, *args: torch.Tensor) -> torch.Tensor | tuple[torch.Tensor, ...]:
        return args[0] if len(args) == 1 else args

    def inverse_transform(self, *args: torch.Tensor) -> torch.Tensor | tuple[torch.Tensor, ...]:
        return args[0] if len(args) == 1 else args


