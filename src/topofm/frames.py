import torch

from .utils import matmul_many


class Frame:
    def transform(self, *args: torch.Tensor) -> torch.Tensor | tuple[torch.Tensor, ...]:
        """Transform tensors into the frame coordinates."""

    def inverse_transform(self, *args: torch.Tensor) -> torch.Tensor | tuple[torch.Tensor, ...]:
        """Inverse transform tensors back to the original coordinates."""


class SpectralFrame(Frame):
    def __init__(
        self, 
        L: torch.Tensor | None = None, 
        eigenvalues: torch.Tensor | None = None, 
        eigenvectors: torch.Tensor | None = None,
    ) -> None:
        super().__init__()
        if L is not None:
            self._eigenvalues, self._eigenvectors = torch.linalg.eigh(L)
        elif eigenvalues is not None and eigenvectors is not None:
            self._eigenvalues, self._eigenvectors = eigenvalues, eigenvectors
        else:
            raise ValueError("Either L or eigenvalues and eigenvectors must be provided")

    def transform(self, *args: torch.Tensor) -> torch.Tensor | tuple[torch.Tensor, ...]:
        return matmul_many(self.eigenvectors.mT, *args)

    def inverse_transform(self, *args: torch.Tensor) -> torch.Tensor | tuple[torch.Tensor, ...]:
        return matmul_many(self.eigenvectors, *args)

    @property
    def eigenvectors(self) -> torch.Tensor:
        return self._eigenvectors

    @property
    def eigenvalues(self) -> torch.Tensor:
        return self._eigenvalues
        

class StandardFrame(Frame):
    def transform(self, *args: torch.Tensor) -> torch.Tensor | tuple[torch.Tensor, ...]:
        return args[0] if len(args) == 1 else args

    def inverse_transform(self, *args: torch.Tensor) -> torch.Tensor | tuple[torch.Tensor, ...]:
        return args[0] if len(args) == 1 else args


