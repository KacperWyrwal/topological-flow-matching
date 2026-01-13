import torch
from torch import Tensor
from enum import Enum


class AmbientCoordinates(Enum):
    STANDARD = 'standard'
    SPECTRAL = 'spectral'
    


class Frame:

    def __init__(self, eigvecs: Tensor, ambient: AmbientCoordinates | str) -> None:
        """
        Args:
            eigvecs: (D, E)
        """
        super().__init__()
        self.eigvecs = eigvecs
        self.ambient = AmbientCoordinates(ambient)

    def standard_to_spectral(self, x: Tensor) -> Tensor:
        """
        Args:
            x: (..., D)
        Returns:
            y: (..., E)
        """
        return torch.einsum('de,...d->...e', self.eigvecs, x)

    def spectral_to_standard(self, y: Tensor) -> Tensor:
        """
        Args:
            y: (..., E)
        Returns:
            x: (..., D)
        """
        return torch.einsum('de,...e->...d', self.eigvecs, y)
    
    def ambient_to_spectral(self, x: Tensor) -> Tensor:
        """
        Args:
            x: (..., D)
        Returns:
            y: (..., E)
        """
        if self.ambient == AmbientCoordinates.SPECTRAL:
            return x
        else:
            return self.standard_to_spectral(x)
    
    def spectral_to_ambient(self, y: Tensor) -> Tensor:
        """
        Args:
            y: (..., E)
        Returns:
            x: (..., D)
        """
        if self.ambient == AmbientCoordinates.SPECTRAL:
            return y
        else:
            return self.spectral_to_standard(y)

    def standard_to_ambient(self, x: Tensor) -> Tensor:
        """
        Args:
            x: (..., D)
        Returns:
            y: (..., E)
        """
        if self.ambient == AmbientCoordinates.STANDARD:
            return x
        else:
            return self.standard_to_spectral(x)

    def ambient_to_standard(self, x: Tensor) -> Tensor:
        """
        Args:
            x: (..., D)
        Returns:
            y: (..., E)
        """
        if self.ambient == AmbientCoordinates.STANDARD:
            return x
        else:
            return self.spectral_to_standard(x)
