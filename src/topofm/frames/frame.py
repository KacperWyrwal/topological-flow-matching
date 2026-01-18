import torch
from torch import Tensor
from enum import Enum
from topofm.distributions import Covariance


class Coordinates(Enum):
    STANDARD = 'standard'
    SPECTRAL = 'spectral'


def _change_coordinates_vector(U: Tensor, x: Tensor) -> Tensor:
    """
    Change coordinates using the basis U.

    Args:
        U: (D, E)
        x: (..., D)
    Returns:
        y: (..., E)
    """
    return torch.einsum('de,...d->...e', U, x)


def _change_coordinates_covariance(U: Tensor, cov: Covariance) -> Covariance:
    """
    Change coordinates using the basis U.

    Args:
        U: (D, E)
        cov: (D, D)
    Returns:
        cov: (E, E)
    """
    return Covariance(U @ cov.Sigma @ U.T)


def change_coordinates(U: Tensor, x: Tensor | Covariance) -> Tensor | Covariance:
    """
    Change coordinates using the basis U.

    Args:
        U: (D, E)
        x: (..., D) or (D, D)
    Returns:
        y: (..., E) or (E, E)
    """
    if isinstance(x, Covariance):
        return _change_coordinates_covariance(U, x)
    else:
        return _change_coordinates_vector(U, x)


class Frame:
    def __init__(self, eigvecs: Tensor, coords: Coordinates | str) -> None:
        """
        Args:
            eigvecs: (D, E)
        """
        super().__init__()
        self.eigvecs = eigvecs
        self.coords = Coordinates(coords)

    @property
    def standard_dim(self) -> int:
        return self.eigvecs.shape[0]

    @property
    def spectral_dim(self) -> int:
        return self.eigvecs.shape[1]

    @property
    def dim(self) -> int:
        if self.coords == Coordinates.STANDARD:
            return self.standard_dim
        else:
            return self.spectral_dim

    def standard_to_spectral(self, x: Tensor) -> Tensor:
        """
        Args:
            x: (..., D)
        Returns:
            y: (..., E)
        """
        return change_coordinates(self.eigvecs, x)

    def spectral_to_standard(self, y: Tensor) -> Tensor:
        """
        Args:
            y: (..., E)
        Returns:
            x: (..., D)
        """
        return change_coordinates(self.eigvecs.mT, y)
    
    def to_spectral(self, x: Tensor) -> Tensor:
        """
        Args:
            x: (..., D)
        Returns:
            y: (..., E)
        """
        if self.coords == Coordinates.SPECTRAL:
            return x
        else:
            return self.standard_to_spectral(x)
    
    def from_spectral(self, y: Tensor) -> Tensor:
        """
        Args:
            y: (..., E)
        Returns:
            x: (..., D)
        """
        if self.coords == Coordinates.SPECTRAL:
            return y
        else:
            return self.spectral_to_standard(y)

    def to_standard(self, x: Tensor) -> Tensor:
        """
        Args:
            x: (..., D)
        Returns:
            y: (..., E)
        """
        if self.coords == Coordinates.STANDARD:
            return x
        else:
            return self.spectral_to_standard(x)

    def from_standard(self, x: Tensor) -> Tensor:
        """
        Args:
            x: (..., D)
        Returns:
            y: (..., E)
        """
        if self.coords == Coordinates.STANDARD:
            return x
        else:
            return self.standard_to_spectral(x)
