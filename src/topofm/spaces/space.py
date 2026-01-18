from abc import ABC
from topofm.frames import Frame, Coordinates
from torch import Tensor


class Space(ABC):

    def __init__(self, eigvals: Tensor, eigvecs: Tensor, coords: Coordinates | str) -> None:
        super().__init__()
        self.frame = Frame(eigvecs=eigvecs, coords=coords)
        self.eigvals = eigvals

    @property
    def dim(self) -> int:
        return self.frame.dim
