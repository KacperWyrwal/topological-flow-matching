from abc import ABC
from torch import Size, Tensor, device, dtype


class TimeDistribution(ABC):
    def __init__(self, device: device, dtype: dtype) -> None:
        super().__init__()
        self.device = device
        self.dtype = dtype

    @abstractmethod
    def sample(self, shape: Size) -> Tensor:
        pass
