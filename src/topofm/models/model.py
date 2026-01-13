from torch import nn
from enum import Enum


class ModelMode(Enum):
    V = "v"
    SV = "sv"


class Model(nn.Module):
    """
    Base class for models parameterizing the flow ODE. 

    Args:
        mode: The mode of the model.
    """
    def __init__(self, mode: ModelMode | str) -> None:
        super().__init__()
        self.mode = ModelMode(mode)
