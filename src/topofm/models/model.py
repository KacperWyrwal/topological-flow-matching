from torch import nn, Tensor
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

    def forward(self, t: Tensor, xt: Tensor) -> Tensor:
        """
        Forward pass of the model.

        Args:
            t (Tensor): (batch_size, 1) The time variable.
            xt (Tensor): (batch_size, data_dim) The input variable.

        Returns:
            Tensor: (batch_size, data_dim) The output of the model.
        """
