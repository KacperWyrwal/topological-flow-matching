import numpy as np
import torch
from abc import abstractmethod
from torch import nn, Tensor
from topofm.models.model import ModelMode, Model
from topofm.models.timestep_embedding import timestep_embedding


class FCs(nn.Module):
    def __init__(self, dim_in: int, dim_hid: int, dim_out: int, num_layers: int = 2) -> None:
        super().__init__()
        self.model = torch.nn.Sequential()
        self.model.add_module('fc_in', torch.nn.Linear(dim_in, dim_hid))
        self.model.add_module('relu_in', torch.nn.ReLU())
        for i in range(num_layers - 2):
            self.model.add_module(f'fc_{i}', torch.nn.Linear(dim_hid, dim_hid))
            self.model.add_module(f'relu_{i}', torch.nn.ReLU())
        self.model.add_module('fc_out', torch.nn.Linear(dim_hid, dim_out))

    def forward(self, x: Tensor) -> Tensor:
        return self.model(x)


class ResNet_FC(nn.Module):
    def __init__(self, data_dim: int, hidden_dim: int, num_res_blocks: int) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.map = torch.nn.Linear(data_dim, hidden_dim)
        self.res_blocks = torch.nn.ModuleList([self.build_res_block() for _ in range(num_res_blocks)])

    def build_res_block(self) -> torch.nn.Sequential:
        hid = self.hidden_dim
        layers = []
        widths = [hid] * 4
        for i in range(len(widths) - 1):
            layers.append(torch.nn.Linear(widths[i], widths[i + 1]))
            layers.append(torch.nn.SiLU())
        return torch.nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        h = self.map(x)
        for res_block in self.res_blocks:
            h = (h + res_block(h)) / np.sqrt(2)
        return h


class TimestepBlock(nn.Module):
    @abstractmethod
    def forward(self, x: Tensor, emb: Tensor) -> Tensor: ...


class TimestepEmbedSequential(nn.Sequential, TimestepBlock):
    def forward(self, x: Tensor, emb: Tensor) -> Tensor:
        for layer in self:
            if isinstance(layer, TimestepBlock):
                x = layer(x, emb)
            else:
                x = layer(x)
        return x


class ResidualNN(Model):
    def __init__(self, data_dim: int, hidden_dim: int = 256, time_embed_dim: int = 128, num_res_block: int = 1, mode: ModelMode = ModelMode.V) -> None:
        super().__init__(mode=mode)
        self.time_embed_dim = time_embed_dim
        hid = hidden_dim
        self.t_module = torch.nn.Sequential(
            torch.nn.Linear(self.time_embed_dim, hid),
            torch.nn.SiLU(),
            torch.nn.Linear(hid, hid),
        )
        self.x_module = ResNet_FC(data_dim, hidden_dim, num_res_blocks=num_res_block)
        self.out_module = torch.nn.Sequential(
            torch.nn.Linear(hid, hid),
            torch.nn.SiLU(),
            torch.nn.Linear(hid, data_dim),
        )

    def forward(self, t: Tensor, xt: Tensor) -> Tensor:
        """
        Forward pass of the model.

        Args:
            t (Tensor): (n, 1) The time variable.
            xt (Tensor): (n, data_dim) The input variable.

        Returns:
            Tensor: (n, data_dim) The output of the model.
        """
        assert t.ndim == 2 and t.shape[-1] == 1, "t should be of shape (batch_size, 1)"

        t = t.squeeze(-1)
        t_emb = timestep_embedding(timesteps=t, dim=self.time_embed_dim)
        t_out = self.t_module(t_emb)
        x_out = self.x_module(xt)
        out = self.out_module(x_out + t_out)
        return out

    @property
    def device(self) -> torch.device:
        # Check that all parameters are on the same device
        device = next(self.parameters()).device
        for param in self.parameters():
            if param.device != device:
                raise ValueError(f"All parameters should be on the same device.")
        return device