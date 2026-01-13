import math
import numpy as np
import torch
from abc import abstractmethod

from .model import ModelMode


def timestep_embedding(timesteps: torch.Tensor, dim: int, max_period: int = 10000) -> torch.Tensor:
    dim_over_2, dim_mod_2 = divmod(dim, 2)
    freqs = torch.exp(-math.log(max_period) * torch.arange(0, dim_over_2, device=timesteps.device, dtype=timesteps.dtype) / dim_over_2)
    args = timesteps.unsqueeze(-1) * freqs.unsqueeze(0)
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim_mod_2 == 1:
        zero_column = embedding.new_zeros(embedding.shape[0], 1)
        embedding = torch.cat([embedding, zero_column], dim=-1)
    return embedding


class FCs(torch.nn.Module):
    def __init__(self, dim_in: int, dim_hid: int, dim_out: int, num_layers: int = 2) -> None:
        super().__init__()
        self.model = torch.nn.Sequential()
        self.model.add_module('fc_in', torch.nn.Linear(dim_in, dim_hid))
        self.model.add_module('relu_in', torch.nn.ReLU())
        for i in range(num_layers - 2):
            self.model.add_module(f'fc_{i}', torch.nn.Linear(dim_hid, dim_hid))
            self.model.add_module(f'relu_{i}', torch.nn.ReLU())
        self.model.add_module('fc_out', torch.nn.Linear(dim_hid, dim_out))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


class ResNet_FC(torch.nn.Module):
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.map(x)
        for res_block in self.res_blocks:
            h = (h + res_block(h)) / np.sqrt(2)
        return h


class TimestepBlock(torch.nn.Module):
    @abstractmethod
    def forward(self, x: torch.Tensor, emb: torch.Tensor) -> torch.Tensor: ...


class TimestepEmbedSequential(torch.nn.Sequential, TimestepBlock):
    def forward(self, x: torch.Tensor, emb: torch.Tensor) -> torch.Tensor:
        for layer in self:
            if isinstance(layer, TimestepBlock):
                x = layer(x, emb)
            else:
                x = layer(x)
        return x


class ResidualNN(torch.nn.Module):
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

    def forward(self, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        t = torch.atleast_1d(t)
        t_emb = timestep_embedding(timesteps=t, dim=self.time_embed_dim)
        t_out = self.t_module(t_emb)
        x_out = self.x_module(x)
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