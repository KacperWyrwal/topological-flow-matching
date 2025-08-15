import math
from abc import abstractmethod

import numpy as np
import torch


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
        self.hidden_dim: int = hidden_dim
        self.map: torch.nn.Linear = torch.nn.Linear(data_dim, hidden_dim)
        self.res_blocks: torch.nn.ModuleList = torch.nn.ModuleList([self.build_res_block() for _ in range(num_res_blocks)])

    def build_res_block(self) -> torch.nn.Sequential:
        hid: int = self.hidden_dim
        layers: list[torch.nn.Module] = []
        widths: list[int] = [hid] * 4
        for i in range(len(widths) - 1):
            layers.append(torch.nn.Linear(widths[i], widths[i + 1]))
            layers.append(torch.nn.SiLU())
        return torch.nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.map(x)
        for res_block in self.res_blocks:
            h = (h + res_block(h)) / np.sqrt(2)
        return h

    def device(self) -> torch.device:
        # Check that all parameters are on the same device
        device = next(self.parameters()).device
        for param in self.parameters():
            if param.device != device:
                raise ValueError(f"All parameters should be on the same device.")
        return device


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


class SparseGCNLayer(torch.nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        conv_order: int,
        laplacian: torch.Tensor,
        *,
        aggr_norm: bool = False,
        initialization: str = "xavier_uniform",
    ) -> None:
        assert initialization in ["xavier_uniform", "xavier_normal"]
        super().__init__()
        self.K = conv_order
        self.L = laplacian
        self.aggr_norm = aggr_norm
        self.W = torch.nn.Parameter(torch.empty(in_channels, out_channels, self.K + 1))
        torch.nn.init.xavier_uniform_(self.W, gain=math.sqrt(2))
        self.deg_inv = torch.sparse.sum(self.L, dim=1).to_dense().reciprocal_()
        self.deg_inv[~torch.isfinite(self.deg_inv)] = 0.0

    def _normalize(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.einsum("i,ij->ij ", self.deg_inv, x)
        x[~torch.isfinite(x)] = 0.0
        return x


class GCNLayer(torch.nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        conv_order: int,
        laplacian: torch.Tensor,
        *,
        aggr_norm: bool = False,
        update_func: str | None = None,
        initialization: str = "xavier_uniform",
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.conv_order = conv_order
        self.laplacian = laplacian
        self.aggr_norm = aggr_norm
        self.update_func = update_func
        self.initialization = initialization
        assert initialization in ["xavier_uniform", "xavier_normal"]
        self.weight: torch.nn.Parameter = torch.nn.Parameter(
            torch.Tensor(self.in_channels, self.out_channels, 1 + self.conv_order)
        )
        self.reset_parameters()

    def reset_parameters(self, gain: float = 1.414) -> None:
        if self.initialization == "xavier_uniform":
            torch.nn.init.xavier_uniform_(self.weight, gain=gain)
        elif self.initialization == "xavier_normal":
            torch.nn.init.xavier_normal_(self.weight, gain=gain)

    def aggr_norm_func(self, conv_operator: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        neighborhood_size = torch.sum(conv_operator.to_dense(), dim=1)
        neighborhood_size_inv = 1 / neighborhood_size
        neighborhood_size_inv[~(torch.isfinite(neighborhood_size_inv))] = 0
        x = torch.einsum("i,ij->ij ", neighborhood_size_inv, x)
        x[~torch.isfinite(x)] = 0
        return x

    def update(self, x: torch.Tensor) -> torch.Tensor | None:
        if self.update_func == "sigmoid":
            return torch.sigmoid(x)
        if self.update_func == "relu":
            return torch.nn.functional.relu(x)
        if self.update_func == "id":
            return x
        return None

    def chebyshev_conv(self, conv_operator: torch.Tensor, conv_order: int, x: torch.Tensor) -> torch.Tensor:
        num_simplices, num_channels = x.shape
        X = torch.empty(size=(num_simplices, num_channels, conv_order))
        X[:, :, 0] = torch.mm(conv_operator, x)
        for k in range(1, conv_order):
            X[:, :, k] = torch.mm(conv_operator, X[:, :, k - 1])
            if self.aggr_norm:
                X[:, :, k] = self.aggr_norm_func(conv_operator, X[:, :, k])
        return X

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        num_simplices, _ = x.shape
        x_identity = torch.unsqueeze(x, 2)
        if self.conv_order > 0:
            x = self.chebyshev_conv(self.laplacian, self.conv_order, x)
            x = torch.cat((x_identity, x), 2)
        y = torch.einsum("nik,iok->no", x, self.weight)
        return y


class GCNBlock(torch.nn.Module):
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        laplacian: torch.Tensor,
        *,
        conv_order: int = 1,
        aggr_norm: bool = False,
        update_func: str = None,
        n_layers: int = 2,
    ) -> None:
        super().__init__()
        self.layers: torch.nn.ModuleList = torch.nn.ModuleList(
            [
                GCNLayer(
                    in_channels=in_channels,
                    out_channels=hidden_channels,
                    conv_order=conv_order,
                    laplacian=laplacian,
                )
            ]
        )
        for i in range(n_layers - 1):
            if i == n_layers - 2:
                out_channels = 1
                layer_update_func = 'id'
            else:
                out_channels = hidden_channels
                layer_update_func = 'relu'
            self.layers.append(
                GCNLayer(
                    in_channels=hidden_channels,
                    out_channels=out_channels,
                    conv_order=conv_order,
                    laplacian=laplacian,
                    aggr_norm=aggr_norm,
                    update_func=layer_update_func,
                )
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
        return x


class GCN(torch.nn.Module):
    def __init__(self, laplacian: torch.Tensor, hidden_dim: int = 256, time_embed_dim: int = 128) -> None:
        super().__init__()
        data_dim: int = laplacian.shape[-1]
        self.time_embed_dim: int = time_embed_dim
        hid: int = hidden_dim
        self.t_module: torch.nn.Sequential = torch.nn.Sequential(
            torch.nn.Linear(self.time_embed_dim, hid),
            torch.nn.SiLU(),
            torch.nn.Linear(hid, hid),
        )
        self.x_module1: GCNBlock = GCNBlock(in_channels=1, hidden_channels=hidden_dim, n_layers=2, laplacian=laplacian)
        self.x_module2: ResNet_FC = ResNet_FC(data_dim, hidden_dim, num_res_blocks=0)
        self.out_module: torch.nn.Sequential = torch.nn.Sequential(
            torch.nn.Linear(hid, hid),
            torch.nn.SiLU(),
            torch.nn.Linear(hid, data_dim),
        )

    @property
    def inner_dtype(self) -> torch.dtype:
        # kept for API compatibility; not used internally
        return next(self.parameters()).dtype

    def forward(self, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        if len(t.shape) == 0:
            t = t[None]
        t_emb: torch.Tensor = timestep_embedding(t, self.time_embed_dim)
        t_out: torch.Tensor = self.t_module(t_emb)
        x = x.unsqueeze(-1)
        x_out = torch.empty_like(x)
        for i in range(x.shape[0]):
            x_out[i] = self.x_module1(x[i])
        x_out = x_out.squeeze(-1)
        x_out = self.x_module2(x_out)
        out = self.out_module(x_out + t_out)
        return out


class ResidualNN(torch.nn.Module):
    def __init__(self, data_dim: int, hidden_dim: int = 256, time_embed_dim: int = 128, num_res_block: int = 1) -> None:
        super().__init__()
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


