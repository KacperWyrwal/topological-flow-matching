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


class GCNLayer(torch.nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        conv_order: int,
        eigenvalues: torch.Tensor,
        update_func: str = 'relu',
    ) -> None:
        assert conv_order > 0

        super().__init__()
        self.conv_order = conv_order
        self.eigenvalues = eigenvalues
        self.weights = torch.nn.Parameter(
            eigenvalues.new_empty(in_channels, out_channels, 1 + self.conv_order),
        )
        torch.nn.init.xavier_uniform_(self.weights, gain=1.414)

        # set update function
        if update_func == 'relu':
            self.update_func = torch.nn.ReLU()
        elif update_func == 'id':
            self.update_func = torch.nn.Identity()
        else:
            raise ValueError(f"Update function {update_func} not supported")

        # Precompute eigenvalue powers
        powers = torch.arange(1 + self.conv_order, device=self.eigenvalues.device, dtype=self.eigenvalues.dtype)
        self.eigenvalues_pow = torch.pow(self.eigenvalues[:, None], powers) # [D, K]

    def forward(self, y: torch.Tensor) -> torch.Tensor:
        """
        y [..., eigdim, indim]
        """
        weights = self.weights
        eigenvalues_pow = self.eigenvalues_pow
        H = torch.einsum("iok,lk->lio", weights, eigenvalues_pow)      # [L, I, O]
        out = torch.einsum("...li,lio->...lo", y, H) 
        return self.update_func(out)


class GCNBlock(torch.nn.Module):
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        eigenvalues: torch.Tensor,
        *,
        conv_order: int = 1,
        n_layers: int = 2,
    ) -> None:
        super().__init__()
        self.layers = torch.nn.ModuleList()
        # Input layer
        self.layers.append(
            GCNLayer(
                in_channels=in_channels,
                out_channels=hidden_channels,
                conv_order=conv_order,
                eigenvalues=eigenvalues,
                update_func='id',
            )
        )
        # Hidden layers
        for i in range(n_layers - 2):
            self.layers.append(
                GCNLayer(
                    in_channels=hidden_channels,
                    out_channels=hidden_channels,
                    conv_order=conv_order,
                    eigenvalues=eigenvalues,
                    update_func='relu',
                )
            )
        # Output layer
        self.layers.append(
            GCNLayer(
                in_channels=hidden_channels,
                out_channels=1,
                conv_order=conv_order,
                eigenvalues=eigenvalues,
                update_func='id',
            )
        )

    def forward(self, y: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            y = layer(y)
        return y


class GCN(torch.nn.Module):
    def __init__(
        self,
        eigenvalues: torch.Tensor,
        hidden_dim: int = 256,
        time_embed_dim: int = 128,
        n_layers: int = 2,
    ) -> None:
        super().__init__()
        data_dim = eigenvalues.shape[-1]
        self.time_embed_dim = time_embed_dim
        self.t_module = torch.nn.Sequential(
            torch.nn.Linear(self.time_embed_dim, hidden_dim),
            torch.nn.SiLU(),
            torch.nn.Linear(hidden_dim, hidden_dim),
        )
        self.x_module1 = GCNBlock(in_channels=1, hidden_channels=hidden_dim, n_layers=2, eigenvalues=eigenvalues)
        self.x_module2 = ResNet_FC(data_dim, hidden_dim, num_res_blocks=0)
        self.out_module = torch.nn.Sequential(
            torch.nn.Linear(hidden_dim, hidden_dim),
            torch.nn.SiLU(),
            torch.nn.Linear(hidden_dim, data_dim),
        )

    def forward(self, t: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        assert y.ndim > 1
        t = torch.atleast_1d(t)
        y = y.unsqueeze(-1)
        t_emb = timestep_embedding(t, self.time_embed_dim)
        t_out = self.t_module(t_emb)
        y_out = self.x_module1(y)
        y_out = y_out.squeeze(-1)
        y_out = self.x_module2(y_out)
        out = self.out_module(y_out + t_out)
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

    @property
    def device(self) -> torch.device:
        # Check that all parameters are on the same device
        device = next(self.parameters()).device
        for param in self.parameters():
            if param.device != device:
                raise ValueError(f"All parameters should be on the same device.")
        return device



from torch.nn.parameter import Parameter


class SNNLayer(torch.nn.Module):

    def __init__(
        self,
        in_channels,
        out_channels,
        conv_order_down,
        conv_order_up,
        aggr_norm: bool = False,
        update_func=None,
        initialization: str = "xavier_uniform",
    ) -> None:
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.conv_order_down = conv_order_down
        self.conv_order_up = conv_order_up
        self.aggr_norm = aggr_norm
        self.update_func = update_func
        self.initialization = initialization
        assert initialization in ["xavier_uniform", "xavier_normal"]

        self.weight = Parameter(
            torch.Tensor(
                self.in_channels,
                self.out_channels,
                1 + self.conv_order_down + self.conv_order_up,
            )
        )

        self.reset_parameters()

    def reset_parameters(self, gain: float = 1.414) -> None:
       
        if self.initialization == "xavier_uniform":
            torch.nn.init.xavier_uniform_(self.weight, gain=gain)

    def aggr_norm_func(self, conv_operator, x):

        neighborhood_size = torch.sum(conv_operator.to_dense(), dim=1)
        neighborhood_size_inv = 1 / neighborhood_size
        neighborhood_size_inv[~(torch.isfinite(neighborhood_size_inv))] = 0

        x = torch.einsum("i,ij->ij ", neighborhood_size_inv, x)
        x[~torch.isfinite(x)] = 0
        return x

    def update(self, x):
        if self.update_func == "sigmoid":
            return torch.sigmoid(x)
        if self.update_func == "relu":
            return torch.nn.functional.relu(x)
        if self.update_func == "id":
            return x
        
        return None

    def chebyshev_conv(self, conv_operator, conv_order, x):
        num_simplices, num_channels = x.shape
        X = torch.empty(size=(num_simplices, num_channels, conv_order))
        X[:, :, 0] = torch.mm(conv_operator, x)
        for k in range(1, conv_order):
            X[:, :, k] = torch.mm(conv_operator, X[:, :, k - 1])
            if self.aggr_norm:
                X[:, :, k] = self.aggr_norm_func(conv_operator, X[:, :, k])

        return X

    def forward(self, x, laplacian_down, laplacian_up):
        num_simplices, _ = x.shape
        x_identity = torch.unsqueeze(x, 2)

        if self.conv_order_down > 0 and self.conv_order_up > 0:
            x_down = self.chebyshev_conv(laplacian_down, self.conv_order_down, x)
            x_up = self.chebyshev_conv(laplacian_up, self.conv_order_up, x)
            x = torch.cat((x_identity, x_down, x_up), 2)
        elif self.conv_order_down > 0 and self.conv_order_up == 0:
            x_down = self.chebyshev_conv(laplacian_down, self.conv_order_down, x)
            x = torch.cat((x_identity, x_down), 2)
        elif self.conv_order_down == 0 and self.conv_order_up > 0:
            x_up = self.chebyshev_conv(laplacian_up, self.conv_order_up, x)
            x = torch.cat((x_identity, x_up), 2)

        y = torch.einsum("nik,iok->no", x, self.weight)

        if self.update_func is None:
            return y

        return self.update(y)


class SNN(torch.nn.Module):
    def __init__(
        self,
        in_channels,
        hidden_channels,
        conv_order_down=1,
        conv_order_up=1,
        aggr_norm=False,
        update_func=None,
        n_layers=2,
    ):
        super().__init__()
        # First layer -- initial layer has the in_channels as input, and inter_channels as the output
        self.layers = torch.nn.ModuleList(
            [
                SNNLayer(
                    in_channels=in_channels,
                    out_channels=hidden_channels,
                    conv_order_down=conv_order_down,
                    conv_order_up=conv_order_up,
                )
            ]
        )
        for i in range(n_layers - 1):
            if i == n_layers -2: 
                out_channels = 1
                update_func = 'id'
            else: 
                out_channels = hidden_channels
                update_func = 'relu'
            self.layers.append(
                SNNLayer(
                    in_channels=hidden_channels,
                    out_channels=out_channels,
                    conv_order_down=conv_order_down,
                    conv_order_up=conv_order_up,
                    aggr_norm=aggr_norm,
                    update_func=update_func,
                )
            )
            

    def forward(self, x, laplacian_down, laplacian_up):
        
        for layer in self.layers:
            x = layer(x, laplacian_down, laplacian_up)
 

        return x



import torch
import torch.nn as nn
from torch.nn import SiLU

def build_snn(data_dim):
    return SNNPolicy(data_dim)

class SNNPolicy(torch.nn.Module):
    def __init__(self, data_dim, hidden_dim=256, time_embed_dim=128):
        super(SNNPolicy,self).__init__()

        self.time_embed_dim = time_embed_dim
        hid = hidden_dim

        self.t_module = nn.Sequential(
            nn.Linear(self.time_embed_dim, hid),
            SiLU(),
            nn.Linear(hid, hid),
        )
        self.x_module1 = SNN(in_channels=1, hidden_channels=hidden_dim, n_layers=2)
        self.x_module2 = ResNet_FC(data_dim, hidden_dim, num_res_blocks=0)
        self.out_module = nn.Sequential(
            nn.Linear(hid, hid),
            SiLU(),
            nn.Linear(hid, data_dim),
        )

    @property
    def inner_dtype(self):
        """
        Get the dtype used by the torso of the model.
        """
        return next(self.input_blocks.parameters()).dtype

    def forward(self, x, t, lap_down, lap_up):
        """
        Apply the model to an input batch.
        :param x: an [N x C x ...] Tensor of inputs.
        :param timesteps: a 1-D batch of timesteps.
        """
        # make sure t.shape = [T]
        if len(t.shape)==0:
            t=t[None]

        t_emb = timestep_embedding(t, self.time_embed_dim)
        t_out = self.t_module(t_emb)
        x = x.unsqueeze(-1)
        x_out = torch.empty_like(x)

        for i in range(x.shape[0]):
            x_out[i] = self.x_module1(x[i], lap_down, lap_up)
            
        x_out = x_out.squeeze(-1)    
        x_out = self.x_module2(x_out)
        out   = self.out_module(x_out+t_out)
        
        return out
