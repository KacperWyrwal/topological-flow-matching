import torch
import math
from torch import Tensor


def timestep_embedding(timesteps: Tensor, dim: int, max_period: int = 10000) -> Tensor:
    """
    Timestep embedding taken from Topological Schrodinger Bridge Matching.

    Args:
        timesteps: (...,), Tensor of timesteps
        dim: int, Dimension of the embedding
        max_period: int, Maximum period of the embedding
    Returns:
        (... , dim) Tensor of embeddings
    """
    dim_over_2, dim_mod_2 = divmod(dim, 2)
    freqs = torch.exp(-math.log(max_period) * torch.arange(0, dim_over_2, device=timesteps.device, dtype=timesteps.dtype) / dim_over_2)
    args = timesteps.unsqueeze(-1) * freqs.unsqueeze(0)
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim_mod_2 == 1:
        zero_column = embedding.new_zeros(embedding.shape[0], 1)
        embedding = torch.cat([embedding, zero_column], dim=-1)
    return embedding
