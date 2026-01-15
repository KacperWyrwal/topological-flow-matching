import torch
from torch import Size, Tensor, device, dtype 

from topofm.distributions.time.time_distribution import TimeDistribution


class GridTimeDistribution(TimeDistribution):
    """Samples time from a grid over [0, 1)."""
    def __init__(self, n: int, device: device, dtype: dtype) -> None:
        super().__init__(device=device, dtype=dtype)
        self.grid = torch.linspace(0, 1, n, requires_grad=False, device=device, dtype=dtype)

    def sample(self, shape: Size) -> Tensor:
        """
        Sample from the time distribution.
        
        Args:
            shape: The shape of the sample.
        Returns:
            sample: (shape, 1)
        """
        idx = torch.randint(0, len(self.grid), shape, device=self.device)
        return self.grid[idx].unsqueeze(-1)
