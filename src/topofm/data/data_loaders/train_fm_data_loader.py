from typing import Iterator
from torch import Tensor
from topofm.couplings.coupling import Coupling


class TrainFMDataLoader:
    def __init__(
        self,
        coupling: Coupling,
        batch_size: int,
        num_samples: int,
    ) -> None:
        self.coupling = coupling
        self.batch_size = batch_size
        self.num_samples = num_samples
        self.num_batches = num_samples // batch_size

    def __iter__(self) -> Iterator[tuple[Tensor, Tensor]]:
        for _ in range(self.num_batches):
            yield self.coupling.sample((self.batch_size,))
        