from ..distributions.boundary import BoundaryDistribution


class FMDataLoader:
    def __init__(
        self, 
        mu0: BoundaryDistribution,
        mu1: BoundaryDistribution,
        num_samples: int,
        batch_size: int,
    ) -> None:
        self.mu0 = mu0
        self.mu1 = mu1
        self.num_samples = num_samples
        self.batch_shape = (batch_size,)
        # Drop the last batch if it is not full.
        self.num_batches = num_samples // batch_size

    def __iter__(self):
        for _ in range(self.num_batches):
            yield self.mu0.sample(self.batch_shape), self.mu1.sample(self.batch_shape)
