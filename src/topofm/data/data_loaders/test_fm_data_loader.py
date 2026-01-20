import logging
from typing import Iterator
from torch import Tensor
from topofm.data.datasets.fm_dataset import FMDataset
from topofm.distributions.boundary.boundary_distribution import BoundaryDistribution
from topofm.distributions.boundary.empirical_distribution import EmpiricalDistribution


logger = logging.getLogger(__name__)


class TestFMDataLoader:
    """Load batches of data from the test set. For empirical distributions, 
    this will load the data in order. For other distributions, sampling is used."""
    def __init__(
        self,
        dataset: FMDataset,
        batch_size: int | None = None,
        num_samples: int | None = None,
    ) -> None:
        logger.info(f"Initializing TestFMDataLoader with dataset={dataset}, batch_size={batch_size}, num_samples={num_samples}")
        # Process num_samples
        if isinstance(dataset.mu1, EmpiricalDistribution) and isinstance(dataset.mu0, EmpiricalDistribution):
            assert dataset.mu1.samples.shape[0] == dataset.mu0.samples.shape[0], (
                "If mu0 and mu1 are both empirical, they must have the same number of samples. "
                f"Got mu0.shape[0]={dataset.mu0.samples.shape[0]} and mu1.shape[0]={dataset.mu1.samples.shape[0]}"
            )
            mu0_num_samples = dataset.mu0.samples.shape[0]
            if num_samples is not None:
                logger.info((
                    "Both mu0 and mu1 are empirical distributions. "
                    f"Ignoring num_samples={num_samples} "
                    f"and using the number of samples in the distributions ({mu0_num_samples})."
                ))
            num_samples = mu0_num_samples
        else:
            assert num_samples is not None, (
                "num_samples must be specified if mu0 and mu1 are not both empirical. "
            )

        # Process batch_size
        if batch_size is None or batch_size > num_samples:
            batch_size = num_samples

        self.mu0 = dataset.mu0
        self.mu1 = dataset.mu1
        self.batch_size = batch_size
        self.num_samples = num_samples
        self.num_batches = num_samples // batch_size

    def _take_batch(self, mu: BoundaryDistribution, batch_num: int) -> Tensor:
        """
        Take a batch of samples from the distribution. If the distribution is empirical, 
        this will load the data in order. For other distributions, sampling is used.
        
        Args:
            mu: The distribution to sample from.
            batch_num: The batch number.
        Returns:
            batch: (batch_size, d)
        """
        if isinstance(mu, EmpiricalDistribution):
            from_idx = batch_num * self.batch_size
            if from_idx >= mu.samples.shape[0]:
                return mu.samples[:0]
            to_idx = (batch_num + 1) * self.batch_size
            return mu.samples[from_idx:to_idx]
        else:
            return mu.sample((self.batch_size,))

    def __iter__(self) -> Iterator[tuple[Tensor, Tensor]]:
        for batch_num in range(self.num_batches):
            x0 = self._take_batch(self.mu0, batch_num)
            x1 = self._take_batch(self.mu1, batch_num)
            yield x0, x1

    def __len__(self) -> int:
        return self.num_batches
